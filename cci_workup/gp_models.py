import tensorflow as tf
import gpflow
from typing import Optional, Tuple


class SwitchedGPR(gpflow.models.GPModel, gpflow.models.InternalDataTrainingLossMixin):
    """Creates GPR with switched likelihood so can have datasets with different variances.
    Avoids needing to use a VGP, which seems to complicate things.
    Y data will need to have a second column to specify which likelihood to use.
    """
    
    def __init__(
        self,
        data: gpflow.models.model.RegressionData,
        kernel: gpflow.kernels.Kernel,
        mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
        noise_variance: list = [float(1.0)],
    ):
        likelihood = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.Gaussian(variance=v) 
                                                            for v in noise_variance])
        _, Y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1]-1)
        self.data = gpflow.models.util.data_input_to_tensor(data)
        
    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()

    def _add_noise_cov(self, K: tf.Tensor) -> tf.Tensor:
        """
        Returns K + σ² I, where σ² is the likelihood noise variance (scalar),
        and I is the corresponding identity matrix.
        """
        k_diag = tf.linalg.diag_part(K)
        
        ind = tf.cast(self.data[1][..., -1], tf.int32)
        like_vars = [tf.convert_to_tensor(l.variance) for l in self.likelihood.likelihoods]
        s_diag = tf.reshape(tf.gather(like_vars, ind), tf.shape(k_diag))
                
        return tf.linalg.set_diag(K, k_diag + s_diag)

    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        X, Y = self.data
        Y = Y[..., :-1]
        K = self.kernel(X)
        ks = self._add_noise_cov(K)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = gpflow.logdensities.multivariate_normal(Y, m, L)
        return tf.reduce_sum(log_prob)

    def predict_f(
        self, Xnew: gpflow.models.model.InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> gpflow.models.model.MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        X_data, Y_data = self.data
        Y_data = Y_data[..., :-1]
        err = Y_data - self.mean_function(X_data)

        kmm = self.kernel(X_data)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X_data, Xnew)
        kmm_plus_s = self._add_noise_cov(kmm)

        conditional = gpflow.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var
    
    