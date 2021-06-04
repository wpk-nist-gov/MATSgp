import numpy as np
import tensorflow as tf

"""Functions from hapi.py necessary for the pcqsdhc lineshape model, recoded to be
fully compatible with tensorflow.
"""

# ------------------ complex probability function -----------------------
def cpf3(X, Y):
    # define static data
    zone = np.complex128(1.0e0 + 0.0e0j)
    zi = np.complex128(0.0e0 + 1.0e0j)
    tt = np.float64([0.5e0,1.5e0,2.5e0,3.5e0,4.5e0,5.5e0,6.5e0,7.5e0,8.5e0,9.5e0,10.5e0,11.5e0,12.5e0,13.5e0,14.5e0])
    pipwoeronehalf = np.float64(0.564189583547756e0)

    zm1 = zone/tf.complex(X, Y) # maybe redundant
    zm2 = zm1**2
    zsum = zone
    zterm = zone

    for tt_i in tt:
        zterm *= zm2*tt_i
        zsum += zterm
    
    zsum *= zi*zm1*pipwoeronehalf
    
    return zsum

# ------------------ Schreier CPF ------------------------

# "Optimized implementations of rational approximations 
#  for the Voigt and complex error function".
# Franz Schreier. JQSRT 112 (2011) 1010-10250
# doi:10.1016/j.jqsrt.2010.12.010
def cef(x,y,N):
    # Computes the function w(z) = exp(-zA2) erfc(-iz) using a rational
    # series with N terms. It is assumed that Im(z) > 0 or Im(z) = 0.
    z = tf.complex(x, y)
    M = 2*N; M2 = 2*M; k = np.arange(-M+1,M) #'; # M2 = no. of sampling points.
    L = np.sqrt(N/np.sqrt(2)); # Optimal choice of L.
    theta = k*np.pi/M; t = L*np.tan(theta/2); # Variables theta and t.
    #f = exp(-t.A2)*(LA2+t.A2); f = [0; f]; # Function to be transformed.
    f = np.zeros(len(t)+1); f[0] = 0
    f[1:] = np.exp(-t**2)*(L**2+t**2)
    #f = insert(exp(-t**2)*(L**2+t**2),0,0)
    a = np.real(np.fft.fft(np.fft.fftshift(f)))/M2; # Coefficients of transform.
    a = np.flipud(a[1:N+1]); # Reorder coefficients.
    Z = (L+1.0j*z)/(L-1.0j*z)
    p = tf.math.polyval([tf.cast(val, tf.complex128) for val in a.tolist()], Z); # Polynomial evaluation.
    #Not sure if above line will work with Tensorflow...
    #Need to check because may fail quietly, i.e. break gradient without throwing error
    #If can figure out what this whole function is supposed to be doing, though, may be able to just
    #directly return w(z)... just need to know what zA2 and iz are...
    #Hopefully using tf.math.polyval instead of np.polyval will do the trick.
    w = 2*p/(L-1.0j*z)**2+(1/np.sqrt(np.pi))/(L-1.0j*z); # Evaluate w(z).
    return w
# weideman24 by default    
#weideman24 = lambda x,y: cef(x,y,24)
#weideman = lambda x,y,n: cef(x,y,n)
def hum1_wei(x,y,n=24):
    """
    z = x+1j*y
    cerf = 1j*z/sqrt(pi)/(z**2-0.5)
    """
    mask = abs(x)+y<15.0
    t = tf.complex(y, -x)
    #cerf_false = 1/np.sqrt(np.pi)*t/(0.5+t**2)
    #if np.any(mask):
    #    w24 = weideman(x[mask],y[mask],n)
    #    cerf = tf.dynamics_stitch([np.arange(x.shape[0])[~mask], np.arange(x.shape[0])[mask]],
    #                              [cerf_false[~mask], w24])
    #else:
    #    cerf = cerf_false
    #Above is fancy, but seems like below should just work...
    #Because in original, output of weideman() is same length as input, which is x and y where mask is True
    #So first N values of output should be weideman() of all x and y then with mask applied
    #For efficiency, still check for any being True
    if tf.reduce_any(mask):
        cerf = tf.where(mask, cef(x, y, n), 1/np.sqrt(np.pi)*t/(0.5+t**2))
    else:
        cerf = 1/np.sqrt(np.pi)*t/(0.5+t**2)
    return cerf

# ------------------ Hartmann-Tran Profile (HTP) ------------------------
# Made compatible with Tensorflow
def tf_pcqsdhc(sg0,GamD,Gam0,Gam2,Shift0,Shift2,anuVC,eta,sg):
    #-------------------------------------------------
    #      "pCqSDHC": partially-Correlated quadratic-Speed-Dependent Hard-Collision
    #      Subroutine to Compute the complex normalized spectral shape of an 
    #      isolated line by the pCqSDHC model
    #
    #      Reference:
    #      H. Tran, N.H. Ngo, J.-M. Hartmann.
    #      Efficient computation of some speed-dependent isolated line profiles.
    #      JQSRT, Volume 129, November 2013, Pages 199–203
    #      http://dx.doi.org/10.1016/j.jqsrt.2013.06.015
    #
    #      Input/Output Parameters of Routine (Arguments or Common)
    #      ---------------------------------
    #      T          : Temperature in Kelvin (Input).
    #      amM1       : Molar mass of the absorber in g/mol(Input).
    #      sg0        : Unperturbed line position in cm-1 (Input).
    #      GamD       : Doppler HWHM in cm-1 (Input)
    #      Gam0       : Speed-averaged line-width in cm-1 (Input).       
    #      Gam2       : Speed dependence of the line-width in cm-1 (Input).
    #      anuVC      : Velocity-changing frequency in cm-1 (Input).
    #      eta        : Correlation parameter, No unit (Input).
    #      Shift0     : Speed-averaged line-shift in cm-1 (Input).
    #      Shift2     : Speed dependence of the line-shift in cm-1 (Input)       
    #      sg         : Current WaveNumber of the Computation in cm-1 (Input).
    #
    #      Output Quantities (through Common Statements)
    #      -----------------
    #      LS_pCqSDHC_R: Real part of the normalized spectral shape (cm)
    #      LS_pCqSDHC_I: Imaginary part of the normalized spectral shape (cm)
    #
    #      Called Routines: 'CPF'      (Complex Probability Function)
    #      ---------------  'CPF3'      (Complex Probability Function for the region 3)
    #
    #      Called By: Main Program
    #      ---------
    #
    #     Double Precision Version
    #
    #-------------------------------------------------
    
    # sg is the only vector argument which is passed to function
    # Just make sure it's a vector
    sg = tf.constant(sg)
    
    #With Tensorflow, can't assign values within tensors, so must stitch together at end
    #number_of_points = sg.shape[0]
    #Aterm_GLOBAL = np.zeros(number_of_points, dtype=tf.complex128)
    #Bterm_GLOBAL = np.zeros(number_of_points, dtype=tf.complex128)
    
    #Multiplying by tensor of ones to make sure shape is right even if just provide scalars as parameters
    cte = tf.ones_like(sg, dtype=tf.complex128) * np.sqrt(np.log(2.0e0))/tf.cast(GamD, tf.complex128)
    rpi = np.sqrt(np.pi)

    c0 = tf.ones_like(sg, dtype=tf.complex128) * tf.complex(Gam0, Shift0)
    c2 = tf.ones_like(sg, dtype=tf.complex128) * tf.complex(Gam2, Shift2)
    c0t = tf.ones_like(sg, dtype=tf.complex128) * (1.0e0 - tf.cast(eta, tf.complex128)) * (c0 - 1.5e0 * c2) + tf.cast(anuVC, tf.complex128)
    c2t = tf.ones_like(sg, dtype=tf.complex128) * (1.0e0 - tf.cast(eta, tf.complex128)) * c2
        
    #In Tensorflow cannot do dynamic assignment, so keep track of indices and merge together at end
    Aterm_PART1 = tf.constant([], dtype=tf.complex128)
    Bterm_PART1 = tf.constant([], dtype=tf.complex128)
    merge_inds_PART1 = tf.constant([], dtype=tf.int32)
    Aterm_PART2 = tf.constant([], dtype=tf.complex128)
    Bterm_PART2 = tf.constant([], dtype=tf.complex128)
    merge_inds_PART2 = tf.constant([], dtype=tf.int32)
    Aterm_PART3 = tf.constant([], dtype=tf.complex128)
    Bterm_PART3 = tf.constant([], dtype=tf.complex128)
    merge_inds_PART3 = tf.constant([], dtype=tf.int32)
    Aterm_PART4 = tf.constant([], dtype=tf.complex128)
    Bterm_PART4 = tf.constant([], dtype=tf.complex128)
    merge_inds_PART4 = tf.constant([], dtype=tf.int32)
    
    # PART1
    index_PART1 = (abs(c2t) == 0.0e0)
    if tf.reduce_any(index_PART1):
        #print('Executing Part 1')
        Z1 = ((tf.complex(tf.zeros_like(sg), sg0 - sg) + c0t) * cte)[index_PART1]
        xZ1 = -tf.math.imag(Z1)
        yZ1 = tf.math.real(Z1)
        W1 = hum1_wei(xZ1,yZ1)
        Aterm = rpi*cte[index_PART1]*(W1)
        index_Z1 = (abs(Z1) <= 4.0e3)
        index_NOT_Z1 = ~index_Z1
        #if tf.reduce_any(index_Z1):
        if tf.reduce_all(index_Z1):
            Bterm = rpi*cte[index_PART1]*((1.0e0 - Z1**2)*(W1) + Z1/rpi)
        #Original in hapi.py was another if statement... if any(index_NOT_Z1)
        #Unless index_Z1 is all true, that overwrites everything
        #Not sure if that is the intended behavior, but if it is, accomplish by changing above to if all(index_Z1)
        #and changing below to just else
        #if tf.reduce_any(index_NOT_Z1): 
        else:
            Bterm = cte[index_PART1]*(rpi*(W1) + 0.5e0/Z1 - 0.75e0/(Z1**3))
        merge_inds_PART1 = tf.range(sg.shape[0])[index_PART1]
        Aterm_PART1 = Aterm
        Bterm_PART1 = Bterm

    # PART2, PART3 AND PART4   (PART4 IS A MAIN PART)
    # X - vector, Y - scalar
    X = (tf.complex(tf.zeros_like(sg), sg0 - sg) + c0t) / c2t
    Y = (1.0e0 / ((2.0e0*cte*c2t))**2)
    csqrtY = tf.complex(Gam2, -Shift2) / (2.0e0*cte*(1.0e0-tf.cast(eta, tf.complex128)) * tf.cast(Gam2**2 + Shift2**2, tf.complex128))
        
    index_PART2 = ((abs(X) <= 3.0e-8 * abs(Y)) & ~index_PART1)
    index_PART3 = ((abs(Y) <= 1.0e-15 * abs(X)) & ~index_PART2 & ~index_PART1)
    index_PART4 = (~(index_PART2 | index_PART3) & ~index_PART1)
        
    # PART4
    if tf.reduce_any(index_PART4):
        #print('Executing Part 4')
        #X_TMP = X[index_PART4]
        Z1 = (tf.sqrt(X + Y) - csqrtY)[index_PART4]
        Z2 = Z1 + 2.0e0 * csqrtY[index_PART4]
        xZ1 = -tf.math.imag(Z1)
        yZ1 =  tf.math.real(Z1)
        xZ2 = -tf.math.imag(Z2)
        yZ2 =  tf.math.real(Z2)
        SZ1 = tf.sqrt(xZ1**2 + yZ1**2)
        SZ2 = tf.sqrt(xZ2**2 + yZ2**2)
        DSZ = tf.abs(SZ1 - SZ2)
        SZmx = tf.maximum(SZ1,SZ2)
        SZmn = tf.minimum(SZ1,SZ2)
        index_CPF3 = ((DSZ <= 1.0e0) & (SZmx > 8.0e0) & (SZmn <= 8.0e0))
        W1_PART4 = tf.where(index_CPF3, cpf3(xZ1, yZ1), hum1_wei(xZ1, yZ1))
        W2_PART4 = tf.where(index_CPF3, cpf3(xZ2, yZ2), hum1_wei(xZ2, yZ2))
        Aterm = rpi*cte[index_PART4]*((W1_PART4) - (W2_PART4))
        Bterm = (-1.0e0 +
                  rpi/(2.0e0*csqrtY[index_PART4])*(1.0e0 - Z1**2)*(W1_PART4)-
                  rpi/(2.0e0*csqrtY[index_PART4])*(1.0e0 - Z2**2)*(W2_PART4)) / c2t[index_PART4]
        merge_inds_PART4 = tf.range(sg.shape[0])[index_PART4]
        Aterm_PART4 = Aterm
        Bterm_PART4 = Bterm

    # PART2
    if tf.reduce_any(index_PART2):
        #print('Executing Part 2')
        #X_TMP = X[index_PART2]
        Z1 = ((tf.complex(tf.zeros_like(sg), sg0 - sg) + c0t) * cte)[index_PART2]
        Z2 = (tf.sqrt(X + Y) + csqrtY)[index_PART2]
        xZ1 = -tf.math.imag(Z1)
        yZ1 = tf.math.real(Z1)
        xZ2 = -tf.math.imag(Z2)
        yZ2 = tf.math.real(Z2)
        W1_PART2 = hum1_wei(xZ1,yZ1)
        W2_PART2 = hum1_wei(xZ2,yZ2) 
        Aterm = rpi*cte[index_PART2]*((W1_PART2) - (W2_PART2))
        Bterm = (-1.0e0 +
                  rpi/(2.0e0*csqrtY[index_PART2])*(1.0e0 - Z1**2)*(W1_PART2)-
                  rpi/(2.0e0*csqrtY[index_PART2])*(1.0e0 - Z2**2)*(W2_PART2)) / c2t[index_PART2]
        merge_inds_PART2 = tf.range(sg.shape[0])[index_PART2]
        Aterm_PART2 = Aterm
        Bterm_PART2 = Bterm
            
    # PART3
    if tf.reduce_any(index_PART3):
        #print('Executing Part 3')
        X_TMP = X[index_PART3]
        Z1 = tf.sqrt(X + Y)[index_PART3]
        xZ1 = -tf.math.imag(Z1)
        yZ1 = tf.math.real(Z1)
        W1_PART3 =  hum1_wei(xZ1,yZ1) 
        index_ABS = (tf.abs(tf.sqrt(X_TMP)) <= 4.0e3)
        Wb = hum1_wei(-tf.math.imag(tf.sqrt(X_TMP)), tf.math.real(tf.sqrt(X_TMP))) #Original just had X, not X_TMP, which would break below (multiplication by Wb)
        Aterm = tf.where(index_ABS,
                         (2.0e0*rpi/c2t[index_PART3])*(1.0e0/rpi - tf.sqrt(X_TMP)*(Wb)),
                         (1.0e0/c2t[index_PART3])*(1.0e0/X_TMP - 1.5e0/(X_TMP**2))
                        )
        Bterm = tf.where(index_ABS,
                         (1.0e0/c2t[index_PART3])*(-1.0e0+
                                      2.0e0*rpi*(1.0e0 - X_TMP-2.0e0*Y[index_PART3])*(1.0e0/rpi-tf.sqrt(X_TMP)*(Wb))+
                                      2.0e0*rpi*tf.sqrt(X_TMP + Y[index_PART3])*(W1_PART3)),
                         (1.0e0/c2t[index_PART3])*(-1.0e0 + (1.0e0 - X_TMP - 2.0e0*Y[index_PART3])*
                                      (1.0e0/X_TMP - 1.5e0/(X_TMP**2))+
                                      #2.0e0*rpi*sqrt(X_TMP + Y)*(W1)) #original, but would fail b/c W1 out of scope
                                      2.0e0*rpi*tf.sqrt(X_TMP + Y[index_PART3])*(W1_PART3))
                        )
        merge_inds_PART3 = tf.range(sg.shape[0])[index_PART3]
        Aterm_PART3 = Aterm
        Bterm_PART3 = Bterm
    
    Aterm_GLOBAL = tf.dynamic_stitch([merge_inds_PART1, merge_inds_PART2, merge_inds_PART3, merge_inds_PART4],
                                     [Aterm_PART1, Aterm_PART2, Aterm_PART3, Aterm_PART4])
    Bterm_GLOBAL = tf.dynamic_stitch([merge_inds_PART1, merge_inds_PART2, merge_inds_PART3, merge_inds_PART4],
                                     [Bterm_PART1, Bterm_PART2, Bterm_PART3, Bterm_PART4])
            
    # common part
    LS_pCqSDHC = (1.0e0/np.pi) * (Aterm_GLOBAL / (1.0e0 - (tf.cast(anuVC, tf.complex128)-tf.cast(eta, tf.complex128)*(c0-1.5e0*c2))*Aterm_GLOBAL + tf.cast(eta, tf.complex128)*c2*Bterm_GLOBAL))
    #LS_pCqSDHC = tf.reshape(LS_pCqSDHC, (-1, 1))
    return tf.math.real(LS_pCqSDHC), tf.math.imag(LS_pCqSDHC)
