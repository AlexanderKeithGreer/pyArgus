# -*- coding: utf-8 -*-
"""
                                                   PyArgus
                                            Direction Estimation


     Description:
     ------------
       Implements Direction of Arrival estimation methods for antenna arrays.

        Implemented DOA methods:

            - Bartlett method
            - Capon's method
            - Burg's Maximum Entropy Method (MEM)
            - Multiple Signal Classification (MUSIC)
            - Multi Dimension MUSIC (MD-MUSIC)
            - Iterative Adaptive Aproach for Amplitude and Phase Estimation (IAA-APES)
            - Iterative Sparse Asymptotic Minimum Variance (SAMV)

        Corr matrix estimation functions:
            - Sample Matrix Inversion (SMI)
            - Froward-Backward averaging
            - Spatial Smoothing


     Authors: Tamás Pető

     License: GPLv3

     Changelog :
         - Ver 1.0000    : Initial version (2016 12 26)
         - Ver 1.1000    : Reformated code (2017 06 02)
	     - Ver 1.1001    : Improved documentation and comments (2018 02 21)
         - Ver 1.1500    : Algorithms now expects scanning vector matrix insted of array alignment
                           to support more generic anntenna alignments (2018 09 01)
         - Ver 1.1600    : Added two iterative algorithms (for many correlated sources), and an
                           example comparing them to spatial smoothing (2020 07 04)


"""

import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt

def DOA_Bartlett(R, scanning_vectors):
    """
                    Fourier(Bartlett) - DIRECTION OF ARRIVAL ESTIMATION



        Description:
        ------------
           The function implements the Bartlett method for direction estimation

           Calculation method :
		                                                  H
		                PAD(theta) = S(theta) * R_xx * S(theta)


        Parameters:
        -----------

            :param R: spatial correlation matrix
            :param scanning_vectors : Generated using the array alignment and the incident angles

            :type R: 2D numpy array with size of M x M, where M is the number of antennas in the antenna system
            :tpye scanning vectors: 2D numpy array with size: M x P, where P is the number of incident angles

       Return values:
       --------------

            :return PAD: Angular distribution of the power ("Power angular densitiy"- not normalized to 1 deg)
	        :rtype PAD: numpy array

            :return -1, -1: Input spatial correlation matrix is not quadratic
            :return -2, -2: dimension of R not equal with dimension of the antenna array

"""

    # --- Parameters ---

    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1

    if np.size(R, 0) != np.size(scanning_vectors, 0):
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return -2, -2

    PAD = np.zeros(np.size(scanning_vectors, 1),dtype=complex)

    # --- Calculation ---
    theta_index=0
    for i in range(np.size(scanning_vectors, 1)):
        S_theta_ = scanning_vectors[:, i]
        PAD[theta_index]=np.dot(np.conj(S_theta_),np.dot(R,S_theta_))
        theta_index += 1

    return PAD

def DOA_Capon(R, scanning_vectors):
    """
                    Capon's method - DIRECTION OF ARRIVAL ESTIMATION



        Description:
        ------------
            The function implements Capon's direction of arrival estimation method

            Calculation method :

                                                  1
                          SINR(theta) = ---------------------------
                                            H        -1
                                     S(theta) * R_xx * S(theta)

        Parameters:
        -----------
            :param R: spatial correlation matrix
            :param scanning_vectors : Generated using the array alignment and the incident angles

            :type R: 2D numpy array with size of M x M, where M is the number of antennas in the antenna system
            :tpye scanning vectors: 2D numpy array with size: M x P, where P is the number of incident angles

       Return values:
       --------------

            :return ADSINR:  Angular dependenet signal to noise ratio
	        :rtype ADSINR: numpy array

            :return -1, -1: Input spatial correlation matrix is not quadratic
            :return -2, -2: dimension of R not equal with dimension of the antenna array
            :return -3, -3: Spatial correlation matrix is singular
    """
    # --- Parameters ---

    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1
    if np.size(R, 0) != np.size(scanning_vectors, 0):
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return -2, -2


    ADSINR = np.zeros(np.size(scanning_vectors, 1),dtype=complex)

    # --- Calculation ---
    try:
        R_inv  = np.linalg.inv(R) # invert the cross correlation matrix
    except:
        print("ERROR: Signular matrix")
        return -3, -3

    theta_index=0
    for i in range(np.size(scanning_vectors, 1)):
        S_theta_ = scanning_vectors[:, i]
        ADSINR[theta_index]=np.dot(np.conj(S_theta_),np.dot(R_inv,S_theta_))
        theta_index += 1

    ADSINR = np.reciprocal(ADSINR)

    return ADSINR


def DOA_MEM(R, scanning_vectors, column_select = 0 ):
    """
                    Maximum Entropy Method - DIRECTION OF ARRIVAL ESTIMATION



        Description:
         ------------
            The function implements the MEM method for direction estimation


            Calculation method :

                                                  1
                        PAD(theta) = ---------------------------
                                             H        H
                                      S(theta) * rj rj  * S(theta)
        Parameters:
        -----------
            :param R: spatial correlation matrix
            :param scanning_vectors : Generated using the array alignment and the incident angles
            :param column_select: Selects the column of the R matrix used in the MEM algorithm (default : 0)

            :type R: 2D numpy array with size of M x M, where M is the number of antennas in the antenna system
            :tpye scanning vectors: 2D numpy array with size: M x P, where P is the number of incident angles
            :type column_select: int

       Return values:
       --------------

            :return PAD: Angular distribution of the power ("Power angular densitiy"- not normalized to 1 deg)
	        :rtype : numpy array

            :return -1, -1: Input spatial correlation matrix is not quadratic
            :return -2, -2: dimension of R not equal with dimension of the antenna array
            :return -3, -3: Spatial correlation matrix is singular
    """
    # --- Parameters ---

    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1

    if np.size(R, 0) != np.size(scanning_vectors, 0):
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return -2, -2

    PAD = np.zeros(np.size(scanning_vectors,1),dtype=complex)

    # --- Calculation ---
    try:
        R_inv  = np.linalg.inv(R) # invert the cross correlation matrix
    except:
        print("ERROR: Signular matrix")
        return -3, -3

    # Create matrix from one of the column of the cross correlation matrix with
    # dyadic multiplication
    R_invc = np.outer( R_inv [:,column_select],np.conj(R_inv[:,column_select]))

    theta_index=0
    for i in range(np.size(scanning_vectors,1)):
        S_theta_ = scanning_vectors[:, i]
        PAD[theta_index]=np.dot(np.conj(S_theta_),np.dot(R_invc,S_theta_))
        theta_index += 1

    PAD = np.reciprocal(PAD)

    return PAD


def DOA_LPM(R, scanning_vectors, element_select, angle_resolution = 1):
    """
                    LPM - Linear Prediction method



        Description:
         ------------
           The function implements the Linear prediction method for direction estimation

           Calculation method :
                                                  H    -1
                                                 U    R    U
                        PLP(theta) = ---------------------------
                                          |    H   -1           |2
                                          |   U * R  * S(theta) |


        Parameters:
        -----------
            :param R: spatial correlation matrix
            :param scanning_vectors : Generated using the array alignment and the incident angles
            :param element_select: Antenna element index used for the predection.
            :param angle_resolution: Angle resolution of scanning vector s(theta) [deg] (default : 1)

            :type R: 2D numpy array with size of M x M, where M is the number of antennas in the antenna system
            :tpye scanning vectors: 2D numpy array with size: M x P, where P is the number of incident angles
            :type element_select: int
            :type angle_resolution: float

       Return values:
       --------------

            :return PLP : Angular distribution of the power ("Power angular densitiy"- not normalized to 1 deg)
	        :rtype : numpy array

            :return -1, -1: Input spatial correlation matrix is not quadratic
            :return -2, -2: dimension of R not equal with dimension of the antenna array
            :return -3, -3: Spatial correlation matrix is singular
    """
    # --- Parameters ---

    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1

    if np.size(R, 0) != np.size(scanning_vectors, 0):
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return -2, -2

    PLP = np.zeros(np.size(scanning_vectors,1),dtype=complex)

    # --- Calculation ---
    try:
        R_inv  = np.linalg.inv(R) # invert the cross correlation matrix
    except:
        print("ERROR: Signular matrix")
        return -3, -3

    R_inv = np.matrix(R_inv)
    M = np.size(scanning_vectors,0)

    # Create element selector vector
    u = np.zeros(M,dtype=complex)
    u[element_select] = 1
    u = np.matrix(u).getT()

    theta_index=0
    for i in range(np.size(scanning_vectors,1)):
        S_theta_ = scanning_vectors[:, i]
        S_theta_ = np.matrix(S_theta_).getT()
        PLP[theta_index]=  np.real(u.getH() * R_inv * u) / np.abs(u.getH()* R_inv * S_theta_)**2
        theta_index += 1

    return PLP



def DOA_MUSIC(R, scanning_vectors, signal_dimension, angle_resolution = 1):
    """
                    MUSIC - Multiple Signal Classification method



        Description:
         ------------
           The function implements the MUSIC method for direction estimation

           Calculation method :

                                                    1
                        ADORT(theta) = ---------------------------
                                             H        H
                                      S(theta) * En En  * S(theta)
         Parameters:
        -----------
            :param R: spatial correlation matrix
            :param scanning_vectors : Generated using the array alignment and the incident angles
            :param signal_dimension:  Number of signal sources

            :type R: 2D numpy array with size of M x M, where M is the number of antennas in the antenna system
            :tpye scanning vectors: 2D numpy array with size: M x P, where P is the number of incident angles
            :type signal_dimension: int

       Return values:
       --------------

            :return  ADORT : Angular dependent orthogonality. Expresses the orthongonality of the current steering vector to the
                    noise subspace
            :rtype : numpy array

            :return -1, -1: Input spatial correlation matrix is not quadratic
            :return -2, -2: dimension of R not equal with dimension of the antenna array
            :return -3, -3: Spatial correlation matrix is singular
    """
    # --- Parameters ---

    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1

    if np.size(R, 0) != np.size(scanning_vectors, 0):
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return -2, -2

    ADORT = np.zeros(np.size(scanning_vectors, 1),dtype=complex)
    M = np.size(R, 0)

    # --- Calculation ---
    # Determine eigenvectors and eigenvalues
    sigmai, vi = lin.eig(R)
    # Sorting
    eig_array = []
    for i in range(M):
        eig_array.append([np.abs(sigmai[i]),vi[:,i]])
    eig_array = sorted(eig_array, key=lambda eig_array: eig_array[0], reverse=False)

    # Generate noise subspace matrix
    noise_dimension = M - signal_dimension
    E = np.zeros((M,noise_dimension),dtype=complex)
    for i in range(noise_dimension):
        E[:,i] = eig_array[i][1]

    E = np.matrix(E)

    theta_index=0
    for i in range(np.size(scanning_vectors, 1)):
        S_theta_ = scanning_vectors[:, i]
        S_theta_  = np.matrix(S_theta_).getT()
        ADORT[theta_index]=  1/np.abs(S_theta_.getH()*(E*E.getH())*S_theta_)
        theta_index += 1

    return ADORT

def DOAMD_MUSIC(R, array_alignment, signal_dimension, coherent_sources=2, angle_resolution = 1,):
    """
                    MD-MUSIC - Multi Dimensional Multiple Signal Classification method



         Description:
         ------------
           The function implements the MD-MUSIC method for direction estimation

           Calculation method :

                                                    1
                        ADORT(theta) = ---------------------------
                                            H H       H
                                           A*c * En En  * A c

                        A  - Array response matrix
                        C  - Liner combiner vector
                        En - Noise subspace matrix

        Implementation notes:
        ---------------------

            This function works only for two coherent signal sources. Note that, however the algorithm works
            for arbitrary number of coherent sources, the computational cost increases exponentially, thus
            using this algorithm for higher number of sources is impractical.

        Parameters:
        -----------

            :param R: spatial correlation matrix
            :param array_alignment : Array containing the antenna positions measured in the wavelength
            :param signal_dimension: Number of signal sources
            :param coherent_sources: Number of coherent sources
            :param angle_resolution: Angle resolution of scanning vector s(theta) [deg] (default : 1)

            :type R: 2D numpy array with size of M x M, where M is the number of antennas in the antenna system
            :tpye array_alignment: 1D numpy array with size: M x 1
            :type signal_dimension: int
            :type: coherent_sources: int
            :type angle_resolution: float

       Return values:
       --------------

            :return  ADORT : Angular dependent orthogonality. Expresses the orthongonality of the current steering vector to the
                    noise subspace
            :rtype : L dimensional numpy array, where L is the number of coherent sources

            :return -1, -1: Input spatial correlation matrix is not quadratic
            :return -2, -2: dimension of R not equal with dimension of the antenna array

    """

    # --- Parameters ---

    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1

    if np.size(R, 0) != np.size(array_alignment, 0):
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return -2, -2

    incident_angles = np.arange(0,180+angle_resolution,angle_resolution)
    ADORT = np.zeros((int(180/angle_resolution+1), int(180/angle_resolution+1)), dtype=float)

    M = np.size(R, 0) # Number of antenna elements

    # --- Calculation ---
    # Determine eigenvectors and eigenvalues
    sigmai, vi = lin.eig(R)
    # Sorting
    eig_array = []
    for i in range(M):
        eig_array.append([np.abs(sigmai[i]),vi[:,i]])
    eig_array = sorted(eig_array, key=lambda eig_array: eig_array[0],
                       reverse=False)

    # Generate noise subspace matrix
    noise_dimension = M - signal_dimension
    E = np.zeros((M,noise_dimension),dtype=complex)
    for i in range(noise_dimension):
        E[:,i] = eig_array[i][1]

    E = np.matrix(E)

    theta_index  = 0
    theta2_index = 0


    for theta in incident_angles:
        S_theta_  = np.exp(array_alignment*1j*2*np.pi*np.cos(np.radians(theta))) # Scanning vector
        theta2_index=0
        for theta2 in incident_angles[0:theta_index]:
            S_theta_2_ = np.exp(array_alignment*1j*2*np.pi*np.cos(np.radians(theta2))) # Scanning vector
            a = np.matrix(S_theta_+S_theta_2_).getT() # Spatial signiture vector
            ADORT[theta_index,theta2_index]=  np.real(1/np.abs(a.getH()*(E*E.getH())*a))
            theta2_index += 1
        theta_index += 1

    return ADORT, incident_angles



def DOA_IAA_APES(X, scanning_vectors, iterations, R_N, confidence=1):
    """
                IAA-APES - Iterative Adaptive Approach to Amplitude and Phase Estimation



         Description:
         ------------
           The function implements a modified IAA-APES method for direction estimation.
           For more information consult
           T. Yardibi, J. Li, P. Stoica, M. Xue and A. B. Baggeroer, "Source Localization and
            Sensing: A Nonparametric Iterative Adaptive Approach Based on Weighted Least Squares,"


           Calculation method (For iteration "n") :

                                         H       -1
                                     a(k)  * R(n)   * y(t)
                        s(k,n,t) = ---------------------------
                                         H       -1
                                     a(k)  * R(n)   * a(k)

                                 |                    2                             -1        c |
                        p(k,n) = | time_mean( s(k,n,t)  ) * [p(k,n-1) * [a(k) * R(n)  * a(k)]]  |


                        P(n) = diag(p(n))

                                           H
                        R(n) = A * P(n) * A


                        k  - Index associated with look direction (omitted when all angles are refered to)
                        n  - Iteration number
                        c  - Confidence hyperparameter
                        t  - Sample number in time

                        a  - Steering vector in direction theta
                        y  - Vector of signals associated with each array
                        p  - Vector of powers associated with index k

                        A  - Scanning vector matrix
                        P  - Source correlation matrix
                        R  - Array correlation matrix

        Implementation notes:
        ---------------------

            This function will work for both coherent  and uncoherent sources, and the main limitation
            on the number of coherent sources is the number of elements. Iterative refinement of
            source correlation matrices is expensive! Thus, expect this algorithm to take a little more
            time. The SAMV is typically somewhat faster than the IAA-APES. However, IAA-APES shows a
            somewhat higher resolution in moderate noise.

            The hyper-parameter denoted "confidence" allows for higher resolution, and a lower "floor",
            but at a high value is somewhat more likely to not detect lower power sources and mistake
            sidelobes for sources. A good starting value is 1. Values much higher than
            1 show less improvement compared to the SAMV approach, when in very low noise. The results
            for SAMV and IAA-APES are nearly identical for a shared confidence hyperparameter value of 0.

            The confidence hyperparameter serves as an extension the similar concepts in the SAMV
            algorithm to IAA-APES. For the "normal" IAA-APES described in the above paper, let
            confidence = 0. Higher values are almost always superior.

            The algorithm uses an initial "guess" of the pseudospectrum to begin. The
            method selected in this code is the Bartlett.

        Parameters:
        -----------

            :param X : Received multichannel signal matrix from the antenna array.
            :param scanning_vectors: Steering vector matrix, equal to A
            :param iterations: Number of iterations, N
            :param R_N : Covariance matrix used for the initial guess, and only the initial guess
            :param confidence: Hyperparameter used to tune data/existing result dependence

            :type X : 2D numpy array of size MxT, where T is the number of samples
            :param scanning_vectors: 2D numpy array of size MxK, where K is the number of search angles
            :param iterations: int
            :param R_N: 2D numpy array of size MxM
            :param confidence: float

       Return values:
       --------------

            :return ADSINR:  Angular dependenet signal to noise ratio
	        :rtype ADSINR: numpy array

            :return -1, -1: Input spatial correlation matrix is not quadratic
            :return -2, -2: dimension of R not equal with dimension of the antenna array
            :return -3, -3: Spatial correlation matrix is singular

    """
    #Give the scanning vector matrix a more concise and conventional name
    A = scanning_vectors

    n_angles = np.shape(A)[1]
    n_elem = np.shape(X)[0]
    n_samples = np.shape(X)[1]

    if (confidence < 0 or confidence > 2):
        print("Warning - Unusual confidence hyperparameter in IAA-APES")

    if np.size(R_N, 0) != np.size(R_N, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1

    if np.size(R_N, 0) != np.size(scanning_vectors, 0):
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return -2, -2


    s = np.zeros(n_samples, dtype=np.complex128)
    P = np.zeros((n_angles, n_angles), dtype=np.complex128)
    R_inv = np.identity(n_elem, dtype=np.complex128)


    # Initial estimate of the pseudospectrum via the Bartlett method
    a = np.zeros((n_elem, 1), dtype=np.complex128)
    for angle in range(n_angles):
        a[:,0] = A[:,angle]
        P[angle,angle] = np.matmul(np.matmul(a.T.conj(), R_N), a)[0,0]

    for iteration in range(iterations):
        R = np.matmul(np.matmul(A, P) ,A.T.conj())
        R_inv = lin.inv(R)

        for angle in range(n_angles):
            # Decorrelate the existing signals using the Capon method
            # I use an internal implementation, because I want to avoid
            #   uneccessary computions, and don't want coupling with beamforming.py
            a_H = A[:,angle].T.conj()
            a_H_R_inv = np.matmul(a_H, R_inv)
            capon_inv = np.matmul(a_H_R_inv, A[:,angle])
            coef = a_H_R_inv / np.matmul(a_H_R_inv, A[:,angle])

            # Beamforming at the angle using the updated power spectrum
            s = np.matmul(X.T, coef)

            # Update the estimates in the source correlation matrix using
            #   the derived pseudospectrum, as well as previous data
            modifier = capon_inv * P[angle,angle]
            P[angle,angle] = np.abs(np.mean(s**2) * modifier**confidence)

    #Finished: fetch the diagonal of the source correlation matrix and return it
    ADSINR = np.diag(P)
    return ADSINR


def DOA_SAMV(X, scanning_vectors, iterations, R_N, confidence=1):
    """
                SAMV - Iterative Sparse Asymptotic Minimum Variance Approach



         Description:
         ------------
           The function implements a modified Iterative SAMV method for direction estimation.
           For more information consult --
           H. Abeida, Q. Zhang, J. Li and N. Merabtine,
           "Iterative Sparse Asymptotic Minimum Variance Based Approaches for Array Processing"


           Calculation method (For iteration "n") :

                                         H       -1               -1
                                     a(k)  * R(n)  * R_N(n) * R(n)  * y(t)            c
                        p(k,n) = -----------------------------------------  * p(k,n-1)
                                         H       -1       H   2-c
                                    [ a(k)  * R(n)   * a(k)  ]

                                    -2
                                Tr(R  * R_N)
                        s(n) =  --------------------
                                    -2
                                Tr(R  )

                        P(n) = diag(p(n))

                                           H
                        R(n) = A * P(n) * A  +  s(n) * I



                        k  - Index associated with a look direction
                        n  - Iteration number
                        c  - Confidence hyperparameter
                        t  - Sample number in time
                        s  - Estimate of noise power at each iteration (usually a sigma)

                        a  - Steering vector in direction theta
                        y  - Vector of signals associated with each array
                        p  - Vector of powers associated with index k

                        A    - Scanning vector matrix
                        P    - Source correlation matrix
                        R    - Array correlation matrix formed via iterative refinement
                        R_N  - Array correlation matrix formed via direct averaging

        Implementation notes:
        ---------------------

            This function will work for both coherent  and uncoherent sources, and the main limitation
            on the number of coherent sources is the number of elements. Iterative refinement of
            source correlation matrices is expensive! Thus, expect this algorithm to take a little more
            time.

            The hyper-parameter denoted "confidence" allows for higher resolution, and a lower "floor",
            but at a high value is somewhat more likely to not detect lower power sources and mistake
            sidelobes for sources. A good starting value is 1. Confidence values near
            2 have a tendency remove all but the strongest source; setting all the other values in
            the pseudo-spectrum to zero. The results for SAMV and IAA-APES are nearly identical for a
            shared confidence hyperparameter value of 0, but SAMV shows more improvement for a higher
            confidence hyperparameter value.

            This confidence hyperparameter serves as a continuous generalisation of the the different
            variants given in the above paper. However, there are notational differences.

            SAMV-0      corresponds to      confidence = 2
            SAMV-1      corresponds to      confidence = 0
            SAMV-2      corresponds to      confidence = 1

            The algorithm uses an initial "guess" of the pseudospectrum to begin. The
            method selected in this code is the Bartlett.

        Parameters:
        -----------

            :param X : Received multichannel signal matrix from the antenna array.
            :param scanning_vectors: Steering vector matrix, equal to A
            :param iterations: Number of iterations, N
            :param R_N : Data covariance matrix
            :param confidence: Hyperparameter used to tune data/existing result dependence

            :type X : 2D numpy array of size MxT, where T is the number of samples
            :param scanning_vectors: 2D numpy array of size MxK, where K is the number of search angles
            :param iterations: int
            :param R_N: 2D numpy array of size MxM
            :param confidence: float

       Return values:
       --------------

            :return ADSINR:  Angular dependenet signal to noise ratio
	        :rtype ADSINR: numpy array

            :return -1, -1: Input spatial correlation matrix is not quadratic
            :return -2, -2: dimension of R not equal with dimension of the antenna array
            :return -3, -3: Spatial correlation matrix is singular

    """
    #Give the scanning vector matrix a more concise and conventional name
    A = scanning_vectors

    n_elem = np.shape(X)[0]
    n_samples = np.shape(X)[1]
    n_angles = np.shape(A)[1]

    if (confidence < 0 or confidence > 2):
        print("Warning - Unusual confidence hyperparameter in SAMV")

    # --> Input check
    if np.size(R_N, 0) != np.size(R_N, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1

    if np.size(R_N, 0) != np.size(scanning_vectors, 0):
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return -2, -2


    #Find P (initial guess) via the Bartlet method
    #Sigma (noise power) follows the methodology defined in the above paper
    #Let a be a placeholder for column steering vectors
    P = np.zeros((n_angles, n_angles), dtype=np.complex128)
    a = np.zeros((n_elem, 1), dtype=np.complex128)
    for angle in range(n_angles):
        a[:,0] = A[:,angle]
        P[angle,angle] = np.matmul(np.matmul(a.T.conj(), R_N), a)[0,0]
    sigma = (n_elem*n_samples)**(-1) * np.sum(lin.norm(X, axis=0))

    for iteration in range(iterations):
        #Preliminary matrix operations (so that time consumption is minimised)
        R = np.matmul(np.matmul(A, P), A.T.conj()) + sigma * np.identity(n_elem)
        R_inv = lin.inv(R)
        R_prod = np.matmul(np.matmul(R_inv, R_N), R_inv)
        R_inv2 = np.matmul(R_inv, R_inv)

        #Parameters for the next iteration!
        sigma = np.trace(np.matmul(R_inv2, R_N)) / np.trace(R_inv2)
        for angle in range(n_angles):
            a = A[:,angle]
            bottom_line = np.matmul(np.matmul(a.T.conj(), R_inv), a)
            top_line = np.matmul(np.matmul(a.T.conj(), R_prod) ,a)
            modifier = top_line / (bottom_line**(2 - confidence))
            P[angle,angle] = (P[angle,angle]**confidence) * modifier

    #Finished: fetch the diagonal of the source correlation matrix and return it
    ADSINR = np.diag(P)
    return ADSINR



#********************************************************
#*****       CORRELATION MATRIX ESTIMATIONS         *****
#********************************************************
def corr_matrix_estimate(X, imp="mem_eff"):
    """
        Estimates the spatial correlation matrix with sample averaging

    Implementation notes:
    --------------------
        Two different implementation exist for this function call. One of them use a for loop to iterate through the
        signal samples while the other use a direct matrix product from numpy. The latter consumes more memory
        (as all the received coherent multichannel samples must be available at the same time)
        but much faster for large arrays. The implementation can be selected using the "imp" function parameter.
        Set imp="mem_eff" to use the memory efficient implementation with a for loop or set to "fast" in order to use
        the faster direct matrix product implementation.


    Parameters:
    -----------
        :param X : Received multichannel signal matrix from the antenna array.
        :param imp: Selects the implementation method. Valid values are "mem_eff" and "fast". The default value is "mem_eff".
        :type X: N x M complex numpy array N is the number of samples, M is the number of antenna elements.
        :type imp: string

    Return values:
    -------------

        :return R : Estimated spatial correlation matrix
        :rtype R: M x M complex numpy array

        :return -1 : When unidentified implementation method was specified
    """
    N = np.size(X, 0)
    M = np.size(X, 1)
    R = np.zeros((M, M), dtype=complex)

    # --input check--
    if N < M:
        print("WARNING: Number of antenna elements is greather than the number of time samples")
        print("WARNING: You may flipped the input matrix")

    # --calculation--
    if imp == "mem_eff":
        for n in range(N):
            R += np.outer(X[n, :], np.conjugate(X[n, :]))
    elif imp == "fast":
            X = X.T
            R = np.dot(X, X.conj().T)
    else:
        print("ERROR: Unidentified implementation method")
        print("ERROR: No output is generated")
        return -1

    R = np.divide(R, N)
    return R

def extened_mra_corr_mtx(R):
    """

        Fill the defficient correlation matrix when the antenna array is
        placed in MRA (Minimum Redundancy Alignment). To fill the deficient
        elements the Toeplitz and Hermitian property of the correlation matrix
        is utilized.

        Currently it works only for quad element linear arrays.
        TODO: Implementation of general cases

        Implementation notes:
        ---------------------
            Correlation coeffcients corresponding to the blind antenna elements
            must be zero in the spatial correlation matrix.

            example: Quad element linear array with blind element at the third position
                                    | R11 R12 0   R14 |
                                R=  | R21 R22 0   R24 |
                                    | 0   0   0   0   |
                                    | R41 R42 0   R44 |
        Parameters:
        -----------

            :param R : Spatial correlation matrix
            :type  R : M x M complex numpy array, M is the number of antenna elements.

        Return values:
        --------------

            :return R: Extended correlation matrix
            :rtype R: 4 x 4 complex numpy array

            :return -1, -1: Input spatial correlation matrix is not quadratic

    """
    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1

    if np.size(R,0) == 3 and np.size(R,1) == 3:
        R = np.insert(R,2,0,axis=1)
        R = np.insert(R,2,0,axis=0)

    # Fill deficient correlation matrix (Toeplitz matrix)
    R[0, 2] = R[1, 3]
    R[2, 0] = np.conjugate(R[0, 2])

    R[1, 2] = (R[0, 1] + np.conjugate(R[1, 0])) / 2
    R[2, 1] = np.conjugate(R[1, 2])

    R[2, 2] = (R[0, 0] + R[1, 1] + R[3, 3]) / 3

    R[3, 2] = (R[2, 1] + R[1, 0] + np.conjugate(R[1, 2]) + np.conjugate(
        R[0, 1])) / 4
    R[2, 3] = np.conjugate(R[3, 2])
    return R

def forward_backward_avg(R):
    """
        Calculates the forward-backward averaging of the input correlation matrix

    Parameters:
    -----------
        :param R : Spatial correlation matrix
        :type  R : M x M complex numpy array, M is the number of antenna elements.

    Return values:
    -------------

        :return R_fb : Forward-backward averaged correlation matrix
        :rtype R_fb: M x M complex numpy array

        :return -1, -1: Input spatial correlation matrix is not quadratic

    """
    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1

    # --> Calculation
    M = np.size(R, 0)  # Number of antenna elements
    R = np.matrix(R)

    # Create exchange matrix
    J = np.eye(M)
    J = np.fliplr(J)
    J = np.matrix(J)

    R_fb = 0.5 * (R + J*np.conjugate(R)*J)

    return np.array(R_fb)


def spatial_smoothing(X, P, direction="forward"):
    """

        Calculates the forward and (or) backward spatially smoothed correlation matrix

    Parameters:
    -----------
        :param X : Received multichannel signal matrix from the antenna array.
        :param P : Size of the subarray
        :param direction:

        :type X: N x M complex numpy array N is the number of samples, M is the number of antenna elements.
        :type P : int
        :type direction: string

    Return values:
    -------------

        :return R_ss : Forward-backward averaged correlation matrix
        :rtype R_ss: P x P complex numpy array

        -1: direction parameter is invalid
    """
    # --input check--
    N = np.size(X, 0)  # Number of samples
    M = np.size(X, 1)  # Number of antenna elements

    if N < M:
        print("WARNING: Number of antenna elements is greather than the number of time samples")
        print("WARNING: You may flipped the input matrix")
    L = M-P+1 # Number of subarrays
    Rss = np.zeros((P,P), dtype=complex) # Spatiali smoothed correlation matrix

    if direction == "forward" or direction == "forward-backward":
        for l in range(L):
            Rxx = np.zeros((P,P), dtype=complex) # Correlation matrix allocation
            for n in np.arange(0,N,1):
                Rxx += np.outer(X[n,l:l+P],np.conj(X[n,l:l+P]))
            np.divide(Rxx,N) # normalization
            Rss+=Rxx
    if direction == "backward" or direction == "forward-backward":
        for l in range(L):
            Rxx = np.zeros((P,P), dtype=complex) # Correlation matrix allocation
            for n in np.arange(0,N,1):
                d = np.conj(X[n,M-l-P:M-l] [::-1])
                Rxx += np.outer(d,np.conj(d))
            np.divide(Rxx,N) # normalization
            Rss+=Rxx
    if not (direction == "forward" or direction == "backward" or direction == "forward-backward"):
        print("ERROR: Smoothing direction not recognized ! ")
        return -1

    # normalization
    if direction == "forward-backward":
        np.divide(Rss,2*L)
    else:
        np.divide(Rss,L)

    return Rss

def estimate_sig_dim(R):
    """
        Estimates the signal subspace dimension for the MUSIC algorithm

        Notes: Identifying the subspace dimension with K-mean clustering is not
               verified nor theoretically nor experimentally, thus using this
               function is not recommended.

     Parameters:
    -----------
        :param R : Spatial correlation matrix
        :type  R : M x M complex numpy array, M is the number of antenna elements.

    Return values:
    -------------

            :return signal_dimension : Estimated signal dimension
            :rtype signal_dimension: int

            :return -1, -1: Input spatial correlation matrix is not quadratic

    """
    from scipy.cluster import vq

    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1
    print("WARNING: This function is experimental")

    # Identify dominant eigenvalues a.k.a signal subspace dimension with K-Mean clutering
    sigmai, vi = lin.eig(R)
    eigenvalues = np.abs(sigmai)
    centroids, variance = vq.kmeans(eigenvalues,2)
    identified, distance = vq.vq(eigenvalues, centroids)

    cluster_1 = eigenvalues[identified == 0]
    cluster_2 = eigenvalues[identified == 1]
    print(cluster_1)
    print(cluster_2)
    print(centroids)

    if centroids[0] > centroids[1]:
        signal_dimension = len(eigenvalues[identified == 0])
    else:
        signal_dimension = len(eigenvalues[identified == 1])

    return signal_dimension


#********************************************************
#*****            ARRAY UTIL FUNCTIONS              *****
#********************************************************
def gen_ula_scanning_vectors(array_alignment, thetas):
    """
    Description:
    ------------
        This function prepares scanning vectorors for Linear array antenna systems

    Parameters:
    -----------

        :param array_alignment : A vector containing the distances between the antenna elements.
                                e.g.: [0, 0.5*lambda, 1*lambda, ... ]
        :param  thetas : A vector containing the incident angles e.g.: [0deg, 1deg, 2deg, ..., 180 deg]

        :type array_alignment: 1D numpy array
        :type thetas: 1D numpy array

    Return values:
    -------------

        :return scanning_vectors : Estimated signal dimension
        :rtype scanning_vectors: 2D numpy array with size: M x P, where P is the number of incident angles

    """
    M = np.size(array_alignment, 0)  # Number of antenna elements
    scanning_vectors = np.zeros((M, np.size(thetas)), dtype=complex)
    for i in range(np.size(thetas)):
        scanning_vectors[:, i] = np.exp(array_alignment*1j*2*np.pi*np.cos(np.radians(thetas[i]))) # Scanning vector

    return scanning_vectors

def gen_uca_scanning_vectors(M, r, thetas):
    """
    Description:
    ------------
        This function prepares scanning vectorors for Uniform Circular Array antenna systems

    Parameters:
    -----------

        :param M : Number of antenna elements on the circle
        :param r : radius of the antenna system
        :param thetas : A vector containing the incident angles e.g.: [0deg, 1deg, 2deg, ..., 180 deg]

        :type M: int
        :type R: float
        :type thetas: 1D numpy array

    Return values:
    -------------

        :return scanning_vectors : Estimated signal dimension
        :rtype scanning_vectors: 2D numpy array with size: M x P, where P is the number of incident angles

    """
    scanning_vectors = np.zeros((M, np.size(thetas)), dtype=complex)
    for i in range(np.size(thetas)):
        for j in np.arange(0,M,1):
            scanning_vectors[j, i] = np.exp(1j*2*np.pi*r*np.cos(np.radians(thetas[i]-j*(360)/M))) # UCA

    return scanning_vectors

def gen_scanning_vectors(M, x, y, thetas):
    """
    Description:
    ------------
        This function prepares scanning vectorors for general antenna array configurations

    Parameters:
    -----------

        :param M : Number of antenna elements on the circle
        :param x : x coordinates of the antenna elements on a plane
        :param y : y coordinates of the antenna elements on a plane
        :param thetas : A vector containing the incident angles e.g.: [0deg, 1deg, 2deg, ..., 180 deg]

        :type M: int
        :type x: 1D numpy array
        :type y: 1D numpy array
        :type R: float
        :type thetas: 1D numpy array

    Return values:
    -------------

        :return scanning_vectors : Estimated signal dimension
        :rtype scanning_vectors: 2D numpy array with size: M x P, where P is the number of incident angles

    """
    scanning_vectors = np.zeros((M, np.size(thetas)), dtype=complex)
    for i in range(np.size(thetas)):
        scanning_vectors[:,i] = np.exp(1j*2*np.pi* (x*np.cos(np.deg2rad(thetas[i])) + y*np.sin(np.deg2rad(thetas[i]))))

    return scanning_vectors
#********************************************************
#*****          ALIASING UTIL FUNCTIONS             *****
#********************************************************
def alias_border_calc(d):
    """
        Calculate the angle borders of the aliasing region for ULA antenna systems
    Parameters:
    -----------
        :param d: distance between antenna elements [lambda]
        :type d: float

    Return values:
    --------------
        :return anlge_list : Angle borders of the unambious region
        :rtype anlge_list: List with two elements
    """
    theta_alias_min = np.rad2deg(np.arccos(1/(2*d)))
    theta_alias_max = np.rad2deg(np.arccos(1/d -1))
    return (theta_alias_min,theta_alias_max)

#********************************************************
#*****                DISPLAY FUNCTIONS             *****
#********************************************************
def DOA_plot(DOA_data, incident_angles, log_scale_min=None, alias_highlight=True, d=0.5, axes=None):

    DOA_data = np.divide(np.abs(DOA_data),np.max(np.abs(DOA_data))) # normalization
    if(log_scale_min != None):
        DOA_data = 10*np.log10(DOA_data)
        theta_index = 0
        for theta in incident_angles:
            if DOA_data[theta_index] < log_scale_min:
                DOA_data[theta_index] = log_scale_min
            theta_index += 1

    if axes is None:
        fig = plt.figure()
        axes  = fig.add_subplot(111)

    #Plot DOA results
    axes.plot(incident_angles,DOA_data)
    axes.set_title('Direction of Arrival estimation ',fontsize = 16)
    axes.set_xlabel('Incident angle [deg]')
    axes.set_ylabel('Amplitude [dB]')

    # Alias highlight
    if alias_highlight:
        (theta_alias_min,theta_alias_max) = alias_border_calc(d)
        print('Minimum alias angle %2.2f '%theta_alias_min)
        print('Maximum alias angle %2.2f '%theta_alias_max)

        axes.axvspan(theta_alias_min, theta_alias_max, color='red', alpha=0.3)
        axes.axvspan(180-theta_alias_min, 180, color='red', alpha=0.3)

        axes.axvspan(180-theta_alias_min, 180-theta_alias_max, color='blue', alpha=0.3)
        axes.axvspan(0, theta_alias_min, color='blue', alpha=0.3)

    plt.grid()
    return axes
