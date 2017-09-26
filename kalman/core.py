
__all__ = ['KalmanTheano', 'KalmanFilter']

import numpy              as np
import theano
import theano.tensor      as tt

import pymc3
from pymc3.distributions  import Continuous

def _det_and_inv(A):
    """Get matrix det and inv, with special cases for small matrices"""

    if tt.eq(A.ndim, 0):
        det = A
        inv = 1/A
    elif tt.eq(A.shape, (1,1)):
        det = A[0,0]
        inv = 1/A
    else:
        det = tt.nlinalg.det(A)
        inv = tt.nlinalg.matrix_inverse(A)

    return det, inv

def _oneStepPredictionState(ai, Pi, T, c, R, Q):

    afpred = T.dot(ai) + c
    Pfpred = T.dot(Pi).dot(T.T) + R.dot(Q).dot(R.T)

    return afpred, Pfpred

def _oneStepPredictionObs(afpred, Pfpred, Z, d, H):

    bfpred = Z.dot(afpred) + d
    Ffpred = Z.dot(Pfpred).dot(Z.T) + H

    return bfpred, Ffpred

def _oneStepPredictionLlik(y, bfpred, Ffpreddet, Ffpredinv):

    if tt.eq(y.ndim, 0):
        N = 1
    else:
        N = y.shape[0]
    v = (y - bfpred)

    return -0.5*(N*np.log(2 * np.pi) + tt.log(Ffpreddet) + v.dot(Ffpredinv.dot(v)))

def _oneStepUpdateState(y, afpred, Pfpred, bfpred, Ffpredinv, Z):

    af = afpred + Pfpred.dot(Z.T).dot(Ffpredinv).dot(y - bfpred)
    Pf = Pfpred - Pfpred.dot(Z.T).dot(Ffpredinv).dot(Z).dot(Pfpred)

    return af, Pf

def _oneStep(y, Z, d, H, T, c, R, Q, ai, Pi):

    # Prediction of the state vector distribution
    afpred, Pfpred = _oneStepPredictionState(ai, Pi, T, c, R, Q)

    # Prediction of the observation distribution
    bfpred, Ffpred = _oneStepPredictionObs(afpred, Pfpred, Z, d, H)

    # Log-likelihood of the prediction
    Ffpreddet, Ffpredinv = _det_and_inv(Ffpred)
    llik = _oneStepPredictionLlik(y, bfpred, Ffpreddet, Ffpredinv)

    # Update of the state vector distribution based on the new data
    af, Pf = _oneStepUpdateState(y, afpred, Pfpred, bfpred, Ffpredinv, Z)

    return af, Pf, llik

class KalmanTheano(object):

    def __init__(self, Z, d, H, T, c, R, Q, a0, P0):
        
        self.Z = tt.as_tensor_variable(Z, name='Z')
        self.d = tt.as_tensor_variable(d, name='d')
        self.H = tt.as_tensor_variable(H, name='H')
        
        self.T = tt.as_tensor_variable(T, name='T')
        self.c = tt.as_tensor_variable(c, name='c')
        self.R = tt.as_tensor_variable(R, name='R')
        self.Q = tt.as_tensor_variable(Q, name='Q')
        
        self.a0 = tt.as_tensor_variable(a0, name='a0')
        self.P0 = tt.as_tensor_variable(P0, name='P0')
    
    def filter(self, Y, **kwargs):
        
        Y = tt.as_tensor_variable(Y, name='Y')
        
        # Check which arguments are time-dependent sequences
        dy = Y.ndim - 1         # Observations: scalar (0) or vector (1)
        da = self.a0.ndim       # State: scalar (0) or vector (1)
        dims_non_sequences = {'Z': dy+da, 'd': dy, 'H': 2*dy,
                              'T': 2*da , 'c': da, 'R': 2*da, 'Q': 2*da}
        sequences, non_sequences = [], []
        for c, dim_non_seq in dims_non_sequences.items():
            dim_real = getattr(self, c).ndim
            if dim_real - dim_non_seq == 1:
                sequences.append(c)
            elif dim_real == dim_non_seq:
                non_sequences.append(c)
            else:
                raise ValueError('"%s" should have depth %d or %d; got %d'
                                 % (c, dim_non_seq, dim_non_seq+1, dim_real))
        
        # Create function with correct ordering
        fn = eval('lambda %s: _oneStep(y,Z,d,H,T,c,R,Q,ai,Pi)'
                  % ','.join(['y'] + sequences + ['ai', 'Pi'] + non_sequences))
        
        (at, Pt, lliks), updates = theano.scan(
            fn            = fn,
            sequences     = [Y] + [getattr(self, v) for v in sequences],
            outputs_info  = [dict(initial=self.a0),
                             dict(initial=self.P0),
                             None],
            non_sequences = [getattr(self, v) for v in non_sequences],
            strict        = True,
            **kwargs)
        
        return (at, Pt, lliks), updates

class KalmanFilter(Continuous):
    """
    Implements a generic Kalman filter in general state space form
    
    Shape of the input tensors is given as a function of:
    
    * N: number of time steps,
    * n: size of the observation vector
    * m: size of the state vector
    * g: size of the disturbance vector in the transition equation
    
    The following rules define tensor dimension reductions allowed:
    
    * If a tensor is time-invariant, the time dimension N can be omitted
    * If n=1, all dimensions of size n can be omitted
    * If m=1 and g=1, all dimensions of size m and g can be omitted
    
    Parameters
    ----------
    Z : tensor or numpy array, dimensions N x n x m
        Tensor relating observation and state vectors
    d : tensor or numpy array, dimensions N x n
        Shift in the measurement equation
    H : tensor or numpy array, dimensions N x n x n
        Covariance matrix of the disturbances in the measurement equation
    T : tensor or numpy array, dimensions N x m x m
        Tensor relating the state vectors at times t-1, t
    c : tensor or numpy array, dimensions N x m
        Shift in the transition equation
    R : tensor or numpy array, dimensions N x m x g
        Tensor applying transition equation disturbances to state space
    Q : tensor or numpy array, dimensions N x g x g
        Covariance matrix of the disturbances in the transition equation
    a0 : tensor or numpy array, dimensions n
        Mean of the initial state vector
    P0 : tensor or numpy array, dimensions n x n
        Covariance of the initial state vector
    *args, **kwargs
        Extra arguments passed to :class:`Continuous` initialization
    
    Notes
    -----
    
    The general state space form (SSF) applies to a multivariate time series,
    y(t), containing n elements. These observable variables are related to a
    state vector a(t), containing m elements, via a measurement equation:
    
    .. math :
        
        y(t) = Z(t) a(t) + d(t) + \\varepsilon(t)\\,\\qquad
             \\varepsilon(t) \\sim \\mathcal{N}_n(0, H(t))\\ ,
    
    Although a(t) is typically not observable, its dynamics is governed by a
    first-order Markov process,  given by the transition equation:
    
    .. math :
    
        a(t) = T(t) a(t-1) + c(t) + R(t) \\eta(t)\\,\\qquad
             \\eta(t) \\sim \\mathcal{N}_g(0, Q(t))\\ .
    
    Nomenclature taken from:
    
    Forecasting, structural time series models and the Kalman filter  
    Andrew C. Harvey (1989)
    """
    
    def __init__(self, Z, d, H, T, c, R, Q, a0, P0, *args, **kwargs):
        
        super(KalmanFilter, self).__init__(*args, **kwargs)
        
        self._op  = KalmanTheano(Z, d, H, T, c, R, Q, a0, P0)
        self.mean = tt.as_tensor_variable(0.)
    
    def logp(self, Y):
        
        (_, _, lliks), _ = self._op.filter(Y)
        
        return lliks[1:].sum()

