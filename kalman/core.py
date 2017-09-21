
__all__ = ['KalmanFilter']

import numpy              as np
import theano
import theano.tensor      as tt

import pymc3
from pymc3.distributions  import Continuous

class KalmanFilter(Continuous):
    
    def __init__(self, a0, P0, Z, d, H, T, c, R, Q, *args, **kwargs):
        
        super(KalmanFilter, self).__init__(*args, **kwargs)
        
        self.a0 = tt.as_tensor_variable(a0, name='a0')
        self.P0 = tt.as_tensor_variable(P0, name='P0')
        
        self.Z = tt.as_tensor_variable(Z, name='Z')
        self.d = tt.as_tensor_variable(d, name='d')
        self.H = tt.as_tensor_variable(H, name='H')
        
        self.T = tt.as_tensor_variable(T, name='T')
        self.c = tt.as_tensor_variable(c, name='c')
        self.R = tt.as_tensor_variable(R, name='R')
        self.Q = tt.as_tensor_variable(Q, name='Q')
        
        self.mean = tt.as_tensor_variable(0.)
    
    @staticmethod
    def det_and_inv(A):
        """Get matrix det and inv, with special cases for small matrices"""

        if tt.eq(A.shape, (1,1)):
            det = A[0,0]
            inv = 1/A
        else:
            det = tt.nlinalg.det(A)
            inv = tt.nlinalg.matrix_inverse(A)

        return det, inv
    
    @staticmethod
    def oneStepPredictionState(ai, Pi, T, c, R, Q):
        
        afpred = T.dot(ai) + c
        Pfpred = T.dot(Pi).dot(T.T) + R.dot(Q).dot(R.T)

        return afpred, Pfpred
    
    @staticmethod
    def oneStepPredictionObs(afpred, Pfpred, Z, d, H):

        bfpred = Z.dot(afpred) + d
        Ffpred = Z.dot(Pfpred).dot(Z.T) + H

        return bfpred, Ffpred

    @staticmethod
    def oneStepPredictionLlik(y, bfpred, Ffpreddet, Ffpredinv):

        N = y.shape[0]
        v = (y - bfpred)

        return -0.5*(N*np.log(2 * np.pi) + tt.log(Ffpreddet) + v.dot(Ffpredinv.dot(v)))

    @staticmethod
    def oneStepUpdateState(y, afpred, Pfpred, bfpred, Ffpredinv, Z):

        af = afpred + Pfpred.dot(Z.T).dot(Ffpredinv).dot(y - bfpred)
        Pf = Pfpred - Pfpred.dot(Z.T).dot(Ffpredinv).dot(Z).dot(Pfpred)

        return af, Pf

    @staticmethod
    def oneStep(y, ai, Pi, Z, d, H, T, c, R, Q):
        
        # Prediction of the state vector distribution
        afpred, Pfpred = KalmanFilter.oneStepPredictionState(ai, Pi, T, c, R, Q)

        # Prediction of the observation distribution
        bfpred, Ffpred = KalmanFilter.oneStepPredictionObs(afpred, Pfpred, Z, d, H)

        # Log-likelihood of the prediction
        Ffpreddet, Ffpredinv = KalmanFilter.det_and_inv(Ffpred)
        llik = KalmanFilter.oneStepPredictionLlik(y, bfpred, Ffpreddet, Ffpredinv)

        # Update of the state vector distribution based on the new data
        af, Pf = KalmanFilter.oneStepUpdateState(y, afpred, Pfpred, bfpred, Ffpredinv, Z)

        return af, Pf, llik
    
    @staticmethod
    def filter(Y, a0, P0, Z, d, H, T, c, R, Q, **kwargs):
        
        (at, Pt, lliks), updates = theano.scan(
            fn            = KalmanFilter.oneStep,
            sequences     = [tt.as_tensor_variable(Y, name='Y'),],
            outputs_info  = [dict(initial=a0),
                             dict(initial=P0),
                             None],
            non_sequences = [Z, d, H, T, c, R, Q],
            strict        = True,
            **kwargs)
        
        return (at, Pt, lliks), updates
    
    def logp(self, Y):
        
        (_, _, lliks), _ = KalmanFilter.filter(Y, self.a0, self.P0,
                                               self.Z, self.d, self.H,
                                               self.T, self.c, self.R, self.Q)
        
        return lliks[1:].sum()

