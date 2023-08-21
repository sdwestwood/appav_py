"""
Miscellanious functions for running single_trial_analysis
"""
import numpy as np
import math 


def getX(eegdata, cond_id, cond_discr, offset, dur):
    
    """
    INPUT PARAMETERS
    
    eegdata:    epoched data in 3D format (CHANNELS x TIME x TRIALS)
    cond_id:    vector of condition IDs for each trial (1 x TRIALS)
    cond_discr: two element vector with condition IDs for discrimination (e.g. [1, 4])
    offset:     onset time of analysis window in samples (relative to trial onset)
    duration:   size of analysis window in samples 
    """
    
    # get trial indices for condition A
    idx_A = np.where(cond_id == cond_discr[0])[0]
    # get trial indices for condition A
    idx_B = np.where(cond_id == cond_discr[1])[0]
    
    # build a logical (binary) truth labels vector (0: condition A, 1: condition B)
    truth = np.array([0]*len(idx_A) + [1]*len(idx_B))
    
    X1 = eegdata[:, offset:(offset+dur), idx_A].reshape(eegdata.shape[0], dur*len(idx_A),order='F')      
    X2 = eegdata[:, offset:(offset+dur), idx_B].reshape(eegdata.shape[0], dur*len(idx_B),order='F')  
    X = np.concatenate((X1,X2),1)
    
    return {'X': X, 'truth': truth}


#############################################################################################

def geninv(G):
    """
    Returns the Moore-Penrose inverse of the argument
    Transpose if m < n
    """

    pass
    # m,n = size(G)
    # transpose = false
    #
    # if m < n:
    #     transpose = true
    #     A = G*np.transpose(G)
    #     n = m
    # else:
    #     A = np.transpose(G)*G
    #
    # # Full rank Cholesky factorization of A
    # dA = diag(A)
    # tol = min(dA[dA>0])*1e-9
    # L = [0]*size(A)
    # r = 0
    #
    # for k in range(n):
    #     r = r+1
    #     L[k:n,r] = A[k:n, k] - np.transpose(L[k:n, 1:(r-1)]*L[k, 1:(r-1)])
    #     # Note: for r=1, the substracted vector is zero
    #     if L[k,r] > tol:
    #         L[k,r] = sqrt(L(k,r))
    #         if k < n:
    #             L[(k+1):n, r] = L[(k+1):n, r] / L[k, r]
    #     else:
    #         r = r-1
    #
    # L = L[:,1:r]
    #
    # # Computation of the generalized inverse of G
    # M = inv(np.transpose(L)*L)
    #
    # if transpose:
    #     Y = np.transpose(G)*L*M*M*np.transpose(L)
    # else:
    #     Y = L*M*M*np.transpose(L)*np.transpose(G)
    #
    #     return Y


#############################################################################################


def bernoull(x, eta):
    
    """
    Computes Bernoulli distribution of x for "natural parameter" eta.
    The mean m of a Bernoulli distributions relates to eta as:
    m = exp(eta)/(1+exp(eta));
    """
    
    e = math.exp(eta);

    p = np.ones(e.shape);  

    idx = np.where(np.isfinite(e))[0];

    p[idx] = math.exp(np.multiply(eta[idx],x) - math.log(1+e[idx]));

    return p

#############################################################################################



def rocarea(p, label):

    """
    [Az,tp,fp] = rocarea(p,label)
    Computes area under ROC curve corresponding to classification output p.
    Labels contain truth data {0,1}.
    tp and lp are true and false positive rate.
    If no output arguments specified, display an ROC curve with Az and
    approximate fraction correct.
    """

    [tmp,indx] = sort(-p)

    label = label>0

    Np=sum(label==1)
    Nn=sum(label==0)

    tp, fp, Az = 0
    pinc = 1/Np
    finc = 1/Nn

    N = Np+Nn

    tp,fp = [0]*(N+1)

    # for i in range(N):
    #
    #   tp[i+1] = tp[i]+label(indx[i])/Np
    #   fp[i+1] = fp[i]+(~label(indx[i])/Nn
    #   Az = Az + (~label(indx[i]))*tp[i+1]/Nn
    #
    # m,i = min(fp-tp)
    # fc = 1-mean([fp[i]], 1-tp[i]])
    #
    # if nargout==0:
    #     plot a thing
      # plot(fp,tp); axis([0 1 0 1]); hold on
      # plot([0 1],[1 0],':'); hold off
      # xlabel('false positive rate')
      # ylabel('true positive rate')
      # title('ROC Curve'); axis([0 1 0 1]);
      # text(0.4,0.2,sprintf('Az = %.2f',Az))
      # text(0.4,0.1,sprintf('fc = %.2f',fc))
      # axis square

    return Az, tp, fp



