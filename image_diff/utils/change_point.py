import numpy as np
from other import Sigmaclip


def find_changepoints(y, zmax=20):
    '''Finds changepoints in data with hmm and viterbi algo.
    Estimates the transmission and emission probs from data'''
    y = np.asarray(y)
    # estimate transmission pdf as exp with boot straping?
    randomsample = np.random.choice(y, size=len(y)*20)
    pdf_y,pdf_x = np.histogram(np.abs(np.diff(randomsample)), 400, normed=True)
    pdf_y[pdf_y == 0] = 1e-99
    pdf_x = pdf_x[:-1]
    trans_p = General_PDF(pdf_x, pdf_y)
    # Get emmission probs by sigma clipping data
    emiss_mean, emiss_std = Sigmaclip(y)
    # iteratively change upper state till max prob
    best_prob, best_state,best_changepoints  = -np.inf, 0., None
    for upper in np.unique(y)[::-1]:
        if upper <= emiss_mean:
            break
        states = np.array([emiss_mean, upper])
        prob, changepoints = viterbi(y, states, norm(loc=emiss_mean,
                                                 scale=emiss_std),
                                                 trans_p, norm)
        if prob > best_prob:
             best_prob = prob + 0
             best_state = upper + 0
             best_changepoints = np.asarray(changepoints) + 0
    return best_prob, best_changepoints
    
class General_PDF(object):
    def __init__(self, x, y):
        self.x = x
        # renormalize
        self.y = y/y.sum()
    def __call__(self, state1, state2):
        '''Calulates the prob of abs(state1 - state2) from input data'''
        return np.interp(np.abs(state1 - state2), self.x, self.y)
    def pdf(self, state1, state2):
        return self.__call__(state1, state2)
    def logpdf(self, state1, state2):
        return np.log(self.__call__(state1, state2))

def viterbi(obs, states, start_p, trans_p, emit_p):
    '''Viterbi algo from http://en.wikipedia.org/wiki/Viterbi_algorithm'''
    V = [{}]
    path = {}
    states = np.sort(states)
    # Initialize base cases (t == 0)   
    for y in states:
        V[0][y] = start_p.logpdf(y) + emit_p.logpdf(obs[0], y, start_p.std()) 
        path[y] = [y]
 
    # Run Viterbi for t > 0
    for t in xrange(1, len(obs)):
        #print t
        V.append({})
        newpath = {}
        # stuff for loop
        argsort = np.argsort(V[t-1].keys())
        oldV = np.asarray(V[t-1].values())[argsort]
        Emit = emit_p.logpdf(obs[t], states, start_p.std())
        for y in states:
            probs  = (oldV + trans_p.logpdf(states, y) + Emit)
            V[t][y] = probs.max()
            newpath[y] = path[states[probs.argmax()]] + [y]
 
        # Don't need to remember the old paths
        path = newpath
    n = 0           # if only one element is observed max is sought in the initialization values
    if len(obs)!=1:
        n = t
    (prob, state) = max((V[n][y], y) for y in states)
    return (prob, path[state])
