#!/usr/bin/env python
# coding: utf-8

# In[339]:


class CPTAgent:
    """Generate an agent that expresses preferences between two prospects 
    (ie., gambles) according to principles of Cumulative Prospect Theory (1992). 
    
    Generates preferences between prospects by implementing value and 
    probability weighting functions following Tversky & Kahneman, 
    Advances in Prospect Theory: Cumulative Representation of Uncertainty (1992).
    
    Parameters
    ------------
    alpha: float {a | 0 <= a <= 1}
        - parameter controlling the rate of change in value over gains
    beta (optional): float {b | 0 <= b <= 1}, default=None
        - parameter controlling the rate of change in value over losses
    lambda_: float {l| l > 1}
        - parameter controlling the steepness of the value function over losses relative to gains 
    delta: float {d| 0 <= d <= 1}
        - parameter controlling the shape of the transformation of probability weight
        
    Attributes
    ------------
    parameters: dictionary
        - contains parameter names as keys and parameter inputs as values
    value_fun: function
        - input floating point representing utility
        - output calculation of value function based on input, alpha, beta (optional), and lambda_ parameters
    weight_fun: function
        - input floating point representing probability
        - output calculation of value function based on probability and delta parameter
    """
    
    def __init__(self, alpha:float=1, beta:float=None, gamma:float=1, delta:float=None, lambda_:float=1):
        
        beta = beta or alpha
        delta = delta or gamma
        
        # range checks
        assert self._zero_one_bound(alpha, beta, gamma, delta),        'alpha, beta, and delta parameters must lie within 0 and 1 inclusive'
        assert lambda_ >=1,        'lambda_ parameter must be be greater than or equal to 1'
        
        self.parameters = {'alpha':alpha, 
                           'beta':beta, 
                           'gamma':gamma,
                           'delta':delta,
                           'lambda':lambda_
                          }
        
        self._value_fun = lambda x: x**alpha if x >= 0 else -lambda_*((-x)**beta)
        self._weight_fun = lambda p: p**gamma/(p**gamma + (1-p)**gamma)**gamma
        
    def __repr__(self):
        return 'CPTAgent(alpha=%r, beta=%r, gamma=%r, delta=%r, lambda_=%r)'     % tuple(self.parameters.values())

    def _zero_one_bound(self, *args)->bool:
        """Range check on instance parameters.
        Tests whether alpha, delta, and (optional) beta fall within 0 and 1 inclusive"""
        return all([n <= 1 and n >=0 for n in args])
    
    def evaluate(self, prospect:dict, verbose:bool=False)->float:
        """Prospect valuation
        
        Parameters
        ----------
        prospect: dictionary containing two items
            - key 'outcome': list of real numbers corresponding to outcomes 
            - key 'probability': list of floats between 0 and 1
        verbose: bool 
            - True: output list of outcome, probability tuples sorted by outcome; capacities; valuation
            - False: output valuation
        
        returns
        ----------
        valuation (if verbose=False) : float representing overall value to decision maker
        dict (if verbose=True):
            - valuation : float representing overall value to decision maker
            - prospect : list of tuples containing outcome, probability pairs sorted by outcome
            - transform_value : list of outcomes after value transformation function 
            - capacity : list of capacities sorted by outcome
        """
        # consistency checks
        assert len(prospect['outcome'])==len(prospect['probability']),        'number of probabilities must match number of outcomes'
        assert all(p >=0 for p in prospect['probability']),        'probabilities must be greater than or equal to 0'       
        assert sum(prospect['probability']) <= 1, 'sum of probabilities must not exceed one'
        
        prosp_ls = [(o, p) for o, p in zip(prospect['outcome'], prospect['probability'])]
        srt_prosp_ls = sorted(prosp_ls, key=lambda x: x[0]) # prospects sorted by outcome
        gain_ls = [n for n in srt_prosp_ls if n[0]>=0] 
        loss_ls = [n for n in srt_prosp_ls if n[0]<0]
        rvrs_loss_ls = list(reversed(loss_ls))
        
        # compute capacities
        capacity_ge0 = self._cap_fn(gain_ls)
        capacity_lt0 = list(reversed(self._cap_fn(rvrs_loss_ls)))
        capacity = capacity_lt0 + capacity_ge0
        # compute utilities
        outcome = [o[0] for o in srt_prosp_ls]
        trans_val = [self._value_fun(o) for o in outcome]
        valuation = sum(o*c for o,c in zip(trans_val, capacity))
        
        if verbose:
            return dict(valuation=valuation, prospect=srt_prosp_ls, transform_value=trans_val, capacity=capacity)
        else: 
            return valuation
    
    def _cap_fn(self, ls:list)->list:
        """Compute capacities for each option i
        Parameters
        ----------
        ls: list containing tuples of outcomes and probabilities: (outcome, probability)
            - lists should contain either all negative outcomes, or all non-negative outcomes
            - lists should be sorted in ascending order of increasing absolute magnitude of outcomes:
                - positive outcomes: eg., [(0, 0.2), (3.5, 0.1), (4, 0.3)]
                - negative outcomes: eg., [(-1, 0.3), (-5, 0.1)]

        returns: list of capacities for each outcome i
        """
        ls_len = len(ls)
        csum_p_ls = [sum(x[1] for x in ls[opt:]) for opt in range(0, ls_len)]
        weight_i_ls = [self._weight_fun(x) for x in csum_p_ls]
        weight_j_ls = [self._weight_fun(x) for x in csum_p_ls[1:]]+[0]
        return [i-j for i, j in zip(weight_i_ls, weight_j_ls)]
        
    def choose(self, prospect1:dict=None, prospect2:dict=None):
        """Choice between two prospects"""
        
        if self.evaluate(prospect1) > self.evaluate(prospect2):
            return {'prospect1':prospect1} 
        else: return {'prospect2':prospect2}

