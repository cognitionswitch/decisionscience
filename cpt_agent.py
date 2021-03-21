#!/usr/bin/env python
# coding: utf-8

import numpy as np

class PTAgent:
    """Generate an Prospect Theory agent that expresses preferences between two lotteries 
    (ie., gambles) according to principles of Prospect Theory (1979). 
    
    PTAgent expresses preference between lotteries by implementing value and 
    probability weighting functions following Tversky & Kahneman, 
    Prospect Theory: An Analysis of Decision Under Risk (1972). The functional form
    for the value function is taken from Kahneman & Tversky's 
    Advances in Prospect Theory: Cumulative Representation of Uncertainty (1992). 
    The functional form of the weighting function is taken from this website:
    https://sites.duke.edu/econ206_01_s2011/files/2011/04/39b-Prospect-Theory-Kahnemann-Tversky_final2-1.pdf
    
    Parameters
    ------------
    alpha: float {a | 0 <= a <= 1}
        - parameter controlling the rate of change in value over gains
    beta (optional): float {b | 0 <= b <= 1}, default=None
        - parameter controlling the rate of change in value over losses
    lambda_: float {l| l > 1}
        - parameter controlling the steepness of the value function over losses relative to gains 
    delta: float {d| 0 <= d <= 1}
        - weighting function parameter controlling the shape of the transformation of probability
        
    Attributes
    ------------
    parameters: dictionary
        - a mapping of value function and weighting function parameter names (keys) and parameter inputs (values) 
    _value_fn: function
        - input floating point representing outcome value
        - output calculation of value function based on input, alpha, beta (optional), and lambda_ parameters
    _gain_wt_fn: function
        - input floating point representing probability
        - output calculation of weighting function for probabilities associated with non-negative outcomes 
        based on probability and delta parameter
    """
    
    def __init__(self, alpha:float=1, beta:float=None, gamma:float=1, delta:float=None, lambda_:float=1):
        
        beta = beta or alpha
        delta = delta or gamma
        
        # range checks
        assert self._zero_one_bound(alpha, beta, gamma, delta),\
        'alpha, gamma (and optional beta, delta) parameters must lie within 0 and 1 inclusive'
        assert lambda_ >= 1,\
        'lambda_ parameter must be be greater than or equal to 1'
        
        self.parameters = {'alpha':alpha, 
                           'beta':beta, 
                           'gamma':gamma,
                           'delta':delta,
                           'lambda':lambda_
                          }
        
        self._value_fn = lambda x: x**alpha if x >= 0 else -lambda_*((-x)**beta)
        self._gain_wt_fn = lambda p: np.exp(-(-np.log(p))**gamma)
        self._loss_wt_fn = lambda p: np.exp(-(-np.log(p))**delta)
        
    def __repr__(self):
        return 'PTAgent(alpha=%r, beta=%r, gamma=%r, delta=%r, lambda_=%r)' \
    % tuple(self.parameters.values())

    def _zero_one_bound(self, *args)->bool:
        """Range check on instance parameters.
        Tests whether all positional inputs fall within 0 and 1 inclusive"""
        return all([n <= 1 and n >=0 for n in args])
    
    def evaluate(self, lottery:dict, verbose:bool=False)->float:
        """Lottery valuation
        
        Parameters
        ----------
        lottery: dictionary containing two items
            - key 'outcome': list of real numbers corresponding to outcomes 
            - key 'probability': list of floats between 0 and 1 
            Length of list associated with 'probability' must match length of list associated with 'outcome' 
        verbose: bool 
            - True: output list of outcome, probability tuples sorted by outcome; capacities; valuation
            - False: output valuation
        
        returns
        ----------
        valuation (if verbose=False) : float representing overall value of lottery to PTAgent instance
        dict (if verbose=True):
            - valuation : float representing overall value of lottery to PTAgent instance
            - lottery : list of tuples containing outcome, probability pairs sorted by outcome
            - lottery_trans : list of outcomes after value transformation function 
        """
        # consistency checks
        assert len(lottery['outcome'])==len(lottery['probability']),\
        'number of probabilities must match number of outcomes'
        assert all(p >=0 for p in lottery['probability']),\
        'probabilities must be greater than or equal to 0'       
        assert sum(lottery['probability']) <= 1, 'sum of probabilities must not exceed one'
        
        lottery_ls = [(o, p) for o, p in zip(lottery['outcome'], lottery['probability'])]
        lottery_trans_ls = [(self._value_fn(o), self._wt_transform(p, gain=o>=0)) for (o, p) in lottery_ls]
        valuation = sum(v*w for (v, w) in lottery_trans_ls)
        
        if verbose:
            return dict(valuation=valuation, lottery=lottery_ls, lottery_trans=lottery_trans_ls)
        else: 
            return valuation
    
    def _wt_transform(self, p:float, gain:bool=True)->list:
        """Compute weight transformations function for probability p
        
        Parameters
        ----------
        p: float {x: 0 <= x <= 1}
            - probability of a lottery outcome
        gain: bool
            - boolean signifying whether the outcome associated with probability p is non-negative

        returns: weight transformation of probability p
        """

        if gain:
            weight = self._gain_wt_fn(p)
        else:
            weight = self._loss_wt_fn(p)
            
        return weight
    
    def choose(self, lottery1:dict=None, lottery2:dict=None):
        """Choice between two lotteries"""
        
        return {'lottery1':lottery1} if self.evaluate(lottery1) > self.evaluate(lottery2) else {'lottery2':lottery2}
    
    
    

class CPTAgent(PTAgent):
    
    def __init__(self, alpha:float=1, beta:float=None, gamma:float=1, delta:float=None, lambda_:float=1):
        
        beta = beta or alpha
        delta = delta or gamma
        
        super().__init__(alpha, beta, gamma, delta, lambda_)
        
        # override PTAgent weighting functions with cumulative prospect theory weighting functions (Kahneman & Tversky, 1990)
        self._gain_wt_fn = lambda p: p**gamma/(p**gamma + (1-p)**gamma)**gamma
        self._loss_wt_fn = lambda p: p**delta/(p**delta + (1-p)**delta)**delta
        
    def __repr__(self):
        return 'CPTAgent(alpha=%r, beta=%r, gamma=%r, delta=%r, lambda_=%r)' % tuple(self.parameters.values())
    
    def evaluate(self, lottery:dict, verbose:bool=False)->float:
        """Lottery valuation
        
        Parameters
        ----------
        lottery: dictionary containing two items
            - key 'outcome': list of real numbers corresponding to outcomes 
            - key 'probability': list of floats between 0 and 1 
            Length of list associated with 'probability' must match length of list associated with 'outcome' 
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
        assert len(lottery['outcome'])==len(lottery['probability']),\
        'number of probabilities must match number of outcomes'
        assert all(p >=0 for p in lottery['probability']),\
        'probabilities must be greater than or equal to 0'       
        assert sum(lottery['probability']) <= 1, 'sum of probabilities must not exceed one'
        
        lottery_ls = [(o, p) for o, p in zip(lottery['outcome'], lottery['probability'])]
        srt_lottery_ls = sorted(lottery_ls, key=lambda x: x[0]) # prospects sorted by outcome
        gain_ls = [n for n in srt_lottery_ls if n[0]>=0] 
        loss_ls = [n for n in srt_lottery_ls if n[0]<0]
        rvs_loss_ls = list(reversed(loss_ls))
        
        # compute capacities
        capacity_ge0 = self._cap_fn(gain_ls)
        capacity_lt0 = list(reversed(self._cap_fn(rvs_loss_ls, gain=False)))
        capacity_ls = capacity_lt0 + capacity_ge0
        
        # compute utilities and valuation
        trans_val_ls = [self._value_fn(o) for o, p in srt_lottery_ls]
        valuation = sum(o*c for o,c in zip(trans_val_ls, capacity_ls))
        
        if verbose:
            return dict(valuation=valuation, lottery=srt_lottery_ls, transform_value=trans_val_ls, capacity=capacity_ls)
        else: 
            return valuation
        
    def _cap_fn(self, sgn_lottery_ls:list, gain:bool=True)->list:
        """Compute capacities for each outcome-probability pair in list
        Parameters
        ----------
        sgn_lottery_ls: list containing tuples of outcomes and probabilities: (outcome, probability)
            - lists should contain either all negative outcomes, or all non-negative outcomes
            - lists should be sorted in ascending order of increasing absolute magnitude of outcomes:
                - positive outcomes: eg., [(0, 0.2), (3.5, 0.1), (4, 0.3)]
                - negative outcomes: eg., [(-1, 0.3), (-5, 0.1)]
        gain: bool
            - indicates whether all elements of sgn_lottery_ls are negative (False) or all non-negative (True)

        returns: list of capacities for each outcome-probability pair in sgn_lottery_ls
        """
        ls_len = len(sgn_lottery_ls)
        csum_p_ls = [sum(tup[1] for tup in sgn_lottery_ls[opt:]) for opt in range(0, ls_len)]
        weight_i_ls = [self._wt_transform(p, gain) for p in csum_p_ls]
        weight_j_ls = [self._wt_transform(p, gain) for p in csum_p_ls[1:]]+[0]
            
        return [i-j for i, j in zip(weight_i_ls, weight_j_ls)]