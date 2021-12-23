#!/usr/bin/env python
# coding: utf-8

import datetime as dt
import numpy as np


class BaseConstructor:
    
    def __init__(self, *args):
        constructor_vals = locals()['args']
        constructor_parms = [k for k in self.__init__.__code__.co_varnames if k != 'self']
        self.init_args = {k:v for k,v in zip(constructor_parms, constructor_vals)}
        
        for name, val in self.init_args.items():
            setattr(self, name, val)
            

class PTAgent(BaseConstructor):
    """Generate a Prospect Theory agent that expresses preferences between two lotteries according Kahneman & Tversky (1979). 
    
    PTAgent expresses preference between lotteries by implementing value and 
    probability weighting functions following Tversky & Kahneman, 
    Prospect Theory: An Analysis of Decision Under Risk (1972). The functional form
    for the value function is taken from Kahneman & Tversky's 
    Advances in Prospect Theory: Cumulative Representation of Uncertainty (1992). 
    The functional form of the weighting function is taken from this website:
    https://sites.duke.edu/econ206_01_s2011/files/2011/04/39b-Prospect-Theory-Kahnemann-Tversky_final2-1.pdf
    
    Attributes
    ------------
    parameters: dict
        - a mapping of value function and weighting function parameter names (keys) and parameter 
        inputs (values) 
    _value_fn: function
        - input floating point representing outcome value
        - output calculation of value function based on input, alpha, beta (optional), and lambda_ 
        parameters
    _gain_wt_fn: function
        - input floating point representing probability
        - output calculation of weighting function for probabilities associated with non-negative 
        outcomes 
        based on probability and delta parameter
        
    Methods
    ------------
    evaluate(lottery:dict, verbose:bool=False)
    choose(lottery1:dict=None, lottery2:dict=None)
    """
    
    def __init__(self, alpha:float=1, beta:float=None, gamma:float=1, delta:float=None, lambda_:float=1):
        """
        Parameters
        ------------
        alpha: float {a | 0 <= a <= 1}, default=1
            - controls the rate of change in value over gains
        beta (optional): float {b | 0 <= b <= 1}, default=None
            - controls the rate of change in value over losses
        lambda_: float {l| l > 1}, default=1
            - controls the steepness of the value function over losses relative to gains 
        delta: float {d| 0 <= d <= 1}, default=None
            - a weighting function parameter controlling the shape of the transformation of probability
        """
        
        super().__init__(alpha, beta, gamma, delta, lambda_)
        
        beta = beta or alpha
        delta = delta or gamma
        
        self.dt = dt.datetime.now()
        
        self.parameters = {
            'alpha':alpha, 
            'beta':beta, 
            'gamma':gamma,
            'delta':delta,
            'lambda_':lambda_
        }
        
        # Error handling for parameters
        type_check = {k: True for k, v in self.parameters.items() if not isinstance(v, (float, int))}
        if len(type_check) > 0:
            raise TypeError('TypeError: ' + ', '.join(self.parameters.keys()) +
                            ' parameters must be int or float type')
            
        zero_one_check = {k: v < 0 or v > 1 for k, v in self.parameters.items() if k != 'lambda_'}
        if any(zero_one_check.values()):
            param_keys = ', '.join(k for k in self.parameters.keys() if k != 'lambda_')
            raise ValueError('ValueError: ' + param_keys + 
                             ' parameters must lie within 0 and 1 inclusive')

        # value and weighting functions
        self._value_fn = lambda x: x**alpha if x >= 0 else -lambda_*((-x)**beta)
        self._gain_wt_fn = lambda p: np.exp(-(-np.log(p))**gamma)
        self._loss_wt_fn = lambda p: np.exp(-(-np.log(p))**delta)
        
    def __repr__(self):
        
        repr_ls = ['PTAgent('] 
        for k, v in self.parameters.items():
            repr_ls.append(k + '=' + str(v) + ',')
            
        repr_str = '\n '.join(repr_ls) + '\n)\n @ ' + str(self.dt)
        return repr_str
    
    def _validate_lottery(self, lottery:dict):

        try:
            type_check = {k: not isinstance(v, (list, tuple)) for k, v in lottery.items()}
            len_check = len(lottery['outcome']) == len(lottery['probability'])
            zero_one_check = [p.__lt__(0) or p.__gt__(1) for p in lottery['probability']]
            sum_check = sum(lottery['probability']) <= 1
            
            if any(type_check.values()): 
                raise TypeError('Values in lottery must be iterables of type list or tuple')  
            elif any(zero_one_check):
                raise ValueError('ValueError: lottery["probability"] values must lie within 0 and 1 inclusive')
            assert len_check, ('AssertionError: Length of lottery["outcome"] must match length of lottery["probability"]')
            assert sum_check, ('AssertionError: Sum of lottery["probability"] must not exceed one')
        except TypeError as te:
            print(te)
        except ValueError as ve:
            print(ve)
        except AssertionError as ae:
            print(ae)
    
    def evaluate(self, lottery:dict, verbose:bool=False)->float:
        """Evaluate a lottery.
        
        Assigns valuation to a lottery.
        
        Parameters
        ----------
        lottery: dict 
            - contains two keys: 'outcome' and 'probability'.
            - 'outcome' key maps to a list of real numbers corresponding to outcomes. 
            - 'probability' key maps to a list of floats between 0 and 1.
            Length of list associated with 'probability' must match length of list associated 
            with 'outcome'.
        verbose: bool 
            - True: output list of outcome, probability tuples sorted by outcome; capacities; 
            valuation
            - False: output valuation
        
        Returns
        ----------
        valuation (if verbose=False) 
            - float representing overall value of lottery to PTAgent instance
        dict (if verbose=True)
            - valuation : float representing overall value of lottery to PTAgent instance
            - lottery : list of tuples containing outcome, probability pairs sorted by outcome
            - lottery_trans : list of outcomes after value transformation function 
        """
        
        # consistency checks
        self._validate_lottery(lottery)
        
        lottery_ls = [(o, p) for o, p in zip(lottery['outcome'], lottery['probability'])]
        lottery_trans_ls = [(self._value_fn(o), self._wt_transform(p, gain=o>=0)) for (o, p) in lottery_ls]
        
        if (len(lottery_ls) == 2) & (sum(tup[1] for tup in lottery_ls) < 1) & (all(tup[0] >= 0 for tup in lottery_ls) | all(tup[0] < 0 for tup in lottery_ls)):
            min_abs_val_idx = np.argmin(abs(tup[0]) < 0 for tup in lottery_ls)
            max_abs_val_idx = 1 - min_abs_val_idx 
            low_abs_xpect = lottery_ls[min_abs_val_idx]
            high_abs_xpect = lottery_ls[max_abs_val_idx]
            valuation = low_abs_xpect[0] + high_abs_xpect[1]*(high_abs_xpect[0] - low_abs_xpect[0])
        else: 
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
            - boolean signifying whether the outcome associated with probability p is 
            non-negative

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
    """Generate a Cumulative Prospect Theory agent that expresses preferences between 
    two lotteries according to Kahneman & Tversky (1992). 
    
    Generates preferences between lotteries by implementing value and 
    probability weighting functions following Tversky & Kahneman, 
    Advances in Prospect Theory: Cumulative Representation of Uncertainty (1992). 
    CPTAgent inherits from parent class PTAgent, but overrides .evaluate() method. 
        
    Attributes
    ------------
    parameters: dictionary
        - contains parameter names as keys and parameter inputs as values
    value_fun: function
        - input floating point representing utility
        - output calculation of value function based on input, alpha, beta (optional), 
        and lambda_ parameters
    weight_fun: function
        - input floating point representing probability
        - output calculation of value function based on probability and delta parameter
    """
    
    def __init__(self, alpha:float=1, beta:float=None, gamma:float=1, delta:float=None, lambda_:float=1):
        """
        Parameters
        ------------
        alpha: float {a | 0 <= a <= 1}
            - controls the rate of change in value over gains
        beta (optional): float {b | 0 <= b <= 1}, default=None
            - controls the rate of change in value over losses
        lambda_: float {l| l > 1}
            - controls the steepness of the value function over losses relative to gains 
        delta: float {d| 0 <= d <= 1}
            - a weighting function parameter controlling the shape of the transformation 
            of probability
        """
        
        super().__init__(alpha, beta, gamma, delta, lambda_)
        
        beta = beta or alpha
        delta = delta or gamma
        
        # override PTAgent weighting functions with cumulative prospect theory weighting functions (Kahneman & Tversky, 1990)
        self._gain_wt_fn = lambda p: p**gamma/(p**gamma + (1-p)**gamma)**gamma
        self._loss_wt_fn = lambda p: p**delta/(p**delta + (1-p)**delta)**delta
        
    def __repr__(self):
        
        repr_ls = ['CPTAgent('] 
        for k, v in self.parameters.items():
            repr_ls.append(k + '=' + str(v) + ',')
            
        repr_str = '\n '.join(repr_ls) + '\n)\n @ ' + str(self.dt)
        return repr_str
    
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
        self._validate_lottery(lottery)
        
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