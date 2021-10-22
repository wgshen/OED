import numpy as np
from bayes_opt import BayesianOptimization
import emcee
from .utils import *

class OED(object):
    """
    A Bayesian optimal experimental design class.
    This class provides design of simulation-based experiments from a Bayesian 
    perspective, accommodates both batch and greedy design, and continuous 
    design space.
    This version only uses Bayesian optimization, we will provide more 
    optimization choices in the future.

    Parameters
    ----------
    model_fun : function
        Forward model function G(theta, d). It will be abbreviated as m_f 
        inside this class.
        The forward model function should take following inputs:
            * theta, numpy.ndarray of size (n_sample or 1, n_param)
                Parameter samples.
            * d, numpy.ndarray of size (n_sample or 1, n_design)
                Designs.
        and the output is 
            * numpy.ndarray of size (n_sample, n_obs).
        When the first dimension of theta or d is 1, it should be augmented
        to align with the first dimension of other inputs (i.e., we reuse it for
        all samples).
        The model function should accommodate multiple stages or batches.
    n_param : int
        Dimension of parameter space.
    n_design : int
        Dimension of design space. n_design should be equal
        #design variables * #stages * #batches(repeats), and the model function
        should accommodate multiple stages or batches.
    n_obs : int
        Dimension of observation space.
    prior_rvs : function
        It takes number of samples as input, and output a numpy.ndarray of size
        (n_sample, n_param) which are prior samples.
    design_bounds : list, tuple or numpy.ndarray of size (n_design, 2)
        It includes the constraints of the design variable. In this version, we
        only allow to set hard limit constraints. In the future version, we may
        allow users to provide their own constraint function.
        The length of design_bounds should be n_design.
        k-th entry of design_bounds is a list, tuple or numpy.ndarray like 
        (lower_bound, upper_bound) for the limits of k-th design variable.
    noise_info : list, tuple or numpy.ndarray of size (n_obs, 3)
        It includes the statistics of additive Gaussian noise.
        The length of noise_info should be n_obs. k-th entry of noise_info is a 
        list, tuple or numpy.ndarray including
            * noise_loc : float or int
            * noise_base_scale : float or int
                It will be abbreviated as noise_b_s in this class.
            * noise_ratio_scale : float or int
                It will be abbreviated as noise_r_s in this class.
        The corresponding noise will follow a gaussian distribution with mean
        noise_loc, std (noise_base_scale + noise_ratio_scale * abs(G)).
    prior_logpdf : function, optional(default=None)
        This function is required if users want to use post_logpdf, post_pdf
        and post_rvs function. It provides logpdf of samples in the prior
        distribution. The input is
            * theta, numpy.ndarray of size (n_sample or 1, n_param)
        and the output is
            * numpy.ndarray of size(n_sample).
    reward_fun : function, optional(default=None)
        User-provided non-KL-divergence based reward function. 
        It will be abbreviated as nlkd_rw_f inside this 
        class.
        The reward function should take following inputs:
            * d : np.ndarray of size (n_design)
                The design variable.
            * y : np.ndarray of size (n_obs)
                The observation.
        and the output is 
            * A float which is the reward.
        Note that the information gain is computed within this class, and does
        not needed to be included in reward_fun.
        When reward_fun is None, we only consider information gain.
    random_state : int, optional(default=None)
        It is used as the random seed.   
    
    Methods
    -------
    post_logpdf(), post_pdf()
        Evaluate the posterior logpdf (pdf) of parameter samples.
    post_rvs()
        Generate samples from the posterior.
    exp_utility()
        Estimate the expected utility of a given design.
    oed()
        Find the optimal design that maximizes the expected utility using
        Bayesian optimization.


    Notes
    -----
    Please refer to https://arxiv.org/pdf/1108.4146.pdf for more details.
    """
    def __init__(self, model_fun, n_param, n_design, n_obs, 
                 prior_rvs, design_bounds, noise_info, 
                 prior_logpdf=None, reward_fun=None, random_state=None):
        NoneType = type(None)
        assert isinstance(random_state, (int, NoneType)), (
               "random_state should be an integer or None.")
        np.random.seed(random_state)
        self.random_state = random_state
        
        assert callable(model_fun), (
               "model_fun should be a function.")
        self.m_f = model_fun
        self.model_fun = self.m_f

        assert isinstance(n_param, int) and n_param > 0, (
               "n_param should be an integer greater than 0.")
        self.n_param = n_param
        assert isinstance(n_design, int) and n_design > 0, (
               "n_design should be an integer greater than 0.")
        self.n_design = n_design
        assert isinstance(n_obs, int) and n_obs > 0, (
               "n_obs should be an integer greater than 0.")
        self.n_obs = n_obs

        assert callable(prior_rvs), (
               "prior_rvs should be a function to generate prior samples.")
        self.prior_rvs = prior_rvs

        assert isinstance(design_bounds, (list, tuple, np.ndarray)), (
               "design_bounds should be a list, tuple or numpy.ndarray of " 
               "size (n_design, 2).")
        assert len(design_bounds) == n_design, (
               "Length of design_bounds should equal n_design.")
        for i in range(n_design):
            assert len(design_bounds[i]) == 2, (
                   "Each entry of prior_info is of size 2, including "
                   "lower bound and upper bound.")
            l_b, u_b = design_bounds[i]
            assert isinstance(l_b, (int, float)), (
                   "{}-th lower bound should be a number.".format(i))
            assert isinstance(u_b, (int, float)), (
                   "{}-th upper_bound should be a number.".format(i))
        # size (n_design, 2)
        self.design_bounds = np.array(design_bounds)

        assert isinstance(noise_info, (list, tuple, np.ndarray)), (
               "noise_info should be a list, tuple or numpy.ndarray of " 
               "size (n_obs, 3).")
        assert len(noise_info) == n_obs, (
               "Length of noise_info should equal n_obs.")
        for i in range(n_obs):
            assert len(noise_info[i]) == 3, (
                   "Each entry of noise_info is of size 3, including "
                   "noise_loc, noise_base_scale and noise_ratio_scale.")
            noise_loc, noise_b_s, noise_r_s = noise_info[i]
            assert isinstance(noise_loc, (int, float)), (
                   "{}-th noise_loc should be a number.".format(i))
            assert isinstance(noise_b_s, (int, float)), (
                   "{}-th noise_base_scale should be a number.".format(i))
            assert isinstance(noise_r_s, (int, float)), (
                   "{}-th noise_ratio_scale should be a number.".format(i))
            assert noise_b_s ** 2 + noise_r_s ** 2 > 0, (
                   "Either {}-th noise_base_scale or noise_ratio_scale "
                   "should be greater than 0.".format(i))
        # size (n_obs, 3)
        self.noise_info = np.array(noise_info)
        self.noise_loc = self.noise_info[:, 0]
        self.noise_b_s = self.noise_info[:, 1]
        self.noise_r_s = self.noise_info[:, 2]

        if prior_logpdf is not None:
            assert callable(prior_logpdf), (
                   "prior_logpdf should be a function.")
        self.prior_logpdf = prior_logpdf

        # Non-KL-divergence based reward function
        if reward_fun is None:
            self.nkld_rw_f = lambda *args, **kws: 0
        else:
            assert callable(reward_fun), (
                   "reward_fun should be a function.")
            self.nkld_rw_f = reward_fun

        self.optimizer = None
        self.thetas = None
        self.noises = None
        return

    def post_logpdf(self, thetas, 
                    d=None, y=None, include_prior=True):
        """
        A function to compute the log-probability of unnormalized posterior  
        after observing observations 'y' by conducting experiments under 
        designs 'd'.

        Parameters
        ----------
        thetas : numpy.ndarray of size (n_sample, n_param)
            The parameter samples whose log-posterior are required.
        d : numpy.ndarray of size (n_design), optional(default=None)
            Designs.
        y : numpy.ndarray of size (n_obs), optional(default=None)
            Observations.
        include_prior : bool, optional(default=True)
            Include the prior in the posterior or not. It not included, the
            posterior is just a multiplication of likelihoods.

        Returns
        -------
        A numpy.ndarray of size (n_sample) which are log-posteriors.
        """
        if include_prior:
            assert self.prior_logpdf is not None, (
                "prior_logpdf should be a function to evaluate prior PDFS, "
                "with input of size (n_sample, n_param), and output of "
                "size (n_sample).")
        if d is None: d = []
        if y is None: y = []
        logpost = np.zeros(len(thetas))
        if len(d) > 0:
            G = self.m_f(thetas, d.reshape(1, -1))
            loglikeli = norm_logpdf(y.reshape(1, -1), 
                                    G + self.noise_loc,
                                    self.noise_b_s + self.noise_r_s * np.abs(G))
            logpost += loglikeli
        if include_prior:
            logpost += self.prior_logpdf(thetas)
        return logpost

    def post_pdf(self, *args, **kws):
        return np.exp(self.post_logpdf(*args, **kws))

    def post_rvs(self, n_sample, 
                 d=None, y=None):
        """
        A function to generate samples from the posterior distribution,
        after observing observations 'y' by conducting experiments under designs
        'd'.

        Parameters
        ----------
        n_sample : int
            Number of posterior samples we want to generate.
        d : numpy.ndarray of size (n_design), optional(default=None)
            Designs.
        y : numpy.ndarray of size (n_obs), optional(default=None)
            Observations.

        Returns
        -------
        A numpy.ndarray of size (n_sample, n_param).
        """
        log_prob = lambda x : self.post_logpdf(x.reshape(-1, self.n_param), 
                                               d=d,
                                               y=y)
        n_dim, n_walkers = self.n_param, 2 * self.n_param
        theta0 = self.prior_rvs(n_walkers)
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob)
        sampler.run_mcmc(theta0, 
                         int(n_sample // n_walkers * 1.2), 
                         progress=False)
        return sampler.get_chain().reshape(-1, self.n_param)[-n_sample:]

    def exp_utility(self, d, thetas, noises=None):
        """
        A function to estimate the expected utility 
        U(d) = int_y p(y|d) int_theta p(theta|d,y) *
                                      ln{p(theta|d,y) / p(theta)} dtheta dy
        on given design 'd' with given prior sample "thetas". 
        User provided non-KL-divergence based reward will also be considered.

        Parameters
        ----------
        d : numpy.ndarray of size (n_design)
            Designs.
        thetas : numpy.ndarray of size (n_sample, n_param)
            Prior samples.
        noises : numpy.ndarray of size (n_sample, n_obs), optional(default=None)
            Noises sampled from standard normal distribution. Using same noises
            on different designs can make the results more consistent and 
            smoothen the utility plots.

        Returns
        -------
        A float which is the expected utility.
        """
        if noises is not None:
            assert len(thetas) == len(noises) and noises.shape[1] == self.n_obs 
        n_sample = len(thetas)
        Gs = self.m_f(thetas, d.reshape(1, -1))
        if noises is None:
            noises = np.random.normal(size=Gs.shape)
        ys = (Gs +
              self.noise_loc +
              (self.noise_b_s + self.noise_r_s * np.abs(Gs)) * noises)
        loglikelis = norm_logpdf(ys, Gs + self.noise_loc,
                                 self.noise_b_s + self.noise_r_s * np.abs(Gs))
        evids = np.zeros(n_sample)
        nkld_rewards = np.zeros(n_sample)
        for i in range(n_sample):
            inner_likelis = norm_pdf(ys[i:i + 1],
                                     Gs + self.noise_loc,
                                     self.noise_b_s + 
                                     self.noise_r_s * np.abs(Gs))
            evids[i] = np.mean(inner_likelis)
            nkld_rewards[i] = self.nkld_rw_f(d, ys[i])
        return (loglikelis - np.log(evids) + nkld_rewards).mean()

    def oed(self, n_sample=1000, n_init=10, n_iter=90,
            restart=False):
        """
        A function to find the optimal design with maximum expected utility.

        Parameters
        ----------
        n_sample : int, optional(default=1000)
            Number of samples we will use to estimate the expected utility.
        n_init : int, optional(default=10)
            Number of initial exploration of Bayesian optimization.
        n_iter : int, optional(default=90)
            Number of iterations of Bayesian optimization after initial search.
        restart : bool, optional(default=False)
            If True, will reinitialize the optimizer.

        Returns
        -------
        A numpy.ndarray of size (n_design) which is the optimal design.
        A float which is the maximum expected utility.
        """
        print("Optimizing")
        pbounds = {}
        for i in range(self.n_design):
            pbounds['d' + str(i + 1)] = self.design_bounds[i, :]
        if self.thetas is None:
            self.thetas = self.prior_rvs(n_sample)
            self.noises = np.random.normal(size=(n_sample, self.n_obs))
        elif len(self.thetas) < n_sample:
            self.thetas = np.r_[self.thetas, self.prior_rvs(n_sample - 
                                                            len(self.thetas))]
            self.noises = np.r_[self.noises, 
                                np.random.normal(size=(n_sample - 
                                                       len(self.noises),
                                                       self.n_obs))]
        else:
            self.thetas = self.thetas[:n_sample]
            self.noises = self.noises[:n_sample]
        objective = lambda **kws: self.exp_utility(np.array(list(kws.values())),
                                                   self.thetas,
                                                   self.noises)
        if self.optimizer is None or restart:
            self.optimizer = BayesianOptimization(
                f=objective, pbounds=pbounds,
                random_state=self.random_state)
        self.optimizer.maximize(init_points=n_init,
                                n_iter=n_iter,
                                acq='ucb')
        self.d_opt = np.array(list(self.optimizer.max['params'].values()))
        self.U_opt = self.optimizer.max['target']
        self.opt_res = self.optimizer.res
        return self.d_opt, self.U_opt