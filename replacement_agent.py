import os
import numpy as np
from numpy import ndarray
import scipy.optimize as spo
from tqdm import tqdm

import my_utils as mu
from basic_agent import Agent
from saver import Saver
from simulator import Simulator


class ReplacementAgent(Agent):
    def __init__(self, cf: dict) -> None:
        super().__init__(cf=cf)
        self.saver = Saver(save_dir="./perf/maint_perf", save_subdir="best_perf")
        return

    # ===========================================================================
    # =========================== Heuristic policies ============================
    # ===========================================================================

    def r_opt_pol(self, tfs: ndarray) -> list:
        """
        Ideal policy. Dilemma here on how to include failed components before first possible
        decision time. Either:
        - exclude samples -> decreased sample size & changed distribution
        - include samples -> smaller M due to higher failure costs in optimal policy
        I opted here for 2nd option; ideally you check beforehand how many samples fall in
        this category, and choose your parameters such that not too many samples are excluded
        -> for the parameters outlined in the paper, no sample was excluded
        """
        costs = np.zeros(tfs.shape)
        tlcs = np.zeros(tfs.shape)
        # check number of failures before first decision time:
        #   -> c_c costs and tf tlcs
        c_c_ind = np.where(tfs <= self.S.cf["Delta_T"])[0]
        if np.size(c_c_ind) != 0:
            costs[c_c_ind] = self.S.cf["c_c"]
            tlcs[c_c_ind] = tfs[c_c_ind]

        # otherwise:
        #   -> c_p costs last decision time before tf as tlcs
        valid_ind = np.where(tfs > self.S.cf["Delta_T"])[0]
        if np.size(valid_ind) != 0:
            tlcs_cp = (tfs[valid_ind] // self.S.cf["Delta_T"]) * self.S.cf["Delta_T"]
            ectr_cp = self.S.cf["c_p"] / tlcs_cp
            ectr_cc = self.S.cf["c_c"] / tfs[valid_ind]

            let_fail_ind = ectr_cc < ectr_cp

            costs[valid_ind[let_fail_ind]] = self.S.cf["c_c"]
            costs[valid_ind[~let_fail_ind]] = self.S.cf["c_p"]

            tlcs[valid_ind[let_fail_ind]] = tfs[valid_ind[let_fail_ind]]
            tlcs[valid_ind[~let_fail_ind]] = tlcs_cp[~let_fail_ind]
        return self.output_checker(c=costs, t=tlcs)

    def rh1(self, mu_ln_RUL_pred_k: ndarray) -> ndarray:
        """
        Simple threshold based policy with default c_p/c_c.

        Args:
            mu_ln_RUL_pred_k: mean parameter of ln RUL distribution

        Returns:
            a: action vector
        """
        a = self.Pr_smaller_thresh(
            mu_ln_RUL_pred_k=mu_ln_RUL_pred_k,
            ub=self.S.cf["Delta_T"],
            p_thres=self.S.cf["c_p"] / self.S.cf["c_c"],
        )
        return a

    def rh1_opt(self, mu_ln_RUL_pred_k: ndarray, p_thres: float) -> ndarray:
        """
        Simple threshold based policy with given threshold.

        Args:
            mu_ln_RUL_pred_k: mean parameter of ln RUL distribution
            p_thres: probability treshold that triggers preventive replacement

        Returns:
            a: action vector
        """
        a = self.Pr_smaller_thresh(
            mu_ln_RUL_pred_k=mu_ln_RUL_pred_k, ub=self.S.cf["Delta_T"], p_thres=p_thres
        )
        return a

    def single_f_Trk_rh2(
        self, m: float, S: Simulator, ectr: float, iaccr: float, Trk: float
    ) -> ndarray:
        """
        Single evaluation of the optimization function in Eq. 13.

        Args:
            m: mean parameter of ln RUL distribution
            S: Simulator object containing necessary functions
            ectr: assumed ectr value
            iaccr: prescribed accuracy for the integration
            Trk: prescripted replacement time

        Returns:
            g: function evaluation
        """
        Ppr = 1 - S.RUL_pred_k_cdf(x=Trk, mu_ln_RUL_pred_k=m)
        g = (
            Ppr * S.cf["c_p"]
            + (1 - Ppr) * S.cf["c_c"]
            + ectr
            * self.single_integral(
                m=m, S=S, shift=Trk, ilb=Trk, iub=np.inf, iaccr=iaccr
            )
        )
        return g

    def single_optimization(
        self,
        m: float,
        S: Simulator,
        ectr: float,
        iaccr: float,
        olb: float,
        oub: float,
        oacca: float = 1e-3,
    ) -> ndarray:
        """
        Single optimization of optimal replacement time.

        Args:
            m: mean parameter of ln RUL distribution
            S: Simulator object containing necessary functions
            ectr: assumed ectr value
            iaccr: prescribed accuracy for the integration
            olb: optimization lower bound
            oub: optimization upper bound
            oacca: required optimization accuracy

        Returns:
            optimal replacement time
        """
        res = spo.minimize_scalar(
            lambda T: self.single_f_Trk_rh2(
                m=m, S=S, ectratio=ectr, iaccr=iaccr, Trk=T
            ),
            bounds=(olb, oub),
            tol=oacca,
        )
        return res.x

    def rh2(self, mu_ln_RUL_pred_k: ndarray, ectr: float) -> ndarray:
        """
        Decide to repair based on an ideal replacement time.

        Args:
            mu_ln_RUL_pred_k: mean parameter of ln RUL distribution
            ectr: assumed ectr value

        Returns:
            a: action vector
        """
        T_opts = self.P.parallel_runner(
            A=self,
            f="multi_optimization_rh2",
            m=mu_ln_RUL_pred_k,
            ectr=ectr,
            iaccr=self.S.cf["iaccr"],
            olb=0,
            oub=1.5 * self.S.cf["Delta_T"],
            oacca=self.S.cf["oacca"],
            rounds_chooser=True,
        )
        a = np.zeros(np.shape(mu_ln_RUL_pred_k))  # default is do nothing
        a[self.S.cf["Delta_T"] >= T_opts] = 1
        return a

    def rdoa(self, mu_ln_RUL_pred_k: ndarray, ectr: float) -> ndarray:
        """
        Discrete option assessment.

        Args:
            mu_ln_RUL_pred_k: mean parameter of ln RUL distribution
            ectr: assumed ectr value

        Returns:
            a: action vector
        """
        rep_now = ectr * self.S.cf["Delta_T"]
        pf = self.S.RUL_pred_k_cdf(
            x=self.S.cf["Delta_T"], mu_ln_RUL_pred_k=mu_ln_RUL_pred_k
        )
        cond_rul = self.S.cond_rul_smaller(m=mu_ln_RUL_pred_k, iub=self.S.cf["Delta_T"])
        rep_later = -ectr * cond_rul + pf * (
            self.S.cf["c_c"] - self.S.cf["c_p"] + ectr * self.S.cf["Delta_T"]
        )
        a = np.zeros(np.shape(mu_ln_RUL_pred_k))
        a[rep_later >= rep_now] = 1
        return a

    def rdoa_single_core(self, mu_ln_RUL_pred_k: ndarray, ectr: float) -> ndarray:
        """
        Single core implementation of above function -> can be wrapped in parallelizer.
        """
        rep_now = ectr * self.S.cf["Delta_T"]
        pf = self.S.RUL_pred_k_cdf(
            x=self.S.cf["Delta_T"], mu_ln_RUL_pred_k=mu_ln_RUL_pred_k
        )
        cond_rul = self.single_core_cond_int(
            mu_vec=mu_ln_RUL_pred_k,
            ilb=0,
            iub=self.S.cf["Delta_T"],
            shift=0,
            iaccr=self.S.cf["iaccr"],
        )
        rep_later = -ectr * cond_rul + pf * (
            self.S.cf["c_c"] - self.S.cf["c_p"] + ectr * self.S.cf["Delta_T"]
        )
        a = np.zeros(np.shape(mu_ln_RUL_pred_k))  # default is do nothing
        a[rep_later >= rep_now] = 1
        return a

    # return a cost vector and state vector associated with taking an action vector
    def step(
        self,
        s: ndarray,
        a: ndarray,
        c: ndarray,
        t: ndarray,
        f: ndarray,
        time: int,
    ) -> list:
        """
        Performs a single update step.

        Args:
            s: state array
            a: action array
            c: cost array
            t: life time array
            f: failure time array
            time: current time

        Returns:
            s: new state array
            c: new cost array
            t: new life time array
        """
        # check for actions, set states to 0, where a=1
        a_ind = np.where(a == 1)[0]
        if np.size(a_ind) != 0:
            c[a_ind] = self.S.cf["c_p"]
            t[a_ind] = time
            s[a_ind] = 0

        not_a_ind = a == 0
        fail = f < time + self.S.cf["Delta_T"]
        if (np.sum(fail) > 0) and np.sum(not_a_ind) > 0:
            f_ind = np.where(np.logical_and(fail, not_a_ind))[0]
            # f_ind = np.where((f < time+self.S.cf['Delta_T']) * (a==0))[0]
            if np.size(f_ind) > 0:
                c[f_ind] = self.S.cf["c_c"]
                t[f_ind] = f[f_ind]
                s[f_ind] = 0
        return s, c, t

    def eval_rh(
        self,
        fun: str,
        tfs: ndarray,
        muks: ndarray,
        opt_mode: bool = False,
        par: float = None,
    ) -> list:
        """
        Evalulation of a heuristic for a given cost ratio.

        Args:
            fun: heuristic to evaluate
            tfs: array of failure times
            muks: nan-padded array RUL PDF params for all time stamps
            opt_mode: boolean flag for optimization handling. Decides whether to return
                all values or only R from the output_checker
            par: potential parameter for some heuristics

        Returns either:
            a) [cost_vec, tlc_vec, R, var_R]
            b) [R]
            depending on the opt_mode flag
        """
        # actually failures before Delta_t are not implemented in the heuristics check!!
        # state vector, keeps a record of all replaced / failed components
        states = np.ones(tfs.shape, dtype=bool)
        costs = np.zeros(tfs.shape)
        tlcs = np.zeros(tfs.shape)
        self.logger.info(f"par: {par}")
        for k in range(muks.shape[1]):
            if (k % 5) == 0:
                print(f"Step: {k}")
                self.logger.info(f"Step: {k}")
            # get True/False non-nan indices
            if (np.sum(~np.isnan(muks[:, k])) > 0) and (np.size(muks[states, k]) > 0):

                if np.sum(np.isnan(muks[states, k])) > 0:
                    raise Exception("nan value in mu detected!!")

                # calculate actions -> call individual heuristics
                act = np.zeros(states.shape, bool)
                if fun == "rh1":
                    act[states] = self.rh1(muks[states, k])
                elif fun == "rh1_opt":
                    act[states] = self.rh1_opt(muks[states, k], p_thres=par)
                elif fun in ["rh2", "rh2_it", "rh2_optimized"]:
                    act[states] = self.rh2(muks[states, k], ectr=par)
                elif fun in ["rdoa", "rdoa_it", "rdoa_optimized"]:
                    act[states] = self.rdoa(muks[states, k], ectr=par)
                elif fun == "rdoa_single_core":
                    act[states] = self.rdoa_single_core(muks[states, k], ectr=par)
                else:
                    raise Exception(
                        f"Function {fun} not found in list of provided functions!"
                    )

                s2, c2, t2 = self.step(
                    states[states],
                    act[states],
                    costs[states],
                    tlcs[states],
                    tfs[states],
                    (k + 1) * self.S.cf["Delta_T"],
                )
                costs[states] = c2
                tlcs[states] = t2
                states[states] = s2

        # evaluate those trajectories, where performed DN at last decision time before failure
        if np.sum(states) != 0:
            raise RuntimeError()
        return self.output_checker(c=costs, t=tlcs, o=opt_mode)

    def eval_mult_c_ratios(
        self,
        c_rs: ndarray = None,
        fun: str = "r_opt_pol",
        tfs: ndarray = None,
        mu_ln_RUL_preds: ndarray = None,
        par: float = None,
        sub_ind: float = None,
        calc_M: int = True,
        inter_save: bool = False,
        save: bool = False,
        savename: str = None,
    ) -> ndarray:
        """
        Evaluates a given policy for a whole range of cost ratios.

        Args:
            c_rs: array (N_c_rs,) of cost ratios to evaluate the performance
            fun: string specifying the heuristic to choose
            tfs: array (N_samples,) of failure times
            mu_ln_RUL_preds: array (N_samples,N_max_steps) of parameters characterizing the RUL
                prediction for the respective time steps. All samples are nan padded to have the
                the same length, handling is done in the function
            par: potential parameter for each heuristic, e.g., factor for ectr initialization
            sub_ind: array (N_indices,) of random sample indices for which optimization is performed
                in order to investigate sample size effects for each heuristic
            calc_M: boolean flag to indicate whether you want to calculate the percentage-wise
                deviation from the ideal policy
            inter_save: boolean flag to save each c_r immediately
            save: boolean flag for saving the results
            savename: string specifying the name for saving the results

        Returns:
            R: array (N_c_rs,) of performances of the heuristic for each cost ratio
            var_R: array (N_c_rs,) of the corresponding performance variances
            M: array (N_c_rs,) of the percentage-wise deviations from perfect policy, None if
                calc_M flag is False
            var_M: array (N_c_rs,) of the corresponding M variances, None if calc_M flag is False
        """
        par_vec = []
        c_p_init = self.S.cf["c_p"]
        if c_rs is None:
            c_rs = np.array(
                [k / 100 for k in range(1, 10)] + [k / 10 for k in range(1, 10)]
            )

        assert (tfs is not None) or (mu_ln_RUL_preds is not None)
        # overwrite logger to be specific for heuristic function
        self.logger = self.my_logger(file_name=f"{fun}.log")
        Rs = np.zeros(c_rs.shape)
        var_Rs = np.zeros(c_rs.shape)
        for k, c_r in tqdm(enumerate(c_rs)):
            self.S.cf["c_p"] = c_r * self.S.cf["c_c"]

            if fun == "r_opt_pol":
                _, _, Rs[k], var_Rs[k] = self.r_opt_pol(tfs=tfs)

            elif fun == "rh1_opt":
                # sub_ind are random indices to select tf samples used for optimization
                sub_ind = np.arange(np.size(tfs)) if sub_ind is None else sub_ind
                res = spo.minimize_scalar(
                    lambda p: self.eval_rh(
                        fun=fun,
                        tfs=tfs[sub_ind],
                        muks=mu_ln_RUL_preds[sub_ind, :],
                        opt_mode=True,
                        par=p,
                    ),
                    bounds=(0, 1),
                )
                # evaluation still on full population
                _, _, Rs[k], var_Rs[k] = self.eval_rh(
                    fun=fun,
                    tfs=tfs,
                    muks=mu_ln_RUL_preds,
                    opt_mode=False,
                    par=res.x,
                )
                par_vec.append(res.x)

            elif fun in ["rh2", "rh2_opt1", "rh2_opt2", "rh2_opt3"]:
                ectr = self.S.rh_ectr(fun=fun, fac=par)
                _, _, Rs[k], var_Rs[k] = self.eval_rh(
                    fun=fun.split("_")[0], tfs=tfs, muks=mu_ln_RUL_preds, par=ectr
                )
                par_vec.append(ectr)
                if inter_save:
                    self.saver.inter_save_heur(
                        fun, c_r, Rs, var_Rs, ectr, np.size(tfs), None
                    )

            elif fun == "rh2_it":
                # opt par is here the index of the current iteration -> searches in best_perf for previous iteration performance
                init_opt = "opt1" if par is None else par
                ectr_init = mu.previous_heur_ectr(c_r=c_r, opt=init_opt)
                _, _, Rs[k], var_Rs[k] = self.eval_rh(
                    fun=fun, tfs=tfs, muks=mu_ln_RUL_preds, par=ectr_init
                )
                par_vec.append(ectr_init)
                if inter_save:
                    self.saver.inter_save_heur(
                        fun, c_r, Rs, var_Rs, ectr_init, np.size(tfs), init_opt
                    )

            elif fun in ["rh2_optimized", "rdoa_optimized"]:
                sub_ind = np.arange(np.size(tfs)) if sub_ind is None else sub_ind
                self.logger.info(f"par: {par}, cf: {str(self.S.cf)}")
                res = spo.minimize_scalar(
                    fun=lambda p: self.eval_rh(
                        fun=fun,
                        tfs=tfs[sub_ind],
                        muks=mu_ln_RUL_preds[sub_ind, :],
                        opt_mode=True,
                        par=p,
                    ),
                    method="bounded",
                    bounds=(0, 4),
                    tol=self.S.cf["ectr_oacca"],
                    options={
                        "maxiter": self.S.cf["maxiter"],
                        "disp": 3,
                    },
                )
                self.logger.info(f"optimization result: {str(res)}")
                _, _, Rs[k], var_Rs[k] = self.eval_rh(
                    fun=fun,
                    tfs=tfs,
                    muks=mu_ln_RUL_preds,
                    opt_mode=False,
                    par=res.x,
                )
                par_vec.append(res.x)
                if inter_save:
                    self.saver.inter_save_heur(
                        fun, c_r, Rs, var_Rs, res.x, np.size(tfs), res.x
                    )

            elif fun in ["rdoa", "rdoa_opt1", "rdoa_opt2", "rdoa_opt3"]:
                if sub_ind is None:
                    ectr = self.S.rh_ectr(fun=fun, fac=par)
                else:
                    ectr = self.S.rh_ectr_sample_based(
                        fun=fun, tfs=tfs[sub_ind], fac=par
                    )

                _, _, Rs[k], var_Rs[k] = self.eval_rh(
                    fun=fun.split("_")[0],
                    tfs=tfs,
                    muks=mu_ln_RUL_preds,
                    par=ectr,
                )
                par_vec.append(ectr)
                if inter_save:
                    self.saver.inter_save_heur(
                        fun, c_r, Rs, var_Rs, par, np.size(tfs), None
                    )

            else:
                _, _, Rs[k], var_Rs[k] = self.eval_rh(
                    fun=fun, tfs=tfs, muks=mu_ln_RUL_preds
                )

        if calc_M:
            opt_perf = mu.load_file(
                filename=os.path.join(
                    self.saver.save_dir, self.saver.save_subdir, "r_opt_pol_1e6"
                )
            )
            Ms, var_Ms = mu.calc_M(R=Rs, R_perfect=opt_perf["R"], var_R=var_Rs)
        else:
            Ms, var_Ms = None, None

        if save:
            if savename is None:
                savename = fun + mu.convert_int_to_estr(np.size(tfs))
            self.saver.final_save_heur(
                savename, np.size(tfs), c_r, Rs, var_Rs, Ms, var_Ms, par_vec
            )

        self.S.cf["c_p"] = c_p_init
        return Rs, var_Rs, Ms, var_Ms
