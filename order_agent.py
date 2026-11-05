import os
import numpy as np
from numpy import ndarray
import scipy.optimize as spo
from tqdm import tqdm

import my_utils as mu
from basic_agent import Agent
from saver import Saver


class OrderAgent(Agent):
    def __init__(self, cf: dict) -> None:
        super().__init__(cf=cf)
        self.saver = Saver(
            save_dir="./perf/order_perf", save_subdir=f"L{self.S.cf['L']}"
        )
        self.w = (
            int(np.ceil(self.S.cf["L"] / self.S.cf["Delta_T"])) * self.S.cf["Delta_T"]
        )
        return

    def o_opt_pol(self, tfs: ndarray) -> list:
        """
        Ideal policy. Dilemma here on how to include failed components before first possible
        component arrival. Either:
        - exclude samples -> decreased sample size
        - include samples -> smaller M due to higher failure costs in optimal policy
        I opted here for 1st option; ideally you check beforehand how many samples fall in
        this category, and choose your parameters such that not too many samples are excluded
        """
        costs = np.zeros(tfs.shape)
        tlcs = np.zeros(tfs.shape)
        # check number of late orders before first decision time:
        #   -> c_c costs and tf tlcs
        fail_ind = np.where(tfs <= self.w)[0]
        if np.size(fail_ind) != 0:
            raise RuntimeError()

        # otherwise:
        #   check if it is cheaper to let it fail or preventively order
        perm_ind = np.where(tfs > self.S.cf["Delta_T"])[0]
        if np.size(perm_ind) != 0:
            t_perf_inv = (
                np.floor((tfs[perm_ind] - self.S.cf["L"]) / self.S.cf["Delta_T"])
                * self.S.cf["Delta_T"]
            )
            costs_inv = self.S.cf["c_inv"] * (
                tfs[perm_ind] - (t_perf_inv + self.S.cf["L"])
            )

            t_perf_unav = t_perf_inv + self.S.cf["Delta_T"]
            costs_unav = self.S.cf["c_unav"] * (
                t_perf_unav + self.S.cf["L"] - tfs[perm_ind]
            )

            costs[perm_ind] = np.minimum(costs_inv, costs_unav)
            tlcs[perm_ind] = tfs[perm_ind]
            return self.output_checker(c=costs[perm_ind], t=tlcs[perm_ind])

    def oh1(self, mu_ln_RUL_pred_k: ndarray) -> ndarray:
        """
        Simple threshold based policy with default c_inv/c_unav.

        Args:
            mu_ln_RUL_pred_k: mean parameter of ln RUL distribution

        Returns:
            a: action vector
        """
        return self.Pr_smaller_thresh(
            mu_ln_RUL_pred_k=mu_ln_RUL_pred_k,
            ub=self.w,
            p_thres=self.S.cf["c_inv"] / self.S.cf["c_unav"],
        )

    def oh1_opt(self, mu_ln_RUL_pred_k: ndarray, p_thres: float) -> ndarray:
        """
        Simple threshold based policy with given threshold.

        Args:
            mu_ln_RUL_pred_k: mean parameter of ln RUL distribution
            p_thres: probability treshold that triggers preventive replacement

        Returns:
            a: action vector
        """
        return self.Pr_smaller_thresh(
            mu_ln_RUL_pred_k=mu_ln_RUL_pred_k, ub=self.w, p_thres=p_thres
        )

    def odoa(self, mu_ln_RUL_pred_k: ndarray, ectratio: float) -> ndarray:
        """
        Discrete option assessment.

        Args:
            mu_ln_RUL_pred_k: mean parameter of ln RUL distribution
            ectr: assumed ectr value

        Returns:
            a: action vector
        """
        pf_L = self.S.RUL_pred_k_cdf(
            x=self.S.cf["L"], mu_ln_RUL_pred_k=mu_ln_RUL_pred_k
        )
        pf_L_DT = self.S.RUL_pred_k_cdf(
            x=self.S.cf["L"] + self.S.cf["Delta_T"], mu_ln_RUL_pred_k=mu_ln_RUL_pred_k
        )
        pf_betw = pf_L_DT - pf_L

        cond_rul_betw = self.S.cond_rul_betw(
            m=mu_ln_RUL_pred_k,
            ilb=self.S.cf["L"],
            iub=self.S.cf["L"] + self.S.cf["Delta_T"],
        )

        cost_left = self.S.cf["c_inv"] * self.S.cf["Delta_T"]

        cost_right_1 = (self.S.cf["c_unav"] + self.S.cf["c_inv"] - ectratio) * (
            pf_betw * (self.S.cf["L"] + self.S.cf["Delta_T"]) - cond_rul_betw
        )

        cost_right_2 = (
            pf_L
            * self.S.cf["Delta_T"]
            * (self.S.cf["c_unav"] + self.S.cf["c_inv"] - ectratio)
        )

        cost_right = cost_right_1 + cost_right_2

        a = np.zeros(np.shape(mu_ln_RUL_pred_k))  # default is do nothing
        a[cost_right >= cost_left] = 1
        return a

    def odoa_single_core(self, mu_ln_RUL_pred_k: ndarray, ectratio: float) -> ndarray:
        """
        Single core implementation of above function -> can be wrapped in parallelizer.
        """
        pf_L = self.S.RUL_pred_k_cdf(
            x=self.S.cf["L"], mu_ln_RUL_pred_k=mu_ln_RUL_pred_k
        )
        pf_L_DT = self.S.RUL_pred_k_cdf(
            x=self.S.cf["L"] + self.S.cf["Delta_T"], mu_ln_RUL_pred_k=mu_ln_RUL_pred_k
        )
        pf_betw = pf_L_DT - pf_L

        cond_rul_betw = self.single_core_cond_int(
            mu_vec=mu_ln_RUL_pred_k,
            ilb=self.S.cf["L"],
            iub=self.S.cf["L"] + self.S.cf["Delta_T"],
            shift=0,
            iaccr=self.S.cf["iaccr"],
        )

        cost_left = self.S.cf["c_inv"] * self.S.cf["Delta_T"]
        try:
            cost_right_1 = (self.S.cf["c_unav"] + self.S.cf["c_inv"] - ectratio) * (
                pf_betw * (self.S.cf["L"] + self.S.cf["Delta_T"]) - cond_rul_betw
            )
        except:
            raise Exception(
                f"pf shape {pf_betw.shape}, cond rul shape {cond_rul_betw.shape}"
            )

        cost_right_2 = (
            pf_L
            * self.S.cf["Delta_T"]
            * (self.S.cf["c_unav"] + self.S.cf["c_inv"] - ectratio)
        )

        cost_right = cost_right_1 + cost_right_2

        a = np.zeros(np.shape(mu_ln_RUL_pred_k))  # default is do nothing
        a[cost_right >= cost_left] = 1
        return a

    # return a cost vector and state vector associated with taking an action vector
    def step(self, s: ndarray, a: ndarray, c: ndarray, f: ndarray, time: int) -> list:
        """
        Performs a single update step. Life times do not need to be set
        specificially, because all life times are simply the failure times.

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
        # check for actions
        # set states to 0, where a=1
        a_ind = a == 1
        if np.sum(a_ind) > 0:
            # check whether component fails before or after lead time
            fail_L = f < time + self.S.cf["L"]
            if np.sum(fail_L) > 0:
                fail_L_ind = np.where(np.logical_and(fail_L, a_ind))[0]
                if np.size(fail_L_ind) > 0:
                    c[fail_L_ind] = self.S.cf["c_unav"] * (
                        time + self.S.cf["L"] - f[fail_L_ind]
                    )

            not_fail_L = f >= time + self.S.cf["L"]
            if np.sum(not_fail_L) > 0:
                not_fail_L_ind = np.where(np.logical_and(not_fail_L, a_ind))[0]
                c[not_fail_L_ind] = self.S.cf["c_inv"] * (
                    f[not_fail_L_ind] - (time + self.S.cf["L"])
                )

            # either way, the life time ends
            s[a_ind] = 0

        # handle here case where no order placed, but component fails before next decision time
        not_a_ind = a == 0
        fail_T = f < time + self.S.cf["Delta_T"]
        if (np.sum(fail_T) > 0) and (np.sum(not_a_ind) > 0):
            fail_T_ind = np.where(np.logical_and(fail_T, not_a_ind))[0]
            # f_ind = np.where((f < time+self.S.cf['Delta_T']) * (a==0))[0]
            if np.size(fail_T_ind) > 0:
                c[fail_T_ind] = self.S.cf["c_unav"] * (
                    np.ceil(f[fail_T_ind] / self.S.cf["Delta_T"]) * self.S.cf["Delta_T"]
                    - f[fail_T_ind]
                    + self.S.cf["L"]
                )
                s[fail_T_ind] = 0
        return s, c

    def eval_oh(
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
            a) [R, var_R, M, var_M]
            b) [R]
            depending on the opt_mode flag
        """
        # actually failures before Delta_t are not implemented in the heuristics check!!
        # state vector, keeps a record of all replaced / failed components
        states = np.ones(tfs.shape, dtype=bool)
        costs = np.zeros(tfs.shape)
        self.logger.info(f"ectratio: {par}")
        for k in range(muks.shape[1]):
            # print(f"Run {k}")
            if (k % 5) == 0:
                print(f"Step: {k}")
                self.logger.info(f"Run: {k}")
            # get True/False non-nan indices
            if (np.sum(~np.isnan(muks[:, k])) > 0) and (np.size(muks[states, k]) > 0):

                if np.sum(np.isnan(muks[states, k])) > 0:
                    raise Exception("nan value in mu detected!!")

                # calculate actions -> call individual heuristics
                act = np.zeros(states.shape, bool)
                if fun == "oh1":
                    act[states] = self.oh1(muks[states, k])
                elif fun == "oh1_opt":
                    act[states] = self.oh1_opt(muks[states, k], p_thres=par)
                elif fun in ["odoa", "odoa_it", "odoa_optimized"]:
                    act[states] = self.odoa(muks[states, k], ectratio=par)
                elif fun == "odoa_single_core":
                    # uses only a single core to perform the steps
                    # -> whole eval_oh can be wrapped in parallelizer
                    # -> speed up evaluations for small samples
                    act[states] = self.odoa_single_core(muks[states, k], ectratio=par)
                else:
                    raise Exception(
                        f"Blabla Function {fun} not found in list of provided functions!, {type(fun)}, {fun == 'odoa_single_core'}"
                    )

                s2, c2 = self.step(
                    states[states],
                    act[states],
                    costs[states],
                    tfs[states],
                    (k + 1) * self.S.cf["Delta_T"],
                )
                costs[states] = c2
                states[states] = s2

        # evaluate those trajectories, where performed DN at last decision time before failure
        if np.sum(states) != 0:
            raise RuntimeError(
                """Some component still on after last possible replacement time,
                check your implementation!"""
            )

        return self.output_checker(c=costs, t=tfs, o=opt_mode)

    # function that provides a loop to evaluate all c_r ratios with appropriate saving and M calculation
    def eval_mult_c_ratios(
        self,
        c_rs: ndarray = None,
        fun: str = "o_opt_pol",
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
        c_inv_init = self.S.cf["c_inv"]
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
            self.S.cf["c_inv"] = c_r * self.S.cf["c_unav"]

            if fun == "o_opt_pol":
                _, _, Rs[k], var_Rs[k] = self.o_opt_pol(tfs=tfs)

            elif fun == "oh1_opt":
                # sub_ind are random indices to select tf samples used for optimization
                sub_ind = np.arange(np.size(tfs)) if sub_ind is None else sub_ind
                res = spo.minimize_scalar(
                    lambda p: self.eval_oh(
                        fun=fun,
                        tfs=tfs[sub_ind],
                        muks=mu_ln_RUL_preds[sub_ind, :],
                        opt_mode=True,
                        par=p,
                    ),
                    bounds=(0, 1),
                )
                _, _, Rs[k], var_Rs[k] = self.eval_oh(
                    fun=fun,
                    tfs=tfs,
                    muks=mu_ln_RUL_preds,
                    opt_mode=False,
                    par=res.x,
                )
                par_vec.append(res.x)

            elif fun in ["odoa_optimized"]:
                sub_ind = np.arange(np.size(tfs)) if sub_ind is None else sub_ind
                self.logger.info(f"par: {par}, cf: {str(self.S.cf)}")
                res = spo.minimize_scalar(
                    fun=lambda p: self.eval_oh(
                        fun=fun,
                        tfs=tfs[sub_ind],
                        muks=mu_ln_RUL_preds[sub_ind, :],
                        opt_mode=True,
                        par=p,
                    ),
                    method="bounded",
                    bounds=(0, 1),
                    tol=self.S.cf["ectr_oacca"],
                    options={
                        "maxiter": self.S.cf["maxiter"],
                        "disp": True,
                    },
                )
                self.logger.info(f"optimization result: {str(res)}")
                _, _, Rs[k], var_Rs[k] = self.eval_oh(
                    fun=fun,
                    tfs=tfs,
                    muks=mu_ln_RUL_preds,
                    opt_mode=False,
                    par=res.x,
                )
                par_vec.append(res.x)
                if inter_save:
                    self.saver.inter_save_heur(
                        fun, c_r, Rs, var_Rs, None, np.size(tfs), res.x
                    )

            elif fun in ["odoa", "odoa_opt1", "odoa_opt2", "odoa_opt3"]:
                if sub_ind is None:
                    ectr = self.S.oh_ectr(fun=fun, fac=par)
                else:
                    ectr = self.S.oh_ectr_sample_based(
                        fun=fun, tfs=tfs[sub_ind], fac=par
                    )

                _, _, Rs[k], var_Rs[k] = self.eval_oh(
                    fun=fun.split("_")[0],
                    tfs=tfs,
                    muks=mu_ln_RUL_preds,
                    par=ectr,
                )
                par_vec.append(ectr)
                if inter_save:
                    self.saver.inter_save_heur(
                        fun, c_r, Rs, var_Rs, ectr, np.size(tfs), par
                    )

            else:
                _, _, Rs[k], var_Rs[k] = self.eval_oh(
                    fun=fun, tfs=tfs, muks=mu_ln_RUL_preds
                )

        if calc_M:
            opt_perf = mu.load_file(
                filename=os.path.join(
                    self.saver.save_dir, self.saver.save_subdir, "o_opt_pol_1e6"
                )
            )
            Ms, var_Ms = mu.calc_M(R=Rs, R_perfect=opt_perf["R"], var_R=var_Rs)
        else:
            Ms, var_Ms = None, None

        if save:
            if savename is None:
                savename = fun + mu.convert_int_to_estr(np.size(tfs))
            self.saver.final_save_heur(
                savename, np.size(tfs), c_rs, Rs, var_Rs, Ms, var_Ms, par_vec, self.S.cf
            )

        self.S.cf["c_inv"] = c_inv_init
        return Rs, var_Rs, Ms, var_Ms
