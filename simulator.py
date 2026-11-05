import numpy as np
from numpy import ndarray
import scipy.integrate as spi
import scipy.optimize as spo
import scipy.stats as sps
from numpy.random import Generator


class Simulator:
    def __init__(self, cf: dict) -> None:
        self.cf = cf
        self.rng = np.random.default_rng(self.cf["rng_seed"])
        self.sigexp = np.exp(0.5 * self.cf["sigma_lnerr"] ** 2)
        return

    def set_new_seed(self, new_seed) -> None:
        if isinstance(new_seed, int):
            self.rng = np.random.default_rng(new_seed)
        elif isinstance(new_seed, Generator):
            self.rng = new_seed
        return

    def draw_tf_samples(self, Ns: int) -> ndarray:
        return sps.norm.rvs(
            loc=self.cf["mu_TF"],
            scale=self.cf["sigma_TF"],
            size=Ns,
            random_state=self.rng,
        )

    def decision_times_given_tf(self, tf_sample: ndarray) -> ndarray:
        return np.arange(self.cf["Delta_T"], tf_sample, self.cf["Delta_T"])

    def corr_matrix_given_tk(self, tk: ndarray) -> ndarray:
        R = np.zeros((len(tk), len(tk)))
        # correlation model
        for i in range(len(tk)):
            R[i, :] = np.exp(-abs(tk[i] - tk) / self.cf["corr_len"])
        return R

    def cov_matrix_given_R(self, R: ndarray) -> ndarray:
        D = np.eye(R.shape[0]) * self.cf["sigma_lnerr"]
        C = D @ R @ D
        return C

    def draw_lnerr_sample_given_C(self, C: ndarray, size: int = 1) -> ndarray:
        ln_e_k = sps.multivariate_normal.rvs(
            mean=np.zeros(C.shape[0]), cov=C, size=size
        )
        return ln_e_k

    def nan_pad(self, mu_list: list, max_len: int) -> list:
        for k in range(len(mu_list)):
            if mu_list[k].size != max_len:
                if mu_list[k].ndim == 1:
                    conc = np.full((max_len - mu_list[k].size,), np.nan)
                else:
                    conc = np.full(
                        (
                            mu_list[k].shape[0],
                            max_len - mu_list[k].shape[1],
                        ),
                        np.nan,
                    )
                mu_list[k] = np.hstack((mu_list[k], conc))
        return mu_list

    def mu_ln_RUL_pred_k_given_tf_vec(self, tf_vec: ndarray, get_npvec: bool = True):
        mu_list = []
        max_len = 0

        # step 1. sort the array
        sorted_idx = np.argsort(tf_vec)
        tf_sorted = tf_vec[sorted_idx]
        reverse_idx = np.argsort(sorted_idx)

        # step 2. find the changes in the number of steps
        steps = np.floor(tf_sorted / 10).astype(int)
        inds = np.unique(steps, return_index=True)[1]
        inds = np.hstack((inds, np.array(tf_vec.size)))

        # step 3. loop over the inds to create the common cov matrices
        for k in range(len(inds) - 1):
            tk = self.decision_times_given_tf(tf_sample=tf_sorted[inds[k]])
            R = self.corr_matrix_given_tk(tk=tk)
            C = self.cov_matrix_given_R(R=R)
            size = inds[k + 1] - inds[k]
            ln_e_k = self.draw_lnerr_sample_given_C(C=C, size=size)
            rul_mat = np.expand_dims(tf_sorted[inds[k] : inds[k + 1]], 1) - tk
            muks = np.log(rul_mat) + ln_e_k
            if muks.shape[1] > max_len:
                max_len = muks.shape[1]
            mu_list.append(muks)

        if get_npvec:
            # pad with nan so that each row has same number of columns
            mu_list = self.nan_pad(mu_list=mu_list, max_len=max_len)
            mu_list = np.vstack(mu_list)
            # reverse initial sorting
            mu_list = mu_list[reverse_idx, :]
        else:
            mu_list = [row for arr in mu_list for row in arr]
            mu_list = [mu_list[inv_ind] for inv_ind in reverse_idx]

        return mu_list

    def RUL_pred_k_pdf(self, x: ndarray, mu_ln_RUL_pred_k: ndarray) -> ndarray:
        return sps.lognorm.pdf(
            x, s=self.cf["sigma_lnerr"], scale=np.exp(mu_ln_RUL_pred_k)
        )

    def RUL_pred_k_cdf(self, x: ndarray, mu_ln_RUL_pred_k: ndarray) -> ndarray:
        return sps.lognorm.cdf(
            x, s=self.cf["sigma_lnerr"], scale=np.exp(mu_ln_RUL_pred_k)
        )

    def RUL_pred_k_icdf(self, q: ndarray, mu_ln_RUL_pred_k: ndarray) -> ndarray:
        return sps.lognorm.ppf(
            q=q, s=self.cf["sigma_lnerr"], scale=np.exp(mu_ln_RUL_pred_k)
        )

    def RUL_pred_k_ci(self, mu_ln_RUL_pred_k: ndarray, alpha: float = 0.99) -> ndarray:
        ci_lower = np.array(
            [
                sps.lognorm.ppf(
                    (1 - alpha) / 2,
                    s=self.cf["sigma_lnerr"],
                    scale=np.exp(mu_ln_RUL_pred_k[m]),
                )
                for m in mu_ln_RUL_pred_k
            ]
        )
        ci_upper = np.array(
            [
                sps.lognorm.ppf(
                    1 - (1 - alpha) / 2,
                    s=self.cf["sigma_lnerr"],
                    scale=np.exp(mu_ln_RUL_pred_k[m]),
                )
                for m in mu_ln_RUL_pred_k
            ]
        )
        return ci_lower, ci_upper

    def RUL_pred_k_mean(self, mu_ln_RUL_pred_k: ndarray) -> ndarray:
        return sps.lognorm.mean(
            s=self.cf["sigma_lnerr"], scale=np.exp(mu_ln_RUL_pred_k)
        )

    def RUL_pred_k_median(self, mu_ln_RUL_pred_k: ndarray) -> ndarray:
        return sps.lognorm.median(
            s=self.cf["sigma_lnerr"], scale=np.exp(mu_ln_RUL_pred_k)
        )

    def RUL_pred_k_var(self, mu_ln_RUL_pred_k: ndarray) -> ndarray:
        return sps.lognorm.var(s=self.cf["sigma_lnerr"], scale=np.exp(mu_ln_RUL_pred_k))

    def RUL_pred_k_std(self, mu_ln_RUL_pred_k: ndarray) -> ndarray:
        return sps.lognorm.std(s=self.cf["sigma_lnerr"], scale=np.exp(mu_ln_RUL_pred_k))

    def RUL_pred_k_ci2(self, mu_ln_RUL_pred_k: ndarray, alpha: float = 0.99) -> ndarray:
        return sps.lognorm.interval(
            confidence=alpha, s=self.cf["sigma_lnerr"], scale=np.exp(mu_ln_RUL_pred_k)
        )

    def RUL_pred_k_rvs(self, mu_ln_RUL_pred_k: ndarray, size: int = 1) -> ndarray:
        return sps.lognorm.rvs(
            s=self.cf["sigma_lnerr"], scale=np.exp(mu_ln_RUL_pred_k), size=size
        )

    def cond_rul_smaller(self, m: ndarray, iub: float) -> ndarray:
        frac = (np.log(iub) - m) / self.cf["sigma_lnerr"]
        return self.sigexp * np.exp(m) * sps.norm.cdf(x=frac - self.cf["sigma_lnerr"])

    def cond_rul_betw(self, m: ndarray, ilb: float, iub: float) -> ndarray:
        frac_lb = (np.log(ilb) - m) / self.cf["sigma_lnerr"]
        frac_ub = (np.log(iub) - m) / self.cf["sigma_lnerr"]
        return (
            self.sigexp
            * np.exp(m)
            * (
                sps.norm.cdf(x=frac_ub - self.cf["sigma_lnerr"])
                - sps.norm.cdf(x=frac_lb - self.cf["sigma_lnerr"])
            )
        )

    ##################################################################################
    ################ Functions for ECTR initialization ###############################
    ##################################################################################

    # for ectr options 1 (ordering & maintenance), initializiation is performed
    # with respect to the optimal repair / ordering time; even though this would
    # not be available to the operator (only k*DT)
    #   -> maybe change this later

    ############ Maintenance functions based on underlying TF-distribution ###########

    def r_ectr_wrt_tf_given_trep(self, t_rep: float) -> float:
        Pf = sps.norm.cdf(t_rep, loc=self.cf["mu_TF"], scale=self.cf["sigma_TF"])
        mu_cm = self.cf["c_c"] * Pf + self.cf["c_p"] * (1 - Pf)
        mu_tlc = spi.quad(
            lambda t: t
            * sps.norm.pdf(t, loc=self.cf["mu_TF"], scale=self.cf["sigma_TF"]),
            a=self.cf["mu_TF"] - self.cf["sig_int_factor"] * self.cf["sigma_TF"],
            b=t_rep,
        )[
            0
        ] + t_rep * (  # * Pf
            1 - Pf
        )
        return mu_cm / mu_tlc

    def r_ectr_init_opt1(self, return_res: bool = False) -> float:
        res = spo.minimize_scalar(
            fun=self.r_ectr_wrt_tf_given_trep,
            method="bounded",
            bounds=(
                0,
                self.cf["mu_TF"]
                + self.cf[
                    "sigma_TF"
                ],  # self.cf["sig_int_factor"] * self.cf["sigma_TF"],
            ),
        )
        if return_res:
            return res
        else:
            return res.fun

    def r_ectr_init_opt2(self) -> float:
        return self.cf["c_p"] / self.cf["mu_TF"]

    def r_ectr_init_opt3(self, fac: float = 0.5) -> float:
        return fac * self.r_ectr_init_opt1() + (1 - fac) * self.r_ectr_init_opt2()

    def rh_ectr(self, fun: str, fac: float = 0.5, sub_ind=None) -> float:
        f_dict = {
            "opt1": self.r_ectr_init_opt1(),
            "opt2": self.r_ectr_init_opt2(),
        }

        # check if there is an appendix separated by
        f_split = fun.split("_")
        if len(f_split) > 1:
            # extract the correct option based on the appendix
            opt = f_split[-1]
            if opt == "opt3":
                fac = 0.5 if fac is None else fac
                return self.r_ectr_init_opt3(fac=fac)
            else:
                return f_dict[opt]
        else:
            return self.r_ectr_init_opt1()

    ######### Maintenance functions based on a subset of tf-samples ######################

    def r_ectr_wrt_tf_given_trep_sample_based(
        self, t_rep: float, tfs: ndarray
    ) -> float:
        lc_vec = np.zeros_like(tfs)
        cost_vec = np.zeros_like(tfs)
        fail_ind = tfs < t_rep

        lc_vec[fail_ind] = tfs[fail_ind]
        lc_vec[~fail_ind] = t_rep

        cost_vec[fail_ind] = self.cf["c_c"]
        cost_vec[~fail_ind] = self.cf["c_p"]
        return np.mean(cost_vec) / np.mean(lc_vec)

    def r_ectr_init_opt1_sample_based(
        self, tfs: ndarray, return_res: bool = False
    ) -> float:
        res = spo.minimize_scalar(
            fun=self.r_ectr_wrt_tf_given_trep_sample_based,
            method="bounded",
            bounds=(
                0,
                self.cf["mu_TF"] + self.cf["sigma_TF"],
            ),
            args=(tfs),
        )
        if return_res:
            return res
        else:
            return res.fun

    def r_ectr_init_opt2_sample_based(self, tfs: ndarray) -> float:
        return self.cf["c_p"] / np.mean(tfs)

    def r_ectr_init_opt3_sampled_based(self, tfs: ndarray, fac: float = 0.5) -> float:
        res = fac * self.r_ectr_init_opt1_sample_based(tfs=tfs) + (
            1 - fac
        ) * self.r_ectr_init_opt2_sample_based(tfs=tfs)
        return res

    def rh_ectr_sample_based(self, fun: str, tfs: ndarray, fac: float = 0.5) -> float:
        f_dict = {
            "opt1": self.r_ectr_init_opt1_sample_based(tfs=tfs),
            "opt2": self.r_ectr_init_opt2_sample_based(tfs=tfs),
        }

        # check if there is an appendix separated by
        f_split = fun.split("_")
        if len(f_split) > 1:
            # extract the correct option based on the appendix
            opt = f_split[-1]
            if opt == "opt3":
                fac = 0.5 if fac is None else fac
                return self.r_ectr_init_opt3_sampled_based(tfs=tfs, fac=fac)
            else:
                return f_dict[opt]
        else:
            return self.r_ectr_init_opt1_sample_based(tfs=tfs)

    ########## Ordering functions based on underlying TF-distribution ##############

    def o_ectr_wrt_tf_given_tord(self, t_ord: float) -> float:
        P_f = sps.norm.cdf(
            t_ord + self.cf["L"], loc=self.cf["mu_TF"], scale=self.cf["sigma_TF"]
        )
        cond_rul_less = spi.quad(
            lambda t: t
            * sps.norm.pdf(t, loc=self.cf["mu_TF"], scale=self.cf["sigma_TF"]),
            a=0,
            b=t_ord + self.cf["L"],
        )[0]
        cond_rul_more = spi.quad(
            lambda t: t
            * sps.norm.pdf(t, loc=self.cf["mu_TF"], scale=self.cf["sigma_TF"]),
            a=t_ord + self.cf["L"],
            b=1000,
        )[0]
        mu_cm_fail = self.cf["c_unav"] * (P_f * (t_ord + self.cf["L"]) - cond_rul_less)
        mu_cm_surv = self.cf["c_inv"] * (
            cond_rul_more - (1 - P_f) * (t_ord + self.cf["L"])
        )
        mu_cm = mu_cm_fail + mu_cm_surv
        mu_tlc = self.cf["mu_TF"]
        return mu_cm / mu_tlc

    def o_ectr_init_opt1(self, return_res: bool = False) -> float:
        res = spo.minimize_scalar(
            fun=self.o_ectr_wrt_tf_given_tord,
            method="bounded",
            bounds=(
                0,
                self.cf["mu_TF"] + self.cf["sig_int_factor"] * self.cf["sigma_TF"],
            ),
        )
        if return_res:
            return res
        else:
            return res.fun

    def o_ectr_init_opt2(self) -> float:
        mu_c = (
            0.5
            * self.cf["c_inv"]
            * self.cf["c_unav"]
            * self.cf["Delta_T"]
            / (self.cf["c_inv"] + self.cf["c_unav"])
        )
        return mu_c / self.cf["mu_TF"]

    def o_ectr_init_opt3(self, fac: float = 0.5) -> float:
        return fac * self.o_ectr_init_opt1() + (1 - fac) * self.o_ectr_init_opt2()

    def oh_ectr(self, fun: str, fac: float = 0.5) -> float:
        f_dict = {
            "opt1": self.o_ectr_init_opt1(),
            "opt2": self.o_ectr_init_opt2(),
        }

        # check if there is an appendix separated by
        f_split = fun.split("_")
        if len(f_split) > 1:
            # extract the correct option based on the appendix
            opt = f_split[-1]
            if opt == "opt3":
                fac = 0.5 if fac is None else fac
                return self.o_ectr_init_opt3(fac=fac)
            else:
                return f_dict[opt]
        else:
            return self.o_ectr_init_opt1()

    ####### Ordering functions based on a subset of tf-samples ################

    # something is not right here -> long-running maintenance cost per unit time not included here

    def o_ectr_wrt_tf_given_tord_sample_based(
        self, t_ord: float, tfs: ndarray
    ) -> float:
        cost_vec = np.zeros_like(tfs)

        fail_ind = tfs < t_ord + self.cf["L"]

        cost_vec[fail_ind] = (t_ord + self.cf["L"] - tfs[fail_ind]) * self.cf["c_unav"]
        cost_vec[~fail_ind] = (tfs[~fail_ind] - t_ord - self.cf["L"]) * self.cf["c_inv"]
        return np.mean(cost_vec) / np.mean(tfs)

    def o_ectr_init_opt1_sample_based(
        self, tfs: ndarray, return_res: bool = False
    ) -> float:
        res = spo.minimize_scalar(
            fun=self.o_ectr_wrt_tf_given_tord_sample_based,
            method="bounded",
            bounds=(
                0,
                self.cf["mu_TF"] + self.cf["sigma_TF"],
            ),
            args=(tfs),
        )
        if return_res:
            return res
        else:
            return res.fun

    def o_ectr_init_opt2_sample_based(self, tfs: ndarray) -> float:
        cost_vec = np.zeros_like(tfs)

        rem = tfs - (tfs // self.cf["Delta_T"]) * self.cf["Delta_T"]

        ind = (
            rem
            < (self.cf["c_unav"] / (self.cf["c_inv"] + self.cf["c_unav"]))
            * self.cf["Delta_T"]
        )

        cost_vec[ind] = rem[ind] * self.cf["c_inv"]
        cost_vec[~ind] = (self.cf["Delta_T"] - rem[~ind]) * self.cf["c_unav"]
        return np.mean(cost_vec) / np.mean(tfs)

    def o_ectr_init_opt3_sampled_based(self, tfs: ndarray, fac: float = 0.5) -> float:
        res = fac * self.o_ectr_init_opt1_sample_based(tfs=tfs) + (
            1 - fac
        ) * self.o_ectr_init_opt2_sample_based(tfs=tfs)
        return res

    def oh_ectr_sample_based(self, fun: str, tfs: ndarray, fac: float = 0.5) -> float:
        f_dict = {
            "opt1": self.o_ectr_init_opt1_sample_based(tfs=tfs),
            "opt2": self.o_ectr_init_opt2_sample_based(tfs=tfs),
        }

        # check if there is an appendix separated by
        f_split = fun.split("_")
        if len(f_split) > 1:
            # extract the correct option based on the appendix
            opt = f_split[-1]
            if opt == "opt3":
                fac = 0.5 if fac is None else fac
                return self.o_ectr_init_opt3_sampled_based(tfs=tfs, fac=fac)
            else:
                return f_dict[opt]
        else:
            return self.o_ectr_init_opt1_sample_based(tfs=tfs)
