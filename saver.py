import my_utils as mu
import numpy as np
from numpy import ndarray


class Saver:
    def __init__(self, save_dir: str, save_subdir: str):
        self.save_dir = save_dir
        self.save_subdir = save_subdir
        return

    def inter_save_heur(
        self,
        fun: str,
        c_r: float,
        Rs: ndarray,
        var_Rs: ndarray,
        ectr: float,
        N: int,
        par: float | str,
        cf: dict,
    ) -> None:
        ind = np.where(Rs != 0)[0]
        tmp_file = {
            "c_r": c_r,
            "R": Rs[ind],
            "var_R": var_Rs[ind],
            "ectr": ectr,
            "N": N,
            "par": par,
            "cf": cf,
        }
        if fun in ["rh3_iter"]:
            mu.save_heur_iter_ectratio(
                obj=tmp_file,
                store_dir=f"{self.save_dir}/tmp_{fun}",
                mh_type="rh3",
                N_s=mu.convert_int_to_estr(N, include_underscore=False),
                opt=par + "_" + str(c_r),
            )
        else:
            mu.save_general_file(
                store_dir=f"{self.save_dir}/tmp_{fun}",
                file_name=f"{fun}_{str(c_r)}_{N}",
                obj_to_store=tmp_file,
            )
        return

    def final_save_heur(
        self,
        savename: str,
        N: int,
        c_rs: list | ndarray,
        Rs: ndarray,
        var_Rs: ndarray,
        Ms: ndarray,
        var_Ms: ndarray,
        par_vec: list | ndarray,
    ) -> None:
        file = {
            "c_r": c_rs,
            "R": Rs,
            "var_R": var_Rs,
            "M": Ms,
            "var_M": var_Ms,
            "opt_par": par_vec,
            "N": N,
        }
        mu.save_general_file(
            store_dir=f"{self.save_dir}/{self.save_subdir}",
            file_name=savename,
            obj_to_store=file,
        )
        return

    def inter_save_sub_inds(
        self,
        fun: str,
        c_r: float,
        Rs: ndarray,
        var_Rs: ndarray,
        Ms: ndarray,
        var_Ms: ndarray,
        N_sub_ind: int,
        N: int,
    ) -> None:
        tmp_file = {
            "c_r": c_r,
            "R": Rs,
            "var_R": var_Rs,
            "M": Ms,
            "var_M": var_Ms,
            "N_sub_ind": N_sub_ind,
            "N": N,
        }
        mu.save_general_file(
            store_dir=f"{self.save_dir}/tmp_{fun}",
            file_name=f"{fun}_{str(c_r)}_{N_sub_ind}_{N}",
            obj_to_store=tmp_file,
        )
        return

    def final_save_sub_inds(
        self,
        fun: str,
        c_r: float,
        Rs: ndarray,
        var_Rs: ndarray,
        Ms: ndarray,
        var_Ms: ndarray,
        sub_inds: int,
        N: int,
    ) -> None:
        tmp_file = {
            "c_r": c_r,
            "sub_ind": sub_inds,
            "R": Rs,
            "var_R": var_Rs,
            "M": Ms,
            "var_M": var_Ms,
            "N": N,
        }
        mu.save_general_file(
            store_dir=f"{self.save_dir}/tmp_{fun}",
            file_name=f"{fun}_{str(c_r)}_{N}_sub_inds",
            obj_to_store=tmp_file,
        )
        return
