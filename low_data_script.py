import argparse
import copy
import numpy as np
from numpy import ndarray
import pathos.multiprocessing as pmp
import matplotlib.pyplot as plt  # noqa: F401

from config import config
import my_utils as mu
import order_agent
from order_agent import OrderAgent as OA
from pathos.multiprocessing import ProcessPool as Pool
import replacement_agent
from replacement_agent import ReplacementAgent as RA
from tqdm import tqdm


def r_eval_mult(
    Agent: RA | OA, fun: str, tf: ndarray, mu: ndarray, par: ndarray
) -> ndarray:
    res = np.array(
        [
            (
                Agent.eval_rh(
                    fun=fun,
                    tfs=tf,
                    muks=mu,
                    opt_mode=True,
                    par=p,
                )
                if ~np.isnan(p)
                else np.nan
            )
            for p in par
        ]
    ).squeeze()
    return res


def o_eval_mult(
    Agent: RA | OA, fun: str, tf: ndarray, mu: ndarray, par: ndarray
) -> ndarray:
    res = np.array(
        [
            (
                Agent.eval_oh(
                    fun=fun,
                    tfs=tf,
                    muks=mu,
                    opt_mode=True,
                    par=p,
                )
                if ~np.isnan(p)
                else np.nan
            )
            for p in par
        ]
    ).squeeze()
    return res


def main(
    fun: str = "rh1_opt",
    Agent: RA | OA | None = None,
    N_trials: int = 20,
    N_p_points: int = 10_000,
) -> None:
    if isinstance(Agent, RA) or type(Agent) == replacement_agent.ReplacementAgent:
        eval_fun = Agent.eval_rh
        mo = mu.load_file("./perf/rep_perf/best_perf/r_opt_pol_1e6.h5")
        eval_mult = r_eval_mult
    elif isinstance(Agent, OA) or type(Agent) == order_agent.OrderAgent:
        eval_fun = Agent.eval_oh
        mo = mu.load_file("./perf/order_perf/L11/o_opt_pol_1e6.h5")
        eval_mult = o_eval_mult

    tf = mu.load_file("./data/tf_samples_1e6")
    muks = mu.load_file("./data/mu_ln_RUL_pred_1e6_nv")

    N = int(1e3)
    N_sub_inds = [10, 100, 1000]
    crs = np.array([0.1])
    opt_ind = np.where(mo["c_r"] == np.array([0.1]))
    R_perfect = mo["R"][opt_ind]
    R_vec = []
    var_R_vec = []
    M_vec = []
    var_M_vec = []

    for i, N_sub_ind in enumerate(N_sub_inds):
        R_trial = []
        var_R_trial = []
        for j in tqdm(range(N_trials)):
            sub_ind = np.random.choice(range(0, N), size=N_sub_ind, replace=False)

            if ("h1_opt") in fun or ("single_core") in fun:
                par_vec = np.linspace(1e-6, 2 - 1e-6, N_p_points)
                n_cpus = pmp.cpu_count()
                pool = Pool(n_cpus)
                par_reformed = Agent.P.reform_x(x=par_vec, n_cpus=n_cpus)
                try:
                    result = pool.amap(
                        eval_mult,
                        [Agent] * n_cpus,
                        [fun] * n_cpus,
                        [tf[sub_ind]] * n_cpus,
                        [muks[sub_ind, :]] * n_cpus,
                        par_reformed,
                    )
                except ValueError:
                    print("Restarting pool")
                    pool.restart()
                    print("Restarting done")
                    result = pool.amap(
                        eval_mult,
                        [Agent] * n_cpus,
                        [fun] * n_cpus,
                        [tf[sub_ind]] * n_cpus,
                        [muks[sub_ind, :]] * n_cpus,
                        par_reformed,
                    )
                if result._success:
                    r = result.get()
                    r_vec = np.array(r).flatten()
                    r_vec = r_vec[~np.isnan(r_vec)]
                else:
                    print("\nParallel run was not succesfull!")

                try:
                    assert np.size(r_vec) == np.size(par_vec)
                except:
                    raise Exception("r_vec and par_vec have different sizes!")

                # can uncomment to check visually for results
                # plt.figure()
                # plt.plot(par_vec, r_vec)
                # plt.show()

                # compute indices which achieved best performance
                min_indices = np.where(np.abs(r_vec - np.min(r_vec)) < 1e-6)[0]
                # randomly sample an index
                min_index = min_indices[
                    np.random.choice(range(0, len(min_indices)), size=1)
                ]
                # calculate performance on whole population based on chosen value
                if fun == "odoa_single_core":
                    efun = "odoa"
                elif fun == "rdoa_single_core":
                    efun = "rdoa"
                else:
                    efun = fun

                _, _, R, var_R = eval_fun(
                    fun=efun,
                    tfs=tf[:N],
                    muks=muks[:N, :],
                    opt_mode=False,
                    par=par_vec[min_index],
                )

            elif "doa" in fun:
                R, var_R, _, _ = Agent.eval_mult_c_ratios(
                    c_rs=crs,
                    tfs=tf[:N],
                    mu_ln_RUL_preds=muks[:N, :],
                    fun=fun,
                    inter_save=False,
                    par=None,
                    calc_M=False,
                    save=False,
                    sub_ind=sub_ind,
                )

            R_trial.append(R)
            var_R_trial.append(var_R)
            print(R)

            # intermediate saving in case it takes long takes long
            tmp = {
                "R": R,
                "var_R": var_R,
                "N_sub_ind": N_sub_ind,
                "N": N,
            }
            mu.save_general_file(
                store_dir="./perf", file_name=f"{fun}_{N_sub_ind}_{j}", obj_to_store=tmp
            )

        M_trial, var_M_trial = mu.calc_M(
            R=R_trial, R_perfect=R_perfect, var_R=var_R_trial
        )
        R_vec.append(np.array(R_trial))
        var_R_vec.append(np.array(var_R_trial))
        M_vec.append(M_trial)
        var_M_vec.append(var_M_trial)

    return [R_vec, var_R_vec, M_vec, var_M_vec]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fun", type=str, default="rh1_opt")
    parser.add_argument("--agent", type=str, default="A")
    parser.add_argument("--c_r", type=float, default=0.1)
    parser.add_argument("--N_trials", type=int, default=20)
    parser.add_argument("--N_p_points", type=int, default=10_000)
    parser.add_argument("--maxiter", type=int, default=12)
    args = parser.parse_args()
    print(args)

    if args.agent == "A":
        Agent = RA(copy.deepcopy(config))
        if abs(args.c_r - config["c_p"] / config["c_c"]) > 1e-6:
            Agent.S.cf["c_p"] = args.c_r * Agent.S.cf["c_c"]

    elif args.agent == "O":
        Agent = OA(copy.deepcopy(config))
        if abs(args.c_r - config["c_inv"] / config["c_unav"]) > 1e-6:
            Agent.S.cf["c_inv"] = args.c_r * Agent.S.cf["c_unav"]

    if args.maxiter != config["maxiter"]:
        Agent.S.cf["maxiter"] = args.maxiter

    print(f"Starting {type(Agent)} evaluation for function {args.fun}")
    main(fun=args.fun, Agent=Agent, N_trials=args.N_trials, N_p_points=args.N_p_points)
