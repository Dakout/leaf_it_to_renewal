import warnings
import numpy as np
from numpy import ndarray
import pathos.multiprocessing as pmp
from pathos.multiprocessing import ProcessPool as Pool

warnings.filterwarnings(
    "ignore",
    message="Method 'bounded' does not support relative tolerance in x; defaulting to absolute tolerance.",
)


class Parallelizer:
    def __init__(self) -> None:
        return

    def multi_optimization_rh2(
        self,
        A: object,
        mu_vec: ndarray,
        ectr: float,
        iaccr: float,
        olb: float,
        oub: float,
        oacca: float,
    ) -> ndarray:
        """
        Parallel implementation for the rh2 optimization.

        Args:
            A: ReplacementAgent object
            mu_vec: mean of ln RUL distribution
            ectr: assumed renewal reward process cost rate
            iaccr: required integration accuracy
            olb: optimization lower bound
            oub: optimization upper boound
            oacca: required optimization accuracy

        Returns:
            res: performed optimization for all non-nan elements in mu_vec
        """
        import scipy.optimize as spo

        res = np.array(
            [
                (
                    spo.minimize_scalar(
                        lambda T: A.single_f_Trk_rh2(
                            m=[mu_vec[k]], S=A.S, ectr=ectr, iaccr=iaccr, Trk=T
                        ),
                        bounds=(olb, oub),
                        method="bounded",
                        tol=oacca,
                    ).x.item()
                    if ~np.isnan(mu_vec[k])
                    else np.nan
                )
                for k in range(len(mu_vec))
            ]
        ).squeeze()
        return res

    def reform_x(self, x: ndarray, n_cpus: int) -> ndarray:
        if np.size(x) % n_cpus != 0:
            x_padded = np.concatenate(
                (x, np.full(shape=n_cpus - np.size(x) % n_cpus, fill_value=np.nan))
            )
        else:
            x_padded = x
        chunk = int(np.floor(np.size(x_padded) / n_cpus))
        x_reformed = list(x_padded.reshape((n_cpus, chunk)))
        return x_reformed

    def parallel_runner(
        self,
        A: object,
        f: str,
        m: ndarray,
        ectr: float = None,
        iaccr: float = None,
        olb: float = None,
        oub: float = None,
        oacca: float = None,
    ) -> ndarray:
        """
        Main parallel runner function. Creates a pool with the available cpus
        distributes the calculations for the passed function respectively.

        Args:
            A: either ReplacementAgent or OrderAgent instance
            f: function string
            m: mean of ln RUL predictions
            ectr: assumed renewal reward process cost rate
            ilb: integration lower bound
            iub: integration upper bound
            shift: shift in integration variable
            iaccr: required integration accuracy
            olb: optimization lower bound
            oub: optimization upper boound
            oacca: required optimization accuracy

        Returns:
            res_vec: result for all elements in m
        """

        n_cpus = pmp.cpu_count()
        # print(f"\nParallel computing with {n_cpus} cpus\n")
        pool = Pool(n_cpus)

        # distribute the jobs equally across all cpus
        m_reformed = self.reform_x(x=m, n_cpus=n_cpus)
        try:
            if f == "multi_optimization_rh2":
                result = pool.amap(
                    self.multi_optimization_rh2,
                    [A] * n_cpus,
                    m_reformed,
                    [ectr] * n_cpus,
                    [iaccr] * n_cpus,
                    [olb] * n_cpus,
                    [oub] * n_cpus,
                    [oacca] * n_cpus,
                )

        except ValueError:
            print("Restarting pool")
            pool.restart()
            print("Restarting done")
            if f == "multi_optimization_rh2":
                result = pool.amap(
                    self.multi_optimization_rh2,
                    [A] * n_cpus,
                    m_reformed,
                    [ectr] * n_cpus,  # [ilb]*n_cpus, [iub]*n_cpus,
                    [iaccr] * n_cpus,
                    [olb] * n_cpus,
                    [oub] * n_cpus,
                    [oacca] * n_cpus,
                )

        if result._success:
            # print(result, result.__dict__)
            r = result.get()
            res_vec = np.array(r).flatten()
            res_vec = res_vec[~np.isnan(res_vec)]
        else:
            print("\nParallel run was not succesfull!")

        try:
            assert np.size(res_vec) == np.size(m)
        except:
            raise Exception("result size does not match input size!")

        pool.close()
        pool.join()
        pool.clear()
        return res_vec
