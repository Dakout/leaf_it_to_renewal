import os
import logging
import warnings
import numpy as np
from numpy import ndarray
import scipy.integrate as spi

from parallelizer import Parallelizer
from simulator import Simulator


class Agent:
    warnings.filterwarnings(
        "ignore",
        message="Method 'bounded' does not support relative tolerance in x; defaulting to absolute tolerance.",
    )

    def __init__(self, cf: dict) -> None:
        self.logger = self.my_logger()
        self.S = Simulator(cf=cf)
        self.P = Parallelizer()
        return

    def my_logger(self, file_name: str = "default_log_file.log") -> logging.Logger:
        """
        Creates a logger based on a given file name
        """
        logger = logging.getLogger("my_logger")
        logger.setLevel(logging.INFO)

        # Create a file handler
        if not logger.handlers:
            os.makedirs("./logs", exist_ok=True)
            file_handler = logging.FileHandler("./logs/" + file_name)
            file_handler.setLevel(logging.INFO)

            # Create a formatter and set it for the handler
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)

            # Add handler to the logger
            logger.addHandler(file_handler)
        return logger

    def var_R(self, c: ndarray, t: ndarray) -> float:
        """
        Computes the variance of the policy performance.

        Args:
            c: array of costs
            t: array of lifecycle

        Returns:
            Var(E[c]/E[t])
        """
        return (
            1
            / np.size(c)
            * (
                np.var(c) / np.mean(t) ** 2
                + np.mean(c) ** 2 * np.var(t) / np.mean(t) ** 4
                - 2 * np.mean(c) * np.cov(c, t)[0, 1] / np.mean(t) ** 3
            )
        )

    def output_checker(self, c: ndarray, t: ndarray, o: bool = False) -> list:
        """
        Depending on the optimization mode, returns different information.

        Args:
            c: array of costs
            t: array of lifecycle

        Returns either:
            a) [R] or
            b) [c, t, R, Var(R)]
            depending on the opt_mode flag
        """
        try:
            assert not (c == 0).any()
            assert not (t == 0).any()
        except AssertionError:
            print("Indices with 0 costs: ", np.where(c == 0)[0])
            print("Indices with 0 life times: ", np.where(t == 0)[0])

        if o:  # optimization -> return only the mean maintenance cost per unit time
            self.logger.info(f"result: {np.mean(c) / np.mean(t)}")
            return np.mean(c) / np.mean(t)
        else:
            return c, t, np.mean(c) / np.mean(t), self.var_R(c=c, t=t)

    def Pr_smaller_thresh(
        self,
        mu_ln_RUL_pred_k: ndarray,
        ub: float,
        p_thres: float,
    ) -> ndarray:
        """
        Returns a maintenance action depending on whether the estimated
        probability of failure is larger than a given threshold.

        Args:
            mu_ln_RUL_pred_k: mean parameter of ln RUL distribution
            ub: upper integration bound
            p_thres: probability threshold

        Returns:
            a: action vector
        """
        a = np.zeros(np.shape(mu_ln_RUL_pred_k))  # default is do nothing
        # compute probability of failure within next timestep
        pf_pred = self.S.RUL_pred_k_cdf(x=ub, mu_ln_RUL_pred_k=mu_ln_RUL_pred_k)
        a[pf_pred > p_thres] = 1
        return a

    def single_integral(
        self,
        m: float,
        S: Simulator,
        shift: float,
        ilb: float,
        iub: float = np.inf,
        iaccr: float = 1e-5,
    ) -> float:
        """
        Computes a single expeceted value integral
        -> can be wrapped for multiprocessing.

        Args:
            m: mean parameter of ln RUL distribution
            S: Simulator object
            shift: shift in integration variable
            ilb: integration lower bound
            iub: integration upper bound
            iaccr: required integration accuracy

        Returns:
            integration result
        """
        return spi.quad(
            lambda t: (t - shift) * S.RUL_pred_k_pdf(t, m), a=ilb, b=iub, epsrel=iaccr
        )[0]

    def single_core_cond_int(
        self,
        mu_vec: ndarray,
        ilb: float,
        iub: float,
        shift: float,
        iaccr: float,
    ) -> ndarray:
        """
        Performs conditional integration for an array of values on a single cpu
        -> can be wrapped for multiprocessing.

        Args:
            mu_vec: mean parameter of ln RUL distribution
            ilb: integration lower bound
            iub: integration upper bound
            shift: shift in integration variable
            iaccr: required integration accuracy

        Returns:
            integrals: vector of conditional integrals
        """
        import scipy.integrate as spi

        integrals = np.array(
            [
                (
                    spi.quad(
                        lambda t: (t - shift) * self.S.RUL_pred_k_pdf(t, mu_vec[k]),
                        ilb,
                        iub,
                        epsrel=iaccr,
                    )[0]
                    if ~np.isnan(mu_vec[k])
                    else np.nan
                )
                for k in range(len(mu_vec))
            ]
        ).squeeze()
        return integrals
