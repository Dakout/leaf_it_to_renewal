import os
import dill
import h5py
import numpy as np
from numpy import ndarray


# iteratively converts a dict to h5 format
def save_dict_to_hdf5(group, dictionary):
    for key, item in dictionary.items():
        if isinstance(item, dict):
            # Create a subgroup for nested dictionary
            subgroup = group.create_group(key)
            save_dict_to_hdf5(subgroup, item)
        else:
            # Save dataset for other items
            try:
                group[key] = item
            except:
                pass
    return


# function that checks if folder and file exist and saves accordingly
def save_general_file(
    store_dir: str,
    file_name: str,
    obj_to_store,
    file_type: str = "h5",
    overwrite: bool = False,
) -> None:
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)

    if file_type == "h5":
        if "h5" not in file_name:
            file_name = file_name + ".h5"

    # do not overwrite existing files
    if os.path.isfile(os.path.join(store_dir, file_name)) and not overwrite:
        print(f"File {file_name} exists already!!")
    else:
        if (file_type == "h5") or (".h5" in file_name):
            with h5py.File(os.path.join(store_dir, file_name), "w") as f:
                if isinstance(obj_to_store, ndarray):
                    f.create_dataset("array", data=obj_to_store)
                elif isinstance(obj_to_store, list):
                    array_data = np.array(obj_to_store, dtype=object)
                    f.create_dataset(
                        "list",
                        data=array_data,
                        dtype=h5py.special_dtype(vlen=np.dtype("float64")),
                    )
                elif type(obj_to_store) == dict:
                    save_dict_to_hdf5(f, obj_to_store)
        else:
            with open(file=os.path.join(store_dir, file_name), mode="wb") as f:
                dill.dump(obj_to_store, file=f)
    return


# Function to recursively load dictionary from HDF5
def load_dict_from_hdf5(group):
    result = {}
    # a numpy array is stored as a dict with a single key 'array'
    if ("array" in group.keys()) and (len(group.keys()) == 1):
        return group["array"][:]
    # a list is stored as a dict with a single key 'list'
    if ("list" in group.keys()) and (len(group.keys()) == 1):
        return list(group["list"][:])
    # traverse the stored dict
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = load_dict_from_hdf5(item)
        else:
            result[key] = item[()]
    return result


# function that loads the file specified by the filename path
def load_file(filename: str, file_type: str = "h5") -> dict:
    if (file_type == "h5") or (".h5" in filename):
        if "h5" not in filename:
            filename = filename + ".h5"
        # Load the dictionary from the HDF5 file
        with h5py.File(filename, "r") as f:
            d = load_dict_from_hdf5(f)
    else:
        print("in dill", filename)
        with open(filename, "rb") as f:
            d = dill.load(f)
    return d


# function that converts a list of lists with different lengths into
# a 2d-numpy array with fixed length, but padded with nan
def convert_list_to_nan_padded_array(vec: list) -> np.array:
    max_len = np.max([len(v) for v in vec])
    nan_array = np.full((len(vec), max_len), np.nan)
    for k in range(len(vec)):
        nan_array[k, : len(vec[k])] = vec[k]
    return nan_array


# converts an integer into a string with exponential notation
def convert_int_to_estr(Ns: int, include_underscore: bool = True) -> str:
    return_str = f"{Ns:.0e}".replace("+0", "")
    if include_underscore:
        return_str = "_" + return_str
    return return_str


# calculates the percentage deviation from the perfect policy as well as the variance
def calc_M(R: np.array, R_perfect: np.array, var_R: np.array) -> list:
    M = (R - R_perfect) / R_perfect
    var_M = var_R / R_perfect**2
    return M, var_M


# goes over all files in the given directory and checks for previous iterations (to not overwrite previous results)
def check_current_iteration_number(
    dir: str = "./perf/best_perf", heur_type: str = "rh3", par: str = "opt1"
) -> int:
    files = os.listdir(dir)
    iter = 0
    for f in files:
        if (heur_type in f) and ("iter" in f) and (par in f):
            split_f = f.split("_")
            for s in split_f:
                if "iter" in s:
                    iter = max(iter, int(s.split("iter")[-1]))
    return iter


# saves the current iteration
def save_heur_iter_ectratio(
    obj,
    store_dir: str = "./perf/tmp_iter",
    N_s: str = "1e4",
    heur_type: str = "rh3",
    par: str = "opt1",
) -> None:
    if isinstance(N_s, int):
        N_s = convert_int_to_estr(Ns=N_s)

    iter = check_current_iteration_number(par=par)
    filename = "_".join([heur_type, par, f"iter{iter+1}", N_s])
    save_general_file(store_dir=store_dir, file_name=filename, obj_to_store=obj)
    return


# returns the ectr corresponding to the c_p/c_c ratio from the previous iteration
def previous_heur_ectratio(
    c_r: float, heur_type: str = "rh3", N_s: str = "12e3", par: str = "opt1"
) -> float:
    if isinstance(N_s, int):
        N_s = convert_int_to_estr(Ns=N_s)

    iter = check_current_iteration_number(par=par)

    if iter == 0:
        d = load_file("_".join(["./perf/best_perf/" + heur_type, par, N_s]))
    else:
        d = load_file(
            "_".join(["./perf/best_perf/" + heur_type, par, f"iter{iter}", N_s])
        )
    ind = np.where(abs(d["c_r"] - c_r) < 1e-6)[0]
    return d["R"][ind]


# function that iterates over the interpolated objective function to find the fix-point
def interpolated_iteration(
    start: float, x: np.array, f: np.array, max_iter: int = 20, tol: float = 1e-3
) -> np.array:
    visit_vec = [start]
    for _ in range(max_iter):
        visit_vec.append(np.interp(x=visit_vec[-1], xp=x, fp=f))
        print(
            visit_vec,
            visit_vec[-1] - visit_vec[-2],
            np.abs(visit_vec[-1] - visit_vec[-2]),
            tol,
        )
        if np.abs(visit_vec[-1] - visit_vec[-2]) < tol:
            break
    return np.array(visit_vec)
