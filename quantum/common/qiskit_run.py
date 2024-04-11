from json import dump, load
from typing import TypedDict

import numpy as np
from qiskit.quantum_info import DensityMatrix, Pauli
from qiskit_ibm_runtime import QiskitRuntimeService


def load_job_data(filename: str) -> None:
    with open(filename, encoding="utf8") as f:
        exp = load(f)

    service = QiskitRuntimeService()
    for e in exp:
        if "id" in e and len(e["jobs"]) == 0:
            ibm_job = service.job(e["id"])
            for i in [1, 2, 3]:
                for j in [1, 2, 3]:
                    e["jobs"].append(
                        {
                            "s": [i, j],
                            "result": ibm_job.result().quasi_dists[
                                (i - 1) * 3 + (j - 1)
                            ],
                        }
                    )

            with open(filename, "w", encoding="utf8") as f:
                dump(exp, f, indent="\t")

            continue

        for j in e["jobs"]:
            if "result" in j:
                continue
            ibm_job = service.job(j["id"])
            res = ibm_job.result().quasi_dists[0]
            j["result"] = res

            with open(filename, "w", encoding="utf8") as f:
                dump(exp, f, indent="\t")


class Job(TypedDict):
    s: [int, int]
    id: str
    result: dict[str, float]


def resolve_stokes(jobs: list[Job]) -> np.array:
    s = np.zeros((4, 4))
    s[0, 0] = 1

    for j in jobs:
        res = j["result"]
        res = {int(k): v for k, v in res.items()}
        for i in range(0, 4):
            val = res[i] if i in res else 0
            if i in (0, 3):
                s[*j["s"]] += val
            else:
                s[*j["s"]] -= val

            # S(0,i) and S(i, 0) values
            s_0i = j["s"][0]
            if j["s"][0] == j["s"][1]:
                if i < 2:
                    s[0, s_0i] += val
                else:
                    s[0, s_0i] -= val

                if i % 2 == 0:
                    s[s_0i, 0] += val
                else:
                    s[s_0i, 0] -= val

    return s


def create_density_matrix(s: np.array) -> np.array:
    dm = np.zeros((4, 4), dtype=np.complex128)
    p_map = ["I", "X", "Y", "Z"]
    for i in [0, 1, 2, 3]:
        for j in [0, 1, 2, 3]:
            pauli = Pauli("".join([p_map[i] for i in [i, j]])).to_matrix()
            dm += s[i, j] * pauli

    dm /= 4

    return dm


def dm_optimize(mu: np.array) -> np.array:
    d = mu.shape[0]
    (mu_val, mu_vec) = np.linalg.eig(mu.data)

    arg = np.flip(np.argsort(mu_val))
    mu_vec = np.transpose(mu_vec)[arg]
    mu_val = mu_val[arg]

    lam = np.zeros(d, dtype=np.complex128)
    i = d - 1
    a = 0
    while mu_val[i] + a / (i + 1) <= 0:
        lam[i] = 0
        a += mu_val[i]
        i -= 1

    for j in range(0, i + 1):
        lam[j] = mu_val[j] + a / (i + 1)

    rho = np.zeros((d, d), dtype=np.complex128)
    for i in range(0, d):
        rho += lam[i] * np.outer(mu_vec[i], mu_vec[i].conj())

    return rho


def jobs_to_dm(jobs: list[Job]) -> DensityMatrix:
    sk = resolve_stokes(jobs)
    dm = create_density_matrix(sk)
    odm = dm_optimize(dm)
    return DensityMatrix(odm)


class ParametricExperiment:
    json_path: str
    exp_index: int
    experiments: list

    def __init__(self, json_path: str, t: int, p: int, job_id: str = None):
        self.json_path = json_path
        with open(json_path, encoding="utf8") as f:
            self.experiments = load(f)

        self.exp_index = len(self.experiments)
        if job_id is not None:
            self.experiments.append({"t": t, "p": p, "id": job_id, "jobs": []})
        else:
            self.experiments.append({"t": t, "p": p, "jobs": []})

        with open(json_path, "w", encoding="utf8") as f:
            dump(self.experiments, f, indent="\t")

    def add_job(self, job: Job):
        self.experiments[self.exp_index]["jobs"].append(job)
        with open(self.json_path, "w", encoding="utf8") as f:
            dump(self.experiments, f, indent="\t")
