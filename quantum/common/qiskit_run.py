from json import load, dump
from typing import TypedDict
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.quantum_info import DensityMatrix, Pauli
import numpy as np


def load_job_data(filename: str) -> None:
    with open(filename, encoding="utf8") as f:
        exp = load(f)

    service = QiskitRuntimeService()
    for e in exp:
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


def create_density_matrix(jobs: list[Job]) -> DensityMatrix:
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
                if i % 2:
                    s[0, s_0i] += val
                else:
                    s[0, s_0i] -= val

                if i < 2:
                    s[s_0i, 0] += val
                else:
                    s[s_0i, 0] -= val

    dm = np.zeros((4, 4), dtype=np.complex128)
    p_map = ["I", "X", "Y", "Z"]
    for i in [0, 1, 2, 3]:
        for j in [0, 1, 2, 3]:
            pauli = Pauli("".join([p_map[i] for i in [i, j]])).to_matrix()
            dm += s[i, j] * pauli

    dm /= 4

    return DensityMatrix(dm)

class ParametricExperiment:
    json_path: str
    exp_index: int
    experiments: list

    def __init__(self, json_path: str, t: int, p: int):
        self.json_path = json_path
        with open(json_path, encoding="utf8") as f:
            self.experiments = load(f)
        
        self.exp_index = len(self.experiments)
        self.experiments.append({
            "t": t,
            "p": p,
            "jobs": []
        })

        with open(json_path, "w", encoding="utf8") as f:
            dump(self.experiments, f, indent="\t")
    
    def add_job(self, job: Job):
        self.experiments[self.exp_index]["jobs"].append(job)
        with open(self.json_path, "w", encoding="utf8") as f:
            dump(self.experiments, f, indent="\t")
        
