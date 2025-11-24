import itertools
import subprocess

n_estimators = [100, 200, 300]
max_depth = [5, 7, 9]

for n, depth in itertools.product(n_estimators, max_depth):
    cmd = [
        "dvc", "exp", "run",
        "-S", f"model.n_estimators={n}",
        "-S", f"model.max_depth={depth}"
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)