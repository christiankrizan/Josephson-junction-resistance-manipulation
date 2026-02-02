import numpy as np
import pandas as pd

## Input data tables here. ##

# Anharmonicity values.
eta = {
    "S1": {
        "Q1": {"C1": None,     "C2": -214.4e6},
        "Q2": {"C1": -210.0e6,   "C2": None},
        "Q3": {"C1": None,     "C2": -222.2e6},
        "Q4": {"C1": -210.3e6,   "C2": None},
        "Q5": {"C1": -202.3e6,   "C2": None},
        "Q6": {"C1": -204.8e6,   "C2": -197.6e6},
        "Q7": {"C1": -201.7e6,   "C2": -207.2e6},
        "Q8": {"C1": -203.2e6,   "C2": -205.3e6},
    },
    "S2": {
        "Q1": {"C1": -212.7e6,   "C2": None},
        "Q2": {"C1": None,     "C2": None},
        "Q3": {"C1": -216.8e6,   "C2": -222.0e6},
        "Q4": {"C1": None,     "C2": None},
        "Q5": {"C1": -207.4e6,   "C2": None},
        "Q6": {"C1": None,     "C2": None},
        "Q7": {"C1": -208.9e6,   "C2": None},
        "Q8": {"C1": None,     "C2": None},
    }
}

# Dispersive shift
twochi = {
    "S1": {
        "Q1": {"C1": -3108e3, "C2": -1030e3},
        "Q2": {"C1": -1764e3, "C2": None},
        "Q3": {"C1": -1024e3, "C2": -411e3},
        "Q4": {"C1": -1435e3, "C2": None},
        "Q5": {"C1": -2887e3, "C2": None},
        "Q6": {"C1": -1671e3, "C2": -899e3},
        "Q7": {"C1": -1434e3, "C2": -965e3},
        "Q8": {"C1": -1627e3, "C2": -1159e3},
    },
    "S2": {
        "Q1": {"C1": -1495e3, "C2": None},
        "Q2": {"C1": None,   "C2": None},
        "Q3": {"C1": -778e3,  "C2": -620e3},
        "Q4": {"C1": None,   "C2": None},
        "Q5": {"C1": -1495e3, "C2": -2054e3},
        "Q6": {"C1": None,   "C2": None},
        "Q7": {"C1": -857e3,  "C2": -574e3},
        "Q8": {"C1": None,   "C2": None},
    }
}

# Detuning
Delta = {
    "S1": {
        "Q1": {"C1": -1102e6, "C2": -1401e6},
        "Q2": {"C1": -1081e6, "C2": None},
        "Q3": {"C1": -2026e6, "C2": -2108e6},
        "Q4": {"C1": -1341e6, "C2": None},
        "Q5": {"C1": -884e6,  "C2": None},
        "Q6": {"C1": -1175e6, "C2": -1611e6},
        "Q7": {"C1": -1350e6, "C2": -1621e6},
        "Q8": {"C1": -1307e6, "C2": -1606e6},
    },
    "S2": {
        "Q1": {"C1": -1175e6, "C2": None},
        "Q2": {"C1": None,   "C2": None},
        "Q3": {"C1": -1756e6, "C2": -1955e6},
        "Q4": {"C1": None,   "C2": None},
        "Q5": {"C1": -1292e6, "C2": -1595.53e6},#-1596e6},
        "Q6": {"C1": None,   "C2": None},
        "Q7": {"C1": -1793e6, "C2": -1938e6},
        "Q8": {"C1": None,   "C2": None},
    }
}

## Function to compute g ##
from time import sleep

def compute_g(Delta, twochi, eta):
    g = np.sqrt((Delta * twochi/2) * ((eta + Delta) / eta))
    print("Δ: "+str(Delta)+" Hz")
    print("χ: "+str(twochi/2)+" Hz")
    print("g: "+str(g)+" Hz")
    sleep(0.3)
    return g

## Compute g for every qubit ##

results = []

for sample in ["S1", "S2"]:
    for qubit in eta[sample].keys():

        for cooldown in ["C1", "C2"]:

            e = eta[sample][qubit][cooldown]
            try:
                c = twochi[sample][qubit][cooldown] / 2
            except TypeError:
                c = twochi[sample][qubit][cooldown] # The value was None, skip the divide-by-2.
            d = Delta[sample][qubit][cooldown]

            # Skip if χ or Δ is missing.
            if c is None or d is None:
                g_value = None
            else:
                # If η is missing in this cooldown, use the other cooldown's η
                if e is None:
                    other = "C2" if cooldown == "C1" else "C1"
                    e = eta[sample][qubit][other]

                # Still missing? Then, computer says no.
                if e is None:
                    g_value = None
                else:
                    g_value = compute_g(d, c, e)

            results.append({
                "Sample": sample,
                "Qubit": qubit,
                "Cooldown": cooldown,
                "g": g_value
            })

df = pd.DataFrame(results)

# Pivot, to see C1 vs C2 side by side
comparison = df.pivot_table(index=["Sample", "Qubit"], columns="Cooldown", values="g")

# Compute change: percentage and absolute
comparison["Δg_abs"] = comparison["C2"] - comparison["C1"]
comparison["Δg_rel"] = (comparison["Δg_abs"] / comparison["C1"]) * 100

pd.set_option("display.max_rows", None)
print(comparison)
