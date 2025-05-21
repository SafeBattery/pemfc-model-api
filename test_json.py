import json
import numpy as np

# PWU: (600, 9)
pwu_input = {
    "type": "PWU",
    "threshold": 0.02,
    "input": np.random.rand(600, 9).round(3).tolist()
}
with open("pwu_input.json", "w") as f:
    json.dump(pwu_input, f, indent=2)

# T3: (600, 4)
t3_input = {
    "type": "T3",
    "threshold": 0.02,
    "input": np.random.rand(600, 4).round(3).tolist()
}
with open("t3_input.json", "w") as f:
    json.dump(t3_input, f, indent=2)
