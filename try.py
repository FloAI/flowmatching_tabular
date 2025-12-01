from flow_bdt import FlowMatchingBDT   # or wherever you saved it
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Training data
X = np.random.randn(300, 4)
conditions = np.random.randint(0, 3, size=(300, 1))

# Very lightweight regressor
reg = DecisionTreeRegressor(max_depth=5)

model = FlowMatchingBDT(
    base_regressor=reg,
    n_flow_steps=20,      # also reduces memory
    n_duplicates=20       # also reduces memory
)

model.fit(X, conditions)

generated = model.predict(50, conditions=np.zeros((50, 1)))

print(generated.shape)

