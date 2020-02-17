# Counterfactual explanations

Code for generating counterfactual explanations is in 'explainer.py'.

Different case studies are presented in jupyter notebooks, but the data is not included.

Below is a brief example of how to generate counterfactual explanations.

```python
# Generate data
import numpy as np

dims = 10
n = 1000
np.random.seed(1)
coefs = np.random.uniform(low=-0.5, high=0.5, size=dims)
X = np.random.uniform(size=(n, dims))
u = np.exp(np.dot(X, coefs))
probs = u / (1 + u)
y = np.random.binomial(1, probs)

# Train model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear')
model.fit(X, y)

# Explain observations
import explainer

default_values = X.mean(axis=0)
decision_boundary = np.percentile(model.predict_proba(X)[:, 1], 95)

def scoring_function(X):
    return model.predict_proba(X)[:, 1]

explain = explainer.Explainer(scoring_function, default_values)
explanations = explain.explain(X, decision_boundary)
# Note observations with default decisions will have empty arrays
explanations
```