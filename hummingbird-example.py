import numpy as np
import xgboost as xgb
from hummingbird.ml import convert, load

# Create some random data for binary classification
num_classes = 2
X = np.load("data/higgs1m_x_train.npy")
y = np.load("data/higgs1m_y_train.npy")

X_test = np.load("data/higgs1m_100x_tile_x_test.npy")

skl_model = xgb.XGBClassifier()
skl_model.fit(X, y)
skl_model.save_model("xgb-model.json")

load_model = xgb.XGBClassifier()
load_model.load_model("xgb-model.json")

# Use Hummingbird to convert the model to PyTorch
model = convert(load_model, 'pytorch', X_test)

# Run predictions on CPU
# model.predict(X_test)

# Run predictions on GPU
#model.to('cuda')
#model.predict(X)

# Save the model
#model.save('hb_model')

# Load the model back
#model = load('hb_model')
