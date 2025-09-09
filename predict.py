import json
import sys
import numpy as np
import tensorflow as tf

model_path = sys.argv[1]
input_file = sys.argv[2]

arr = np.load(input_file)
model = tf.keras.models.load_model(model_path)

pred = model.predict(arr)
safe_pred = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in pred.items()}
print(json.dumps(safe_pred))