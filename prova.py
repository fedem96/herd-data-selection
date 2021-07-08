import importlib

model_name = "conv3d-2018_08_07-v0"

model_path = "models." + model_name + ".model"
model_module = importlib.import_module(model_path)
model = model_module.get_model()

print(model.summary())
