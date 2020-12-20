
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", help="Name of keras model in keras_models folder")
parser.add_argument("--frozen", help="Name of frozen model to save in frozen_models folder")
args = parser.parse_args()


if not args.m:
    model_name = 'model.h5'
else:
    model_name = args.m

if not args.frozen:
    frozen_model_name = "model.pb"
else:
    frozen_model_name = args.frozen

model_dir_path = '../keras_models/'
frozen_model_path = os.path.join(os.getcwd(), "../frozen_models/")

model_path = os.path.join(os.getcwd(), model_dir_path+model_name)
model =  tf.keras.models.load_model(model_path)

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir=frozen_model_path,
                        name=frozen_model_name,
                        as_text=False)