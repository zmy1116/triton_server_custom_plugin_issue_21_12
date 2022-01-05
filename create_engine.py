import pickle

import numpy as np
import tensorrt as trt
import ctypes

ctypes.CDLL('libclipplugin.so')

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(TRT_LOGGER)
config = builder.create_builder_config()

network = builder.create_network()
input_tensor = network.add_input(name='inputs', dtype=trt.float32, shape=[10])

PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list
plugin_creator = [x for x in PLUGIN_CREATORS if x.name == 'CustomClipPlugin'][0]
clipMin_field = trt.PluginField("clipMin", np.array([0.1], dtype=np.float32),
                                trt.PluginFieldType.FLOAT32)
clipMax_field = trt.PluginField("clipMax", np.array([0.5], dtype=np.float32),
                                trt.PluginFieldType.FLOAT32)
field_collection = trt.PluginFieldCollection([clipMin_field, clipMax_field])
plugin = plugin_creator.create_plugin(name='custom_plugin', field_collection=field_collection)

layer = network.add_plugin_v2(inputs=[input_tensor], plugin=plugin)
output = layer.get_output(0)
output.name = 'output'
network.mark_output(output)
config.max_workspace_size = 1 << 25
builder.max_batch_size = 1
engine = builder.build_engine(network, config)

with open('model', "wb") as f:
    f.write(engine.serialize())
