import itertools

import tensorflow as tf
import numpy as np

import hls4ml
from hls4ml.model.attributes import Attribute


class NodeEdgeProjection(tf.keras.layers.Layer):
    def __init__(self, receiving=True, node_to_edge=True, **kwargs):
        super().__init__(**kwargs)

        self._receiving = receiving
        self._node_to_edge = node_to_edge

    def build(self, input_shape):
        if self._node_to_edge:
            self._n_nodes = input_shape[-2]
            self._n_edges = self._n_nodes * (self._n_nodes - 1)
        else:
            self._n_edges = input_shape[-2]
            self._n_nodes = int((np.sqrt(4 * self._n_edges + 1) + 1) / 2)
        self._adjacency_matrix = self._assign_adjacency_matrix()

    def _assign_adjacency_matrix(self):
        receiver_sender_list = itertools.permutations(range(self._n_nodes), r=2)

        if self._node_to_edge:
            shape = (1, self._n_edges, self._n_nodes)
        else:
            shape = (1, self._n_nodes, self._n_edges)
        adjacency_matrix = np.zeros(shape, dtype=float)

        for i, (r, s) in enumerate(receiver_sender_list):
            if self._node_to_edge:
                if self._receiving:
                    adjacency_matrix[0, i, r] = 1
                else:
                    adjacency_matrix[0, i, s] = 1
            else:
                if self._receiving:
                    adjacency_matrix[0, r, i] = 1
                else:
                    adjacency_matrix[0, s, i] = 1

        return tf.Variable(
            initial_value=adjacency_matrix,
            name="adjacency_matrix",
            dtype="float32",
            shape=shape,
            trainable=False,
        )

    def call(self, inputs):
        return tf.matmul(self._adjacency_matrix, inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_nodes": self._n_nodes,
                "n_edges": self._n_edges,
                "receiving": self._receiving,
                "node_to_edge": self._node_to_edge,
            }
        )
        return config


# hls4ml layer implementation
class HLSNodeEdgeProjection(hls4ml.model.layers.Layer):
    """hls4ml implementation of NodeEdgeProjection"""

    _expected_attributes = [
        Attribute("n_in"),
        Attribute("n_nodes"),
        Attribute("n_edges"),
        Attribute("receiving", value_type=bool, default=True),
        Attribute("node_to_edge", value_type=bool, default=True),
        Attribute("in_width"),
        Attribute("out_width"),
    ]

    def initialize(self):
        if self.attributes["node_to_edge"]:
            shape = [self.attributes["n_edges"], self.attributes["n_in"]]
        else:
            shape = [self.attributes["n_nodes"], self.attributes["n_in"]]
        dims = [f"N_OUT_{self.index}_0", f"N_OUT_{self.index}_1"]
        self.add_output_variable(shape, dims)


# parser for converter
def parse_node_edge_projection_layer(
    keras_layer, input_names, input_shapes, data_reader
):
    layer = {}
    layer["class_name"] = "HLSNodeEdgeProjection"
    layer["name"] = keras_layer["config"]["name"]
    layer["n_in"] = input_shapes[0][1]
    layer["n_nodes"] = keras_layer["config"]["n_nodes"]
    layer["n_edges"] = keras_layer["config"]["n_edges"]
    layer["receiving"] = keras_layer["config"]["receiving"]
    layer["node_to_edge"] = keras_layer["config"]["node_to_edge"]
    layer["in_width"] = layer["n_nodes"] if layer["node_to_edge"] else layer["n_edges"]
    layer["out_width"] = layer["n_edges"] if layer["node_to_edge"] else layer["n_nodes"]

    output_shapes = [layer["out_width"], layer["n_in"]]

    if input_names is not None:
        layer["inputs"] = input_names

    return layer, output_shapes


# HLS Templates - No specific pragmas used; generic enough for both Intel and Vivado

config_template = """struct config{index} : nnet::node_edge_projection_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_nodes = {n_nodes};
    static const unsigned n_edges = {n_edges};
    static const bool receiving = {receiving};
    static const bool node_to_edge = {node_to_edge};
    static const unsigned in_width = {in_width};
    static const unsigned out_width = {out_width};
}};\n"""

function_template = (
    "nnet::node_edge_projection<{input_t}, {output_t}, {config}>({input}, {output});"
)
include_list = ["nnet_utils/nnet_node_edge_projection.h"]


class HLSNodeEdgeProjectionConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(HLSNodeEdgeProjection)
        self.template = config_template

    def format(self, node):
        params = self._default_config_params(node)
        params["receiving"] = str(params["receiving"]).lower()
        params["node_to_edge"] = str(params["node_to_edge"]).lower()
        return self.template.format(**params)


class HLSNodeEdgeProjectionFunctionTemplate(
    hls4ml.backends.template.FunctionCallTemplate
):
    def __init__(self):
        super().__init__(HLSNodeEdgeProjection, include_header=include_list)
        self.template = function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


def register_custom_layer():
    # Register the converter for custom Keras layer
    hls4ml.converters.register_keras_layer_handler(
        "NodeEdgeProjection", parse_node_edge_projection_layer
    )

    # Register the hls4ml's IR layer
    hls4ml.model.layers.register_layer("HLSNodeEdgeProjection", HLSNodeEdgeProjection)


if __name__ == "__main__":

    # Register custom layer
    register_custom_layer()

    # Register the optimization passes (if any)
    backend = hls4ml.backends.get_backend("Vivado")

    # Register template passes for the given backend
    backend.register_template(HLSNodeEdgeProjectionConfigTemplate)
    backend.register_template(HLSNodeEdgeProjectionFunctionTemplate)

    from pathlib import Path

    # Register HLS implementation
    backend.register_source(Path(Path.cwd() / "nnet_node_edge_projection.h"))

    # Test if it works
    kmodel = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(8, 4)),
            NodeEdgeProjection(),
        ]
    )

    x = np.random.randint(-5, 5, (8, 4), dtype="int32")
    kres = kmodel(x)

    hmodel = hls4ml.converters.convert_from_keras_model(
        kmodel,
        output_dir="hls4mlprj_node_edge_projection",
        backend="Vivado",
        io_type="io_parallel",
        hls_config={"Model": {"Precision": "ap_int<6>", "ReuseFactor": 1}},
    )

    hmodel.compile()
    hres = hmodel.predict(x.astype("float32"))
