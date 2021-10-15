"""
Excerpt from https://github.com/jkiesele/caloGraphNN/blob/6d1127d807bc0dbaefcf1ed804d626272f002404/caloGraphNN_keras.py
"""

import tensorflow.keras as keras
K = keras.backend

try:
    from qkeras import QDense, ternary, QActivation

    class NamedQDense(QDense):
        def add_weight(self, name=None, **kwargs):
            return super(NamedQDense, self).add_weight(name='%s_%s' % (self.name, name), **kwargs)

    def ternary_1_05():
        return ternary(alpha=1., threshold=0.5)

except ImportError:
    pass

# Hack keras Dense to propagate the layer name into saved weights
class NamedDense(keras.layers.Dense):
    def add_weight(self, name=None, **kwargs):
        return super(NamedDense, self).add_weight(name='%s_%s' % (self.name, name), **kwargs)

class GarNet(keras.layers.Layer):
    def __init__(self, n_aggregators, n_filters, n_propagate, total_bits, int_bits,
                 simplified=False,
                 collapse=None,
                 input_format='xn',
                 output_activation='tanh',
                 mean_by_nvert=False,
                 quantize_transforms=False,
                 **kwargs):
        super(GarNet, self).__init__(**kwargs)

        self._simplified = simplified
        self._output_activation = output_activation
        self._quantize_transforms = quantize_transforms
        self._total_bits = total_bits
        self._int_bits = int_bits
        self._setup_aux_params(collapse, input_format, mean_by_nvert)
        self._setup_transforms(n_aggregators, n_filters, n_propagate, total_bits, int_bits)

    def _setup_aux_params(self, collapse, input_format, mean_by_nvert):
        if collapse is None:
            self._collapse = None
        elif collapse in ['mean', 'sum', 'max']:
            self._collapse = collapse
        else:
            raise NotImplementedError('Unsupported collapse operation')

        self._input_format = input_format
        self._mean_by_nvert = mean_by_nvert

    def _setup_transforms(self, n_aggregators, n_filters, n_propagate, total_bits, int_bits):
        if self._quantize_transforms:
            self._input_feature_transform = NamedQDense(n_propagate, 
                                                        kernel_quantizer="quantized_bits(%i,%i,0,alpha=1)" %(total_bits, int_bits), 
                                                        bias_quantizer="quantized_bits(%i,%i,0,alpha=1)" %(total_bits, int_bits), 
                                                        name='FLR')
            self._output_feature_transform = NamedQDense(n_filters, kernel_quantizer="quantized_bits(%i,%i,0,alpha=1)" %(total_bits, int_bits), 
                                                         name='Fout')
            if (self._output_activation == None or self._output_activation == "linear"):
                self._output_activation_transform = QActivation("quantized_bits(%i, %i)" %(total_bits, int_bits))
            else:
                self._output_activation_transform = QActivation("quantized_%s(%i, %i)" %(self._output_activation, total_bits, int_bits))
        else:
            self._input_feature_transform = NamedDense(n_propagate, name='FLR')
            self._output_feature_transform = NamedDense(n_filters, activation=self._output_activation, name='Fout')
            self._output_activation_transform = keras.layers.Activation(self._output_activation)

        self._aggregator_distance = NamedDense(n_aggregators, name='S')

        self._sublayers = [self._input_feature_transform, self._aggregator_distance, self._output_feature_transform, self._output_activation_transform]

    def build(self, input_shape):
        super(GarNet, self).build(input_shape)

        if self._input_format == 'x':
            data_shape = input_shape
        elif self._input_format == 'xn':
            data_shape, _ = input_shape
        elif self._input_format == 'xen':
            data_shape, _, _ = input_shape
            data_shape = data_shape[:2] + (data_shape[2] + 1,)

        self._build_transforms(data_shape)

        for layer in self._sublayers:
            self._trainable_weights.extend(layer.trainable_weights)
            self._non_trainable_weights.extend(layer.non_trainable_weights)

    def _build_transforms(self, data_shape):
        self._input_feature_transform.build(data_shape)
        self._aggregator_distance.build(data_shape)
        if self._simplified:
            self._output_activation_transform.build(self._output_feature_transform.build(data_shape[:2] + (self._aggregator_distance.units * self._input_feature_transform.units,)))
        else:
            self._output_activation_transform.build(self._output_feature_transform.build(data_shape[:2] + (data_shape[2] + self._aggregator_distance.units * self._input_feature_transform.units + self._aggregator_distance.units,)))

    def call(self, x):
        data, num_vertex, vertex_mask = self._unpack_input(x)

        output = self._garnet(data, num_vertex, vertex_mask,
                              self._input_feature_transform,
                              self._aggregator_distance,
                              self._output_feature_transform,
                              self._output_activation_transform)

        output = self._collapse_output(output)

        return output

    def _unpack_input(self, x):
        if self._input_format == 'x':
            data = x

            vertex_mask = K.cast(K.not_equal(data[..., 3:4], 0.), 'float32')
            num_vertex = K.sum(vertex_mask)

        elif self._input_format in ['xn', 'xen']:
            if self._input_format == 'xn':
                data, num_vertex = x
            else:
                data_x, data_e, num_vertex = x
                data = K.concatenate((data_x, K.reshape(data_e, (-1, data_e.shape[1], 1))), axis=-1)
    
            data_shape = K.shape(data)
            B = data_shape[0]
            V = data_shape[1]
            vertex_indices = K.tile(K.expand_dims(K.arange(0, V), axis=0), (B, 1)) # (B, [0..V-1])
            vertex_mask = K.expand_dims(K.cast(K.less(vertex_indices, K.cast(num_vertex, 'int32')), 'float32'), axis=-1) # (B, V, 1)
            num_vertex = K.cast(num_vertex, 'float32')

        return data, num_vertex, vertex_mask

    def _garnet(self, data, num_vertex, vertex_mask, in_transform, d_compute, out_transform, act_transform):
        features = in_transform(data) # (B, V, F)
        distance = d_compute(data) # (B, V, S)

        edge_weights = vertex_mask * K.exp(-K.square(distance)) # (B, V, S)

        if not self._simplified:
            features = K.concatenate([vertex_mask * features, edge_weights], axis=-1)
        
        if self._mean_by_nvert:
            def graph_mean(out, axis):
                s = K.sum(out, axis=axis)
                # reshape just to enable broadcasting
                s = K.reshape(s, (-1, d_compute.units * in_transform.units)) / num_vertex
                s = K.reshape(s, (-1, d_compute.units, in_transform.units))
                return s
        else:
            graph_mean = K.mean

        # vertices -> aggregators
        edge_weights_trans = K.permute_dimensions(edge_weights, (0, 2, 1)) # (B, S, V)

        aggregated_mean = self._apply_edge_weights(features, edge_weights_trans, aggregation=graph_mean) # (B, S, F)

        if self._simplified:
            aggregated = aggregated_mean
        else:
            aggregated_max = self._apply_edge_weights(features, edge_weights_trans, aggregation=K.max)
            aggregated = K.concatenate([aggregated_max, aggregated_mean], axis=-1)

        # aggregators -> vertices
        updated_features = self._apply_edge_weights(aggregated, edge_weights) # (B, V, S*F)

        if not self._simplified:
            updated_features = K.concatenate([data, updated_features, edge_weights], axis=-1)

        return vertex_mask * act_transform(out_transform(updated_features))

    def _collapse_output(self, output):
        if self._collapse == 'mean':
            if self._mean_by_nvert:
                output = K.sum(output, axis=1) / num_vertex
            else:
                output = K.mean(output, axis=1)
        elif self._collapse == 'sum': 
           output = K.sum(output, axis=1)
        elif self._collapse == 'max':
            output = K.max(output, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return self._get_output_shape(input_shape, self._output_activation_transform)

    def _get_output_shape(self, input_shape, out_transform):
        if self._input_format == 'x':
            data_shape = input_shape
        elif self._input_format == 'xn':
            data_shape, _ = input_shape
        elif self._input_format == 'xen':
            data_shape, _, _ = input_shape

        if self._collapse is None:
            return data_shape[:2] + (out_transform.units,)
        else:
            return (data_shape[0], out_transform.units)

    def get_config(self):
        config = super(GarNet, self).get_config()

        config.update({
            'simplified': self._simplified,
            'collapse': self._collapse,
            'input_format': self._input_format,
            'output_activation': self._output_activation,
            'quantize_transforms': self._quantize_transforms,
            'mean_by_nvert': self._mean_by_nvert
        })

        self._add_transform_config(config)

        return config

    def _add_transform_config(self, config):
        config.update({
            'n_aggregators': self._aggregator_distance.units,
            'n_filters': self._output_feature_transform.units,
            'n_propagate': self._input_feature_transform.units,
            'total_bits': self._total_bits,
            'int_bits': self._int_bits
        })

    @staticmethod
    def _apply_edge_weights(features, edge_weights, aggregation=None):
        features = K.expand_dims(features, axis=1) # (B, 1, v, f)
        edge_weights = K.expand_dims(edge_weights, axis=3) # (B, u, v, 1)

        out = edge_weights * features # (B, u, v, f)

        if aggregation:
            out = aggregation(out, axis=2) # (B, u, f)
        else:
            try:
                out = K.reshape(out, (-1, edge_weights.shape[1].value, features.shape[-1].value * features.shape[-2].value))
            except AttributeError: # TF 2
                out = K.reshape(out, (-1, edge_weights.shape[1], features.shape[-1] * features.shape[-2]))
        
        return out