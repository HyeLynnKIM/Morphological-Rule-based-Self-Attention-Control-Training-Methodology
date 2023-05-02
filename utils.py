import tensorflow as tf
from layers_ import *


def structured_sa_embedding(H, sequence_length, hidden_size, d_a_size, r_size, p_coef, name, fc_size=None, projection=False, reuse=False):
    if fc_size is None:
        fc_size = hidden_size * 2

    with tf.variable_scope(name, reuse=reuse):
        with tf.name_scope("self-attention"):
            H_reshape = tf.reshape(H, [-1, 2 * hidden_size])

            W_s1 = tf.get_variable("W_s1", shape=[2 * hidden_size, d_a_size],
                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            _H_s1 = tf.nn.tanh(tf.matmul(H_reshape, W_s1))
            W_s2 = tf.get_variable("W_s2", shape=[d_a_size, r_size],
                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            _H_s2 = tf.matmul(_H_s1, W_s2)
            print(_H_s2)
            _H_s2_reshape = tf.transpose(tf.reshape(_H_s2, [-1, sequence_length, r_size]), [0, 2, 1])
            A = tf.nn.softmax(_H_s2_reshape, name="attention")

        with tf.name_scope("sentence-embedding"):
            M = tf.matmul(A, H)

        if projection is True:
            M_flat = tf.reshape(M, shape=[-1, 2 * hidden_size * r_size])
            fc = Fully_Connected(M_flat, fc_size, 'fc_layer', activation=tf.nn.tanh, reuse=reuse)

            M_flat = fc
        else:
            M_flat = tf.reshape(M, shape=[-1, 2 * hidden_size * r_size])

        with tf.name_scope("penalization"):
            AA_T = tf.matmul(A, tf.transpose(A, [0, 2, 1]))
            I = tf.reshape(tf.tile(tf.eye(r_size), [tf.shape(A)[0], 1]), [-1, r_size, r_size])
            P = tf.square(tf.norm(AA_T - I, axis=[-2, -1], ord="fro"))

        with tf.name_scope("penalization_loss"):
            loss_P = tf.reduce_mean(P * p_coef)

    return M_flat, loss_P

def self_attention_block(inputs, num_filters, seq_len, mask = None, num_heads = 8,
                         scope = "self_attention_ffn", reuse = None, is_training = True,
                         bias = True, dropout = 0.05, sublayers = (1, 1)):
    with tf.variable_scope(scope, reuse=reuse):
        # Self attention
        outputs = norm_fn(inputs, scope = "layer_norm_1", reuse = reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = multihead_attention(outputs, num_filters,
            num_heads = num_heads, seq_len = seq_len, reuse = reuse,
            mask = mask, is_training = is_training, bias = bias, dropout = dropout)

        residual = outputs + inputs

        # Feed-forward
        outputs = norm_fn(residual, scope = "layer_norm_2", reuse = reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = conv(outputs, num_filters, True, tf.nn.relu, name = "FFN_1", reuse = reuse)
        outputs = conv(outputs, num_filters, True, None, name = "FFN_2", reuse = reuse)
        outputs = residual + outputs

        return outputs

def multihead_attention(queries, units, num_heads,
                        memory = None,
                        seq_len = None,
                        scope = "Multi_Head_Attention",
                        reuse = None,
                        mask = None,
                        is_training = True,
                        bias = True,
                        dropout = 0.05):
    with tf.variable_scope(scope, reuse = reuse):
        # Self attention
        if memory is None:
            memory = queries

        memory = conv(memory, 2 * units, name = "memory_projection", reuse=reuse)
        #print(memory)
        query = conv(queries, units, name = "query_projection", reuse=reuse)
        Q = split_last_dimension(query, num_heads)
        K, V = [split_last_dimension(tensor, num_heads) for tensor in tf.split(memory,2,axis = 2)]

        key_depth_per_head = units // num_heads
        Q *= key_depth_per_head**-0.5
        x = dot_product_attention(Q,K,V,
                                  bias = bias,
                                  seq_len = seq_len,
                                  mask = mask,
                                  is_training = is_training,
                                  scope = "dot_product_attention",
                                  reuse = reuse, dropout = dropout)
        #print(x)
        return combine_last_two_dimensions(tf.transpose(x,[0,2,1,3]))


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def convolution_block(inputs, reuse, num_conv_layers, dropout, num_filters, kernel_size=1, is_training=True):
    outputs = tf.expand_dims(inputs, 2)
    for i in range(num_conv_layers):
        residual = outputs
        outputs = norm_fn(outputs, scope="layer_norm_%d" % i, reuse=reuse)
        if (i) % 2 == 0:
            outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = depthwise_separable_convolution(outputs,
                                                  kernel_size=(kernel_size, 1), num_filters=num_filters,
                                                  scope="depthwise_conv_layers_%d" % i, is_training=is_training,
                                                  reuse=reuse)
        d_bn = batch_norm(name='d_bn' + str(i))

        #print(outputs)
        #print(residual)

        outputs = tf.nn.leaky_relu(d_bn(outputs)) + residual

    return tf.squeeze(outputs,2)

def residual_conv_block(inputs, num_blocks, num_conv_layer, kernel_size, num_filters, input_projection,
                        seq_len, num_heads, name, is_training=True, dropout=0.0, bias=True, reuse=False):
    with tf.variable_scope(name, reuse = reuse):
        if input_projection:
            inputs = conv(inputs, num_filters, name = "input_projection", reuse = reuse)
        outputs = inputs

        for i in range(num_blocks):
            with tf.variable_scope(name + str(i), reuse=reuse):
            #outputs = add_timing_signal_1d(outputs)
                #print('check', reuse)
                outputs = convolution_block(outputs, reuse, num_conv_layer, dropout, num_filters, kernel_size, is_training)
                #print('filter', num_filters)
                #print(outputs)
                outputs = self_attention_block(outputs, num_filters, seq_len, mask=None, num_heads=num_heads,
                                                         scope="self_attention_layers%d" % i, reuse=reuse,
                                                         is_training=is_training,
                                                         bias=bias, dropout=dropout)

        return outputs


def Highway_Network_Fullyconnceted(x, dropout, name, padding, size, activation=tf.sigmoid, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        if padding is True:
            x = Fully_Connected(x, size, name='padding', activation=activation)

        T = Fully_Connected(x, size, 'transform_gate', tf.sigmoid, reuse)
        H = Fully_Connected(x, size, 'activation', activation, reuse)
        C = tf.subtract(1.0, T, name="carry_gate")

        y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")
        return y


def seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def cos_sim(v1, v2):
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
    dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

    return dot_products / (norm1 * norm2)


def difference_loss(private_samples, shared_samples, weight=1.0, name=''):
  private_samples -= tf.reduce_mean(private_samples, 0)
  shared_samples -= tf.reduce_mean(shared_samples, 0)
  private_samples = tf.nn.l2_normalize(private_samples, 1)
  shared_samples = tf.nn.l2_normalize(shared_samples, 1)
  correlation_matrix = tf.matmul( private_samples, shared_samples, transpose_a=True)
  cost = tf.reduce_mean(tf.square(correlation_matrix)) * weight
  cost = tf.where(cost > 0, cost, 0, name='value')
  #tf.summary.scalar('losses/Difference Loss {}'.format(name),cost)
  assert_op = tf.Assert(tf.is_finite(cost), [cost])
  with tf.control_dependencies([assert_op]):
     tf.losses.add_loss(cost)
  return cost


def similiary_score(H_P, H_Q):
    with tf.variable_scope("output") as scope:
        sim = cos_sim(H_P, H_Q)

        score = tf.contrib.layers.fully_connected(
            inputs=sim,
            num_outputs=2,
            activation_fn=tf.nn.sigmoid,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
            biases_initializer=tf.constant_initializer(1e-04),
            scope="FC"
        )

        return score


def Fully_Connected(inp, output, name, activation, initializer_range=3e-7, reuse=False):
    h = tf.contrib.layers.fully_connected(
        inputs=inp,
        num_outputs=output,
        activation_fn=activation,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=3e-7),
        biases_initializer=tf.constant_initializer(3e-7),
        scope=name,
        reuse=reuse
    )

    return h


def class_pred_net(feat, name='class_pred', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        net = Fully_Connected(feat, 256, 'fc1', tf.nn.tanh)
        net = Fully_Connected(net, 128, 'fc2', tf.nn.tanh)
        net = Fully_Connected(net, 64, 'fc2', tf.nn.tanh)

        net = Fully_Connected(net, 2, 'out', None)
    return net


# DOMAIN PREDICTION
def domain_pred_net(feat, flip_gradient, name='domain_pred', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        feat = flip_gradient(feat, -1.0) # GRADIENT REVERSAL
        net = Fully_Connected(feat, 100, 'fc1', tf.nn.tanh)
        net = Fully_Connected(net, 100, 'fc2', tf.nn.tanh)
        net = Fully_Connected(net, 2, 'out', None)
    return net


initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                             mode='FAN_IN',
                                                             uniform=False,
                                                             dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)

def highway(x, size = None, activation = None,
            num_layers=2, scope = "highway", dropout=0.0, reuse=None):
    with tf.variable_scope(scope, reuse):
        if size is None:
            size = x.shape.as_list()[-1]
        else:
            x = conv(x, size, name = "input_projection", reuse = reuse)
        for i in range(num_layers):
            T = conv(x, size, bias = True, activation = tf.sigmoid,
                     name = "gate_%d"%i, reuse = reuse)
            H = conv(x, size, bias = True, activation = activation,
                     name = "activation_%d"%i, reuse = reuse)
            H = tf.nn.dropout(H, 1.0 - dropout)
            x = H * T + x * (1.0 - T)
        return x

def conv(inputs, output_size, bias = None, activation = None, kernel_size = 1, name = "conv", reuse = None):
    with tf.variable_scope(name, reuse = reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1,kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,1,output_size]
            strides = [1,1,1,1]
        else:
            filter_shape = [kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                        filter_shape,
                        dtype = tf.float32,
                        regularizer=regularizer,
                        initializer = initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, "VALID")
        if bias:
            outputs += tf.get_variable("bias_",
                        bias_shape,
                        regularizer=regularizer,
                        initializer = tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs