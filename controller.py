#!/usr/bin/env python  
# -*- coding: utf-8 -*-
"""
@author: Prince
@file: controller.py
@time: 2018-1-29 10: 57
@license: Apache License
@contact: pegasus.wenjia@foxmail.com 
"""
import tensorflow as tf
import numpy as np


class Controller:
    def __init__(self, output_size, num_read_heads, memory_word_size, vocab_size, batch_size):
        """
        build a controller

        :param output_size:
        :param num_read_heads:
        :param memory_word_size:
        :param vocab_size:
        :param batch_size:
        """
        self.num_read_heads = num_read_heads
        self.memory_word_size = memory_word_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.interface_vector_size = self.memory_word_size * self.num_read_heads + 3 * self.memory_word_size + 5 * \
                                     self.num_read_heads + 3

        self.interface_weights = tf.get_variable('interface_vector_weights',
                                                 shape=[self.memory_word_size, self.interface_vector_size],
                                                 dtype=tf.float32)
        self.nn_output_weights = tf.get_variable('nn_output_weights', shape=[self.memory_word_size, self.output_size])

        self.final_output_weights = tf.get_variable('final_output_weights',
                                                    shape=[self.num_read_heads * self.memory_word_size,
                                                           self.output_size])
        self.lstm = self.LSTM()
        self.nn_state = None
        self.initial_nn_state()
        # the following statement used when inputs encoded as embedding vector
        # construct an embedding matrix [vocab_size,embedding_size] = [vocab_size, memory_word_size]
        # self.embedding_size = embedding_size
        # if embedding_size:
        #     self.embedding_matrix = tf.get_variable(name='Embedding_matrix', shape=[vocab_size, self.embedding_size],
        #                                             dtype=tf.float32)


    # def data_encoding(self, input_):
    #     """
    #     convert raw input into embedding vectors
    #
    #     :param input_: [batch_size]
    #     :return: [batch_size,embedding_size]
    #     """
    #
    #     if self.embedding_size:
    #         return tf.nn.embedding_lookup(self.embedding_matrix, input_)
    #     else:
    #         vec = np.zeros([self.batch_size, self.vocab_size], dtype=np.float32)
    #         for i, v in enumerate(input_):
    #             vec[i, v] = 1.0
    #         return tf.convert_to_tensor(vec)


    def initial_nn_state(self):
        self.nn_state = tf.nn.rnn_cell.BasicLSTMCell(self.memory_word_size).zero_state(batch_size=self.batch_size,
                                                                                       dtype=tf.float32)

    def inputs_concatenate(self, embedding_x, last_read_heads):
        """
        concatenate input(embedding_x) with last read vectors to form controller input vector

        :param embedding_x: [batch_size, embedding_size]
        :param last_read_heads: [batch_size, num_read_heads, memory_word_size], read head vectors at time t-1
        :return:
        """
        last_read_heads = tf.reshape(last_read_heads, [-1, self.num_read_heads * self.memory_word_size])
        return tf.concat([embedding_x, last_read_heads], 1)

    def LSTM(self):
        return  tf.nn.rnn_cell.BasicLSTMCell(num_units=self.memory_word_size)

    def network(self, input_, lstm_state):
        """
        some network to serve as the main body of controller to process data, this could be changed as needed.
        in the instance, lstm

        :param input_: [batch_size, concatenate_input_size]=[batch_size, input_size + num_read_heads x word_size]
        :param lstm_state: hidden state of rnn
        :return:
        """

        nn_output, nn_state = self.lstm(input_, state=lstm_state)
        return nn_output, nn_state

    def output_layer(self, nn_output):
        """
        convert nn_output into output vector and interface vector
        :param nn_output:
        :return:
        """
        outputs = tf.matmul(nn_output, self.nn_output_weights)
        interface_vector = tf.matmul(nn_output, self.interface_weights)
        return outputs, interface_vector

    def final_output(self, outputs, read_head_vectors):
        """
        generate final output
        :param outputs: [batch_size,output_size], tensor from output layer
        :param read_head_vectors: [batch_size, self.num_read_heads,self.word_size] read head vectors at time t(current)
        :return:
        """
        read_head_vectors = tf.reshape(read_head_vectors, [-1, self.num_read_heads * self.memory_word_size])
        yt = outputs + tf.matmul(read_head_vectors, self.final_output_weights)
        return yt

    def _parameters_transformation(self, parsed_para_dict):
        parsed_para_dict['read_strengths'] = 1 + tf.nn.softplus(parsed_para_dict['read_strengths'])
        parsed_para_dict['write_strength'] = 1 + tf.nn.softplus(parsed_para_dict['write_strength'])
        parsed_para_dict['erase_vector'] = tf.nn.sigmoid(parsed_para_dict['erase_vector'])
        parsed_para_dict['free_gates'] = tf.nn.sigmoid(parsed_para_dict['free_gates'])
        parsed_para_dict['allocation_gate'] = tf.nn.sigmoid(parsed_para_dict['allocation_gate'])
        parsed_para_dict['write_gate'] = tf.nn.sigmoid(parsed_para_dict['write_gate'])
        parsed_para_dict['read_modes'] = tf.nn.softmax(parsed_para_dict['read_modes'])
        return parsed_para_dict

    def parse_interface_parameters(self, interface_vector):
        """
        parse interface vector into various parameters within correct shapes and scopes

        :param interface_vector: [batch_size, interface_vector_size], tensor
        :return: dict, a dictionary with the parameters of the interface vector parsed.
        """
        parsed_para_dict = dict()

        # the followings are to be used for tensor slicing
        read_keys_end = self.memory_word_size * self.num_read_heads
        read_strengths_end = read_keys_end + self.num_read_heads
        write_key_end = read_strengths_end + self.memory_word_size
        erase_end = write_key_end + 1 + self.memory_word_size
        write_end = erase_end + self.memory_word_size
        free_gates_end = write_end + self.num_read_heads

        # parameters' shapes after slicing, Attention! the following should be defined carefully!!!
        read_keys_shape = [-1, self.num_read_heads, self.memory_word_size]
        read_strengths_shape = [-1, self.num_read_heads, 1]
        write_key_shape = [-1, 1, self.memory_word_size]
        write_shape = [-1, 1, self.memory_word_size]
        erase_shape = [-1, 1, self.memory_word_size]
        free_gates_shape = [-1, self.num_read_heads, 1]
        modes_shape = [-1, self.num_read_heads, 3]

        parsed_para_dict['read_keys'] = tf.reshape(interface_vector[:, :read_keys_end], read_keys_shape)
        parsed_para_dict['read_strengths'] = tf.reshape(interface_vector[:, read_keys_end:read_strengths_end],
                                                        read_strengths_shape)
        parsed_para_dict['write_key'] = tf.reshape(interface_vector[:, read_strengths_end:write_key_end],
                                                   write_key_shape)
        parsed_para_dict['write_strength'] = tf.reshape(interface_vector[:, write_key_end], [-1, 1, 1])
        parsed_para_dict['erase_vector'] = tf.reshape(interface_vector[:, write_key_end + 1:erase_end], erase_shape)
        parsed_para_dict['write_vector'] = tf.reshape(interface_vector[:, erase_end:write_end], write_shape)
        parsed_para_dict['free_gates'] = tf.reshape(interface_vector[:, write_end:free_gates_end], free_gates_shape)
        parsed_para_dict['allocation_gate'] = tf.reshape(interface_vector[:, free_gates_end], [-1, 1, 1])
        parsed_para_dict['write_gate'] = tf.reshape(interface_vector[:, free_gates_end + 1], [-1, 1, 1])
        parsed_para_dict['read_modes'] = tf.reshape(interface_vector[:, free_gates_end + 2:], modes_shape)
        parsed_para_dict = self._parameters_transformation(parsed_para_dict)
        return parsed_para_dict

    def __call__(self, input_, last_read_heads):
        """

        :param input_: [batch_size,1] i.e. x_t
        :param last_read_heads: [batch_size,word_size]
        :return:
        """
        complete_input = self.inputs_concatenate(input_, last_read_heads)
        nn_output, self.nn_state = self.network(complete_input, self.nn_state)
        output_, interface_vector = self.output_layer(nn_output)
        parsed_interface = self.parse_interface_parameters(interface_vector)
        return output_, parsed_interface


if __name__ == '__main__':
    contr = Controller(output_size=5, num_read_heads=3, memory_word_size=12, vocab_size=20, batch_size=7,
                       one_hot_size=159)
    inputs = [1, 2, 3]
    last_read_vectors = tf.random_normal([7, 3, 12])
    output, parsed = contr(inputs, last_read_vectors)
