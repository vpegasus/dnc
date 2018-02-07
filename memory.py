#!/usr/bin/env python  
# -*- coding: utf-8 -*-
"""
@author: Prince
@file: memory.py 
@time: 2018-1-30 01: 42
@license: Apache License
@contact: pegasus.wenjia@foxmail.com 
"""
import tensorflow as tf


class Memory:
    def __init__(self, num_memory, word_size, num_read_heads, batch_size):
        """
        memory and its operation

        :param num_memory: N
        :param word_size: W
        :param num_read_heads:R
        :param batch_size: b
        """
        self.batch_size = batch_size
        self.num_memory = num_memory
        self.num_read_heads = num_read_heads
        self.word_size = word_size
        self.initial_paras()

    def initial_paras(self):
        self.memory = tf.fill([self.batch_size, self.num_memory, self.word_size], value=1e-8,
                              name='memory')  # b x N x W
        self.usage = tf.zeros([self.batch_size, 1, self.num_memory], name='memory_usage')
        self.precedence_weight = tf.zeros(shape=[self.batch_size, 1, self.num_memory], dtype=tf.float32,
                                          name='precedence_weights')
        self.write_weights = tf.fill([self.batch_size, 1, self.num_memory], value=1e-8, name='write_weight')
        self.linkage_matrix = tf.zeros(shape=[self.batch_size, self.num_memory, self.num_memory], dtype=tf.float32,
                                       name='linkage_matrix')  # b x N x N
        self.read_weights = tf.fill([self.batch_size, self.num_read_heads, self.num_memory], value=1e-8,
                                    name='read_weights')  # b x R x N
        self.read_heads = tf.fill([self.batch_size, self.num_read_heads, self.word_size], value=1e-8,
                                  name='read_heads')  # b x R x W

    def content_address(self, keys, strengths):
        """
        apply content based addressing.

        :param keys: [batch_size,num_of_keys, word_size] #where num_of_keys equals num_of_reads for read,
                    or 1 for write.
        :param strengths:[batch_size, num_of_keys,1]
        :return: [batch_size, num_of_keys,num_of_memory]
        """
        normalized_memory = tf.nn.l2_normalize(self.memory, 2)  # b x N x W
        normalized_keys = tf.nn.l2_normalize(keys, 2)  # b x r/w x W
        memory_trans = tf.transpose(normalized_memory, perm=[0, 2, 1])
        similarity = tf.matmul(normalized_keys, memory_trans)  # b x r/w x N
        return tf.nn.softmax(similarity * strengths, 2)  # b x r/w x N

    def retention_vector(self, free_gates):
        """
        get retention vector

        :param free_gates: [batch_size,num_read_heads,1]
        :return:
        """
        return tf.reduce_prod(1 - self.read_weights * free_gates, axis=1, keepdims=True)  # b x 1 x N

    def usage_vector(self, retention_vector):
        """
        get usage vector
        :param retention_vector: b x 1 x N
        :return:
        """
        self.usage = (self.usage + self.write_weights - self.usage * self.write_weights) * retention_vector
        return self.usage

    def usage_sort(self, usage):
        """
        sort usage

        :param usage: b x 1 x N
        :return: sorted_usage, and their original order indices.
        """
        top_k_values, top_k_indices = tf.nn.top_k(-1 * usage, k=self.num_memory)
        return -1 * top_k_values, top_k_indices

    def allocation_address(self, sorted_usage, top_k_indices):
        """
        get allocation weights

        :param sorted_usage:[batch_size,1,num_memory]
        :param top_k_indices: [batch_size,1,num_meory]
        :return:[batch_size,1,num_memory]
        """
        usage_cumprod = tf.cumprod(sorted_usage, axis=2, exclusive=True)
        unordered_allocation_weights = (1 - sorted_usage) * usage_cumprod

        # Trick! the following use a trick to order the allocation weights: as the allocation weights are all
        # no bigger than 1, so its effect could be ignored when sorting with the data(which granularity is 1 or
        # bigger) (the data,I use here is just the top_k_indices x 2(in case of the extreme phenomena occur,
        # i.e allocation_weight =1)
        map_sort = unordered_allocation_weights + tf.cast(top_k_indices, tf.float32) * 2.
        allocation, _ = tf.nn.top_k(-1 * map_sort, k=self.num_memory)
        idx = tf.range(0, self.num_memory, dtype=tf.float32) * 2.
        allocation += idx
        return -1 * allocation

    def _get_write_weights(self, write_gate, allocation_gate, allocation_address, content_address):
        """

        :param write_gate: b x 1 x 1
        :param allocation_gate:
        :param allocation_address: b x 1 x N
        :param content_address: b x 1 x N
        :return:
        """
        self.write_weights = write_gate * (
                allocation_gate * allocation_address + (1 - allocation_gate) * content_address)

    def write_to_memory(self, write_vector, erase_vector):
        """

        :param write_vector:[batch_size,1, word_size]
        :param erase_vector:[batch_size,1, word_size]
        :return: [b x N x W]
        """
        weight_write = tf.transpose(self.write_weights, perm=[0, 2, 1])
        self.memory = self.memory * (1 - tf.matmul(weight_write, erase_vector)) + tf.matmul(weight_write, write_vector)

    def write(self, write_key, write_strength, free_gates, write_gate, allocation_gate, erase_vector, write_vector):
        """

        :param write_key:
        :param write_strength:
        :param free_gates:
        :param write_gate:
        :param allocation_gate:
        :param erase_vector:
        :param write_vector:
        :return:
        """
        content_write = self.content_address(write_key, write_strength)
        retention = self.retention_vector(free_gates)
        usage = self.usage_vector(retention)
        sorted_usage, top_k_indices = self.usage_sort(usage)
        allocation = self.allocation_address(sorted_usage, top_k_indices)
        self._get_write_weights(write_gate, allocation_gate, allocation, content_write)
        self.write_to_memory(write_vector, erase_vector)

    def precedence_update(self):
        """

        :return: [b x 1 x N
        """
        self.precedence_weight = (1 - tf.reduce_sum(self.write_weights, axis=2,
                                                    keepdims=True)) * self.precedence_weight + self.write_weights

    def linkage_matrix_update(self):
        """

        :return: b x N x N
        """
        reset_factor = self._linkage_reset_factor()
        p_weight = tf.transpose(self.precedence_weight, perm=[0, 2, 1])
        linkage_matrix = reset_factor * self.linkage_matrix + tf.matmul(p_weight, self.write_weights)
        I = tf.eye(num_rows=self.num_memory, batch_shape=[self.batch_size])
        self.linkage_matrix = linkage_matrix * (1 - I)

    def _linkage_reset_factor(self):
        """

        :return:
        """
        reshape_weight2 = tf.transpose(self.write_weights, [0, 2, 1])  # b x N x 1
        ones1 = tf.ones(shape=[self.batch_size, 1, self.num_memory])  # b x 1 x N
        ones2 = tf.ones(shape=[self.batch_size, self.num_memory, 1])
        reset_factor = 1 - tf.matmul(reshape_weight2, ones1) - tf.matmul(ones2, self.write_weights)
        return reset_factor  # b x N x N

    def temporal_address(self):
        """

        :return: [batch_size, num_read_heads,num_memory]
        """
        forward_address = tf.matmul(self.read_weights, self.linkage_matrix)  # b x R x N * b x N x N
        transpose_link = tf.transpose(self.linkage_matrix, perm=[0, 2, 1])
        backward_address = tf.matmul(self.read_weights, transpose_link)  # b x R x N * b x N x N
        return forward_address, backward_address  # b x R x N

    def _get_read_weights(self, forward_address, backward_address, content_address, mode_weights):
        """

        :param forward_address: b x R x N
        :param backward_address: b x R x N
        :param content_address:
        :param mode_weights:
        :return:  read weights: b x R x N
        """
        mode1 = tf.slice(mode_weights, [0, 0, 0], [-1, -1, 1])
        mode2 = tf.slice(mode_weights, [0, 0, 1], [-1, -1, 1])
        mode3 = tf.slice(mode_weights, [0, 0, 2], [-1, -1, 1])
        self.read_weights = mode1 * forward_address + mode2 * backward_address + mode3 * content_address

    def read_memory(self):
        """
        generate read heads

        :return:
        """
        self.read_heads = tf.matmul(self.read_weights, self.memory)  # b x R x N * b x N x W
        return self.read_heads  # b x R x W

    def read(self, read_keys, read_strengths, read_modes):
        """

        :param read_keys:
        :param read_strengths:
        :param read_modes:
        :return:
        """

        content_read = self.content_address(read_keys, read_strengths)
        self.linkage_matrix_update()
        forward_address, backward_address = self.temporal_address()
        self._get_read_weights(forward_address, backward_address, content_read, read_modes)
        self.read_memory()
        self.precedence_update()
        return self.read_heads


if __name__ == '__main__':
    pass
