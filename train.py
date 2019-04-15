#!/bin/env python
# -*- encoding:utf-8 -*-

from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy as np
import math
import sys
import os

cluster_train_dir = "./train_data"
cluster_test_dir = "./test_data"


def cluster_data_reader(file_dir):
    """
    cluster data reader
    """
    def data_reader():
        """
        data reader
        """
        files = os.listdir(file_dir)
        for fi in files:
            with open(file_dir + '/' + fi, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ins = np.fromstring(line, dtype=float, sep='\t')
                        d1 = ins[:1]
                        d2 = ins[1:2]
                        d3 = ins[2:3]
                        d4 = ins[3:4]
                        d5 = ins[4:5]
                        d6 = ins[5:6]
                        d7 = ins[6:7]
                        d8 = ins[7:8]
                        d9 = ins[8:9]
                        d10 = ins[9:10]
                        d11 = ins[10:11]
                        d12 = ins[11:12]
                        d13 = ins[12:13]
                        d14 = ins[13:14]
                        d15 = ins[14:15]
                        d16 = ins[15:16]
                        d17 = ins[16:17]
                        d18 = ins[17:18]
                        d19 = ins[18:-1]
                        label = ins[-1:]

                        yield d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,label
    return data_reader


def feature_emb(ipt, dict_size, emb_size):
    embed = fluid.layers.embedding(input=ipt,
                                   dtype='float32',
                                   size=[dict_size, emb_size])
    embed_out = fluid.layers.fc(input=embed, size=eƒmb_size)
    return embed_out


class DataConfig():
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype


def model():
    trade_conf = DataConfig('trade', [-1, 1], 'int64')
    trade2_conf = DataConfig('trade2', [-1, 1], 'int64')
    trade3_conf = DataConfig('trade3', [-1, 1], 'int64')
    convtype_conf = DataConfig('convtype', [-1, 1], 'int64')
    level_conf = DataConfig('level', [-1, 1], 'int64')
    flash_conf = DataConfig('flash', [-1, 1], 'int64')
    net_conf = DataConfig('net', [-1, 1], 'int64')
    age_conf = DataConfig('age', [-1, 1], 'int64')
    sex_conf = DataConfig('sex', [-1, 1], 'int64')
    province_conf = DataConfig('province', [-1, 1], 'int64')
    city_conf = DataConfig('city', [-1, 1], 'int64')
    stage_conf = DataConfig('stage', [-1, 1], 'int64')
    utrade_conf = DataConfig('utrade', [-1, 1], 'int64')
    edu_conf = DataConfig('edu', [-1, 1], 'int64')
    job_conf = DataConfig('job', [-1, 1], 'int64')
    income_conf = DataConfig('income', [-1, 1], 'int64')
    consumption_conf = DataConfig('consumption', [-1, 1], 'int64')
    interest_conf = DataConfig('interest', [-1, 1], 'int64')
    ocr_conf = DataConfig('ocr_emb', [-1, 128], 'float32')
    label_conf = DataConfig('label', [-1, 1], 'int64')
    all_data_conf = [trade_conf, trade2_conf, trade3_conf, convtype_conf, level_conf, flash_conf,
                     net_conf, age_conf, sex_conf, province_conf, city_conf, stage_conf, utrade_conf,
                     edu_conf, job_conf, income_conf, consumption_conf, interest_conf,
                     ocr_conf, label_conf]
    data_shapes = [data.shape for data in all_data_conf]
    data_types = [data.dtype for data in all_data_conf]

    py_reader = fluid.layers.py_reader(capacity=64,
                                       shapes=data_shapes,
                                       dtypes=data_types,
                                       name='py_reader',
                                       use_double_buffer=True)

    trade, trade2, trade3, convtype, level, flash, net, age, sex, province, \
        city, stage, utrade, edu, job, income, consumption, interest, ocr_emb, \
        label = fluid.layers.read_file(py_reader)
    E_trade = feature_emb(trade, 1, 2)
    E_trade2 = feature_emb(trade2, 5, 4)
    E_trade3 = feature_emb(trade3, 23, 8)
    E_convtype = feature_emb(convtype, 7, 4)
    E_level = feature_emb(level, 3, 2)
    E_flash = feature_emb(flash, 3, 2)
    E_net = feature_emb(net, 6, 4)
    E_age = feature_emb(age, 6, 4)
    E_sex = feature_emb(sex, 3, 2)
    E_province = feature_emb(province, 37, 8)
    E_city = feature_emb(city, 446,16)
    E_stage = feature_emb(stage, 15, 4)
    E_utrade = feature_emb(utrade, 25, 8)
    E_edu = feature_emb(edu, 8, 4)
    E_job = feature_emb(job, 9, 4)
    E_income = feature_emb(income, 9, 4)
    E_consumption = feature_emb(comsumption, 8, 4)
    E_interest = feature_emb(interest, 729, 16)

    user_vec = [E_net, E_age, E_sex, E_province, E_city, E_stage, E_utrade,
                E_edu, E_job, E_income, E_consumption, E_interest]
    ad_vec = [E_trade, E_trade2, E_trade3, E_convtype, E_level, E_flash, ocr_emb]
    user_concat = fluid.layers.concat(input=user_vec, axis=1)
    ad_concat = fluid.layers.concat(input=ad_vec, axis=1)
    user_fc = fluid.layers.fc(input=user_concat, size=64, act='tanh')
    ad_fc = fluid.layers.fc(input=ad_concat, size=64, act='tanh')
    all_concat = fluid.layers.concat(input=[user_fc, ad_fc], axis=1)
    predict = fluid.layers.fc(input=all_concat, size=2, act='softmax')
    auc_var, cur_auc, auc_states = fluid.layers.auc(input=predict, label=label, slide_steps=20)
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.reduce_sum(cost)
    avg_cost = fluid.layers.scale(x=avg_cost, scale=int(os.environ['CPU_NUM']))
    datalist = [trade, trade2, trade3, convtype, level, flash, net, age, sex, province,
                city, stage, utrade, edu, job, income, consumption, interest, ocr_emb, label]

    return datalist, predict, auc_var, cur_auc, avg_cost, label, py_reader


def infer(use_cuda, save_dirname=None):
    """
    infer
    """
    if save_dirname is None:
        return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be feeded
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)


def train(use_cuda, save_dirname, is_local):
    datalist, predict, auc_var, cur_auc, avg_cost, label, py_reader = model()

    adam_opt = fluid.optimizer.AdamOptimizer()
    adam_opt.minimize(avg_cost)

    BATCH_SIZE = 256
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            cluster_data_reader(cluster_train_dir), buf_size=1000),
        batch_size=BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    training_role = os.getenv("TRAINING_ROLE", "TRAINER")
    if training_role == "PSERVER":
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    def train_loop(main_program):
        """
        train_loop
        """
        feeder = fluid.DataFeeder(place=place, feed_list=datalist)
        exe.run(fluid.default_startup_program())

        PASS_NUM = 200
        for pass_id in range(PASS_NUM):
            for data in train_reader():
                avg_loss_value, = exe.run(main_program,
                                          feed=feeder.feed(data),
                                          fetch_list=[avg_cost])
                print(avg_loss_value)
                if avg_loss_value[0] < 0.02:
                    if save_dirname is not None:
                        fluid.io.save_inference_model(save_dirname, [],
                                                      [predict], exe)
                    exe.close()
                    return
                if math.isnan(float(avg_loss_value)):
                    sys.exit("got NaN loss, training failed.")
        raise AssertionError("Fit a line cost is too large, {0:2.2}".format(
            avg_loss_value[0]))

    if is_local:
        train_loop(fluid.default_main_program())
    else:
        port = os.getenv("PADDLE_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVERS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        trainers = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))
        current_endpoint = os.getenv("POD_IP") + ":" + port
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        t = fluid.DistributeTranspiler()
        t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            train_loop(t.get_trainer_program())


def main(use_cuda, is_local=True):
    """
    main
    """
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    # Directory for saving the trained model
    save_dirname = "./output/model/"
    if not os.path.isdir(save_dirname):
        os.makedirs(save_dirname)

    train(use_cuda, save_dirname, is_local)
    # infer(use_cuda, save_dirname)


if __name__ == '__main__':
    use_cuda = os.getenv("PADDLE_USE_GPU", "0") == "1"
    is_local = os.getenv("PADDLE_IS_LOCAL", "0") == "1"
    main(use_cuda, is_local)
