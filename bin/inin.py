import paddle.v2 as paddle
import wanxiandata as wd
import cPickle
import copy
from pprint import pprint
__all__ = [
    'init',
]

def get_usr_combined_features():

    uid = paddle.layer.data(
        name='user_id',
        type=paddle.data_type.integer_value(100000))
    usr_emb = paddle.layer.embedding(input=uid, size=32)
    usr_fc = paddle.layer.fc(input=usr_emb, size=32)

    usr_combined_features = paddle.layer.fc(
        input=[usr_fc],
        size=200,
        act=paddle.activation.Tanh())
    return usr_combined_features


def get_con_combined_features():
    cid = paddle.layer.data(
        name='cid',
        type=paddle.data_type.integer_value(2560000))
    con_emb = paddle.layer.embedding(input=cid, size=32)
    con_fc = paddle.layer.fc(input=con_emb, size=32)

    num = paddle.layer.data(
        name='num',
        type=paddle.data_type.integer_value(100))
    num_emb = paddle.layer.embedding(input=num, size=32)
    num_fc = paddle.layer.fc(input=num_emb, size=32)

    con_sc1 = paddle.layer.data(
        name='sc1',
        type=paddle.data_type.integer_value(10000))
    sc1_emb = paddle.layer.embedding(input=con_sc1, size=32)
    sc1_fc = paddle.layer.fc(input=sc1_emb, size=32)


    con_sc2 = paddle.layer.data(
        name='sc2',
        type=paddle.data_type.integer_value(10000))
    sc2_emb = paddle.layer.embedding(input=con_sc2, size=32)
    sc2_fc = paddle.layer.fc(input=sc2_emb, size=32)

    con_combined_features = paddle.layer.fc(
        input=[con_fc, num_fc , sc1_fc , sc2_fc],
        size=200,
        act=paddle.activation.Tanh())
    return con_combined_features

def init():
    paddle.init(use_gpu=False , trainer_count=7)
    usr_combined_features = get_usr_combined_features()
    con_combined_features = get_con_combined_features()
    inference = paddle.layer.cos_sim(
        a=usr_combined_features, b=con_combined_features, size=1, scale=5)
    cost = paddle.layer.mse_cost(
        input=inference,
        label=paddle.layer.data(
            name='isclick', type=paddle.data_type.dense_vector(1)))
    parameters = paddle.parameters.create(cost)
    feeding = {
        'user_id': 0,
        'cid': 1,
        'isclick': 2,
        'num': 3,
        'sc1': 4,
        'sc2': 5,
    }
    return parameters , cost , inference , feeding
[work@bjyz-video-rec.epc.baidu.com train]$
[work@bjyz-video-rec.epc.baidu.com train]$
[work@bjyz-video-rec.epc.baidu.com train]$
[work@bjyz-video-rec.epc.baidu.com train]$ cat init.py
import paddle.v2 as paddle
import wanxiandata as wd
import cPickle
import copy
from pprint import pprint
__all__ = [
    'init',
]

def get_usr_combined_features():

    uid = paddle.layer.data(
        name='user_id',
        type=paddle.data_type.integer_value(100000))
    usr_emb = paddle.layer.embedding(input=uid, size=32)
    usr_fc = paddle.layer.fc(input=usr_emb, size=32)

    usr_combined_features = paddle.layer.fc(
        input=[usr_fc],
        size=200,
        act=paddle.activation.Tanh())
    return usr_combined_features


def get_con_combined_features():
    cid = paddle.layer.data(
        name='cid',
        type=paddle.data_type.integer_value(2560000))
    con_emb = paddle.layer.embedding(input=cid, size=32)
    con_fc = paddle.layer.fc(input=con_emb, size=32)

    num = paddle.layer.data(
        name='num',
        type=paddle.data_type.integer_value(100))
    num_emb = paddle.layer.embedding(input=num, size=32)
    num_fc = paddle.layer.fc(input=num_emb, size=32)

    con_sc1 = paddle.layer.data(
        name='sc1',
        type=paddle.data_type.integer_value(10000))
    sc1_emb = paddle.layer.embedding(input=con_sc1, size=32)
    sc1_fc = paddle.layer.fc(input=sc1_emb, size=32)


    con_sc2 = paddle.layer.data(
        name='sc2',
        type=paddle.data_type.integer_value(10000))
    sc2_emb = paddle.layer.embedding(input=con_sc2, size=32)
    sc2_fc = paddle.layer.fc(input=sc2_emb, size=32)

    con_combined_features = paddle.layer.fc(
        input=[con_fc, num_fc , sc1_fc , sc2_fc],
        size=200,
        act=paddle.activation.Tanh())
    return con_combined_features

def init():
    paddle.init(use_gpu=False , trainer_count=7)
    usr_combined_features = get_usr_combined_features()
    con_combined_features = get_con_combined_features()
    inference = paddle.layer.cos_sim(
        a=usr_combined_features, b=con_combined_features, size=1, scale=5)
    cost = paddle.layer.mse_cost(
        input=inference,
        label=paddle.layer.data(
            name='isclick', type=paddle.data_type.dense_vector(1)))
    parameters = paddle.parameters.create(cost)
    feeding = {
        'user_id': 0,
        'cid': 1,
        'isclick': 2,
        'num': 3,
        'sc1': 4,
        'sc2': 5,
    }
    return parameters , cost , inference , feeding
