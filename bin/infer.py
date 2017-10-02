import init as init
import paddle.v2 as paddle
import wanxiandata as wd
import cPickle
import copy
from pprint import pprint
def main():
    parameters_in_train , cost , inference , feeding = init.init()
    with open('model/params_pass_0.tar', 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)
    infer_dict = copy.copy(feeding)
    del infer_dict['isclick']
    itera = wd.train()()
    for feature in itera:
        prediction = paddle.infer(inference, parameters=parameters, input=[feature], feeding=infer_dict)
        print 'predict = '+ str((prediction[0][0]+1)/2) + ' isclick = '+ str(feature[2])

if __name__ == '__main__':
    main()
