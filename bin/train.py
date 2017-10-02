import init as init
import paddle.v2 as paddle
import wanxiandata as wd
import cPickle
import copy
import os
from pprint import pprint
def get_cost():
    with open('model/test_cost' , 'r') as f:
        return float(f.read())

def write_cost(cost):
    with open('model/test_cost' , 'w') as f:
        return f.write(str(cost))

def main():
    # init cost file
    write_cost(1.0)

    parameters , cost , inference , feeding = init.init()
    trainer = paddle.trainer.SGD(
        cost=cost,
        parameters=parameters,
        update_equation=paddle.optimizer.Adam(learning_rate=1e-4))

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id > 0:
                print "Pass %d, Batch %d, Cost %f" % (
                    event.pass_id, event.batch_id, event.cost)
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(
                reader=paddle.batch(
                    paddle.reader.shuffle(
                        wd.test(), buf_size=81920),
                    batch_size=256),
                        feeding=feeding)
            print "Test with Pass %d, Cost %f\n" % (
                    event.pass_id, result.cost)
            if result.cost < get_cost():
                print "result.cost %f , stored.cost %f \n" % (result.cost ,
                        get_cost())
                write_cost(result.cost)
                if os.path.isfile('model/params_pass_best') :
                    os.remove('model/params_pass_best')
                with open('model/params_pass_best.tar' , 'w') as f:
                    parameters.to_tar(f)

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(
                wd.train(), buf_size=81920),
            batch_size=256),
        event_handler=event_handler,
        feeding=feeding,
        num_passes=5)

if __name__ == '__main__':
    main()

