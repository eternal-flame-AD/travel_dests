import tensorflow as tf
import travel_data
import os
import pprint
import progressbar


SAVENAME="NO_SAVE"
'''
graph=tf.Graph()
graph.as_default()
'''

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, travel_data.CITY_COUNT])
    y_ans = tf.placeholder(tf.float32, [None, travel_data.CITY_COUNT])

def weight_variable(inputn, outputn):
    return tf.Variable(tf.truncated_normal([inputn, outputn],stddev=0.1))

def bias_variable(outputn):
    return tf.Variable(tf.constant(0.1, shape=[outputn]))

def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
    with tf.name_scope("stddev"):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    
    tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, inputn, outputn, name, act=tf.nn.softmax):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            weights = weight_variable(inputn, outputn)
            variable_summaries(weights)

        with tf.name_scope('biases'):
            biases = bias_variable(outputn)
            variable_summaries(biases)

        with tf.name_scope('linear_compute'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('linear', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)

        # 返回激励层的最终输出
        return activations


L0 = nn_layer(x, travel_data.CITY_COUNT, 128, 'LCONV', act=tf.nn.tanh)
L1 = nn_layer(L0, 128, 128, "L1", act=tf.nn.relu6)
LO = nn_layer(L1, 128, travel_data.CITY_COUNT, "LOUT", act=tf.nn.softmax)

with tf.name_scope("predict"):
    y = LO
with tf.name_scope("loss"):
    cross_entropy = -tf.reduce_sum(y_ans*tf.log(y))
    tf.summary.scalar("cross_entropy",cross_entropy)
with tf.name_scope("accuracy"):
    hit_rate = tf.placeholder(tf.float32)
    tf.summary.scalar("hit_rate",hit_rate)
with tf.name_scope("train"):
    train_step = tf.train.AdagradOptimizer(0.01).minimize(cross_entropy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("./summaries/train")

init = tf.initialize_all_variables()
sess = tf.Session()
sess.as_default()
sess.run(init)
train_writer.add_graph(sess.graph)

saver = tf.train.Saver()
if os.path.exists("./save/{}.save.index".format(SAVENAME)):
    saver.restore(sess, "./save/{}.save".format(SAVENAME))
    restored=True
else:
    restored=False
with tf.device("/gpu:0"):
    def save():
        saver.save(sess, "./save/{}.save".format(SAVENAME))
    def train(train_times):
        prog = progressbar.ProgressBar(max_value=train_times*travel_data.BATCH_SIZE, widgets=["Train: " , progressbar.Percentage(), progressbar.Bar(), progressbar.AdaptiveETA(),"|",progressbar.AdaptiveTransferSpeed(unit = "smpl")])
        for i in range(train_times):
            batch_x, batch_y = travel_data.feed_data()
            if i%10==0:
                test_x, test_y = travel_data.feed_test_data()
                results = sess.run(y, feed_dict = {x: test_x})
                ans = test_y
                hr = []
                for p in range(len(ans)):
                    s_hit=0
                    s_total=0
                    res = {x:results[p][x] for x in range(travel_data.CITY_COUNT)}
                    res = sorted(res.items(), key=lambda x:x[1], reverse=True)[:10]
                    res = [item[0] for item in res]
                    for j in range(len(ans[p])):
                        if ans[p][j]==1:
                            s_total+=1
                            if j in res:
                                s_hit+=1
                    hr.append(s_hit/s_total)
                hr = sum(hr)/len(hr)
                train_writer.add_summary(sess.run(merged, feed_dict={x: batch_x, y_ans: batch_y, hit_rate: hr}), i)
            if i%100==0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                sess.run(train_step, feed_dict={x: batch_x, y_ans: batch_y}, options=run_options, run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%d' % i)
            else:
                sess.run(train_step, feed_dict={x: batch_x, y_ans: batch_y})
            prog.update(i*travel_data.BATCH_SIZE)
        prog.finish()
        # save()
    if not restored:
        train(10000)
    res = sess.run(y, feed_dict = {x: [travel_data.create_layer_from_city_list(["瑞金", "井冈山", "南京"])]})[0]
    # res = tf.subtract(res, travel_data.CITY_PROB).eval(session=sess)
    res = {travel_data.city_id_to_name(x):res[x] for x in range(travel_data.CITY_COUNT)}
    print(res['遵义'])
    with open("output.csv", mode="w") as f:
        for item in res.items():
            f.write(str(item[0]))
            f.write(",")
            f.write(str(travel_data.CITY_PROB[travel_data.CITY_ENUM[item[0]]]))
            f.write(",")
            f.write(str(item[1]))
            f.write("\n")