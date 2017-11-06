import argparse
import tensorflow as tf
import os
import time
from datetime import timedelta
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data', help='MNIST dataset location')
parser.add_argument('--batch_size', type=int, default=10,
                    help='Batch size')
parser.add_argument('--num_batches', type=int, default=1000,
                    help='Number of batches to train')
args = parser.parse_args()

## Unpooling

def max_pool_with_argmax(net, stride):
  """
  Tensorflow default implementation does not provide gradient operation on max_pool_with_argmax
  Therefore, we use max_pool_with_argmax to extract mask and
  plain max_pool for, eeem... max_pooling.
  """
  with tf.name_scope('MaxPoolArgMax'):
      # max_pool_with_argmax is only supported on GPU
      _, mask = tf.nn.max_pool_with_argmax(
        net,
          ksize=[1, stride, stride, 1],
          strides=[1, stride, stride, 1],
        padding='SAME')
      mask = tf.stop_gradient(mask)
      net = tf.layers.max_pooling2d(net, [stride, stride], stride)
      return net, mask

def unpool_cpu(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(scope):
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                          shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * ksize[1], set_input_shape[2] * ksize[2], set_input_shape[3]]
        ret.set_shape(set_output_shape)
        return ret

# it is way slower than the cpu version
def unpool_gpu(pool, ind, ksize=(1, 2, 2, 1), scope='unpool'):
    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices (produced by tf.nn.max_pool_with_argmax)
           ksize:     ksize is the same as for the pool
       Return:
           unpooled:    unpooling tensor
    """
    # not sure why, but I still have to manually change pooled_shape[0] to -1
    # OOM, could be caused by using -1
    with tf.variable_scope(scope):
        pooled_shape = pool.get_shape().as_list()
        batch_size = -1
        if pooled_shape[0] is not None:
          batch_size = pooled_shape[0]
        flatten_ind = tf.reshape(ind, (batch_size, pooled_shape[1] * pooled_shape[2] * pooled_shape[3]))
        # sparse indices to dense ones_like matrics
        print('debug:')
        print(flatten_ind, pooled_shape[1] * ksize[1] * pooled_shape[2] * ksize[2] * pooled_shape[3])
        one_hot_ind = tf.one_hot(flatten_ind,  pooled_shape[1] * ksize[1] * pooled_shape[2] * ksize[2] * pooled_shape[3], on_value=1., off_value=0., axis=-1)
        one_hot_ind = tf.reduce_sum(one_hot_ind, axis=1)
        one_like_mask = tf.reshape(one_hot_ind, (batch_size, pooled_shape[1] * ksize[1], pooled_shape[2] * ksize[2], pooled_shape[3]))
        # resize input array to the output size by nearest neighbor
        img = tf.image.resize_nearest_neighbor(pool, [pooled_shape[1] * ksize[1], pooled_shape[2] * ksize[2]])
        unpooled = tf.multiply(img, tf.cast(one_like_mask, img.dtype))
        return unpooled
    
## Model

def mnist_model(input):
    # Q: how should I handle relu?
    max_pool_stride = 2
    mask_stack = []

    # conv1
    conv1_n = 16
    conv1_kernel = [3, 3]
    hidden = tf.layers.conv2d(input, conv1_n, conv1_kernel, activation = tf.nn.relu)
    # conv2
    conv2_n = 32
    conv2_kernel = [3, 3]
    hidden = tf.layers.conv2d(hidden, conv2_n, conv2_kernel, activation = tf.nn.relu)
    hidden, mask = max_pool_with_argmax(hidden, max_pool_stride)
    mask_stack.append(mask)

    # deconv2
    hidden = unpool_cpu(hidden, mask_stack.pop())
    hidden = tf.layers.conv2d_transpose(hidden, conv1_n, conv2_kernel)
    # deconv1
    output = tf.layers.conv2d_transpose(hidden, 1, conv1_kernel)

    return output


# Read in MNIST data

mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
X_train = mnist.train.images.reshape((-1, 28, 28, 1))

def batch_generator(X_train, batch_size):
  while True:
    X = shuffle(X_train)
    offset = 0
    while offset + batch_size <= len(X):
      yield X[offset:offset + batch_size]
      offset += batch_size
            
# Training

input = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')
output = mnist_model(input)
loss = tf.reduce_mean(tf.squared_difference(output, input))
train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  
  tf_board = os.path.join('/tmp/autoencoder')
  writer = tf.summary.FileWriter(os.path.join(tf_board, str(int(time.time()))))
  writer.add_graph(sess.graph)
  tf.summary.scalar('losses/loss', loss, collections=['scalar'])
  tf.summary.image('source/image_pair',
                   tf.concat((input, output), axis=2), collections=['image'])
  scalar_summary_op = tf.summary.merge_all('scalar')
  image_summary_op = tf.summary.merge_all('image')

  batch = batch_generator(X_train, args.batch_size)
  start_time = time.time()
  for i in range(args.num_batches):
    next_batch = next(batch)

    if i % 100 == 0:
      s = sess.run(image_summary_op,
                         feed_dict={input: next_batch})
      writer.add_summary(s, i)
      
    _, s, loss_value = sess.run([train_op, scalar_summary_op, loss],
                                feed_dict={input: next_batch})
    writer.add_summary(s, i)
    print('iter:', i, loss_value)
    
  end_time = time.time()
  time_diff = end_time - start_time
  print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))


