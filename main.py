# comment out the first two lines if you want to suppress figure plots on ipython
import matplotlib
matplotlib.use('Agg')
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pix2pix import Pix2pix
from preprocessing import load_and_preprocess_dataset

# create directories to save the generated images and the trained model
if not os.path.exists('images/'):
    os.makedirs('images/')
if not os.path.exists('trained-model/'):
    os.makedirs('trained-model/')

iters = 200*400 # taken from pix2pix paper §5.2
batch_size = 1 # taken from pix2pix paper §5.2

data_dir = 'lfw-deepfunneled/'
# get the original data and create the noisy data for training
# choose between 3 types of noise; pixelated, gaussian, salt and pepper, and speckle
B, A = load_and_preprocess_dataset(data_dir, 'pix')

# B is the clean image set, A is the noisy image set

def destandardize(X):
    '''
    takes the float image input and turns it to uint8 from scale 0 to 255 
    to convert image from BGR to RGB to save plotted figures using matplotlib
    '''
    X = X*(255)
    X = np.array(X, dtype = np.uint8)
    X  = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
    return X

with tf.device('/gpu:0'):
    model = Pix2pix(250, 250, ichan=3, ochan=3)

# create tensorboard files to visualize the model if wanted
sess = tf.Session()
writer = tf.summary.FileWriter('tensorboard-model')
writer.add_graph(sess.graph)

# initialize saver to further save the trained model
saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(iters):
        a = cv2.resize(A[step % A.shape[0]], (250,250))
        a = np.expand_dims(a, axis=0)

        b = 2. * np.expand_dims(B[step % B.shape[0]], axis=0) - 1. # normalize because generator use tanh activation in its output layer

        gloss_curr, dloss_curr = model.train_step(sess, a, a, b)
        print('Step %d: G loss: %f | D loss: %f' % (step, gloss_curr, dloss_curr))

        if step % 250 == 0:
            fig = plt.figure()
            fig.set_size_inches(10, 10)
            fig.subplots_adjust(left=0, bottom=0,
                                   right=1, top=1, wspace=0, hspace=0.1)
            p = np.random.permutation(B.shape[0])
            for i in range(0, 81, 3):
                # Plot 3 images: First is the architectural label, second the generator output, third the ground truth
                fig.add_subplot(9, 9, i + 1)
                plt.imshow(destandardize(A[p[i // 3]]))
                plt.axis('off')
                fig.add_subplot(9, 9, i + 2)
                g_in = A[p[i // 3]]
                gen = (model.sample_generator(sess, np.expand_dims(g_in, axis=0), is_training=True)[0] + 1.) / 2.
                plt.imshow(destandardize(gen))
                plt.axis('off')
                fig.add_subplot(9, 9, i +3)
                plt.imshow(destandardize(B[p[i // 3]]))
                plt.axis('off')
            plt.savefig('images/iter_%d.jpg' % step, dpi=fig.dpi)
            plt.close()
        if step % 3000 == 0:
            # Save the model
            save_path = saver.save(sess, "trained-model/model.ckpt")
            print("Model saved in file: %s" % save_path)
