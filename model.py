import time

from utils import *
import random

def cnn_par(input, is_training=True, output_channels=3):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 192, 3, padding='same', activation=tf.nn.relu)
    for layers in xrange(2, 15 + 1):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 192, 3, padding='same', name='conv%d' % layers)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block16'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    #input0 = tf.expand_dims(input[:,:,:,0:3], axis=3, name='expandDims')
    tf.summary.image('input0', input[:,:,:,0:3], 1) ###!!! to be removed
    return tf.add(output, input[:,:,:,0:3])


class imdualenh(object):
    def __init__(self, sess, input_c_dim=3, batch_size=128, PARALLAX=65):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.parallax = PARALLAX
        
        # build model
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='gt_p')
        tf.summary.image('gt_patch', self.Y_, 3)
        self.X  = tf.placeholder(tf.float32, [None, None, None, 3*(self.parallax+1)], name='ll_lr_p')
        #tf.summary.image('ll_patch', self.X , 3)
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.Y = cnn_par(self.X, is_training=self.is_training)
        tf.summary.image('nl_patch', self.Y , 3)
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")
        sys.stdout.flush()

    def evaluate(self, iter_num, test_data_YL, test_data_XL, test_data_XR,
                 sample_dir, summary_merged, summary_writer):
        print("[*] Evaluating...")
        sys.stdout.flush()
        psnr_sum = 0
        for idx in xrange(len(test_data_YL)):
            _, im_h, im_w, ch = test_data_YL[idx].shape
            assert ch == 3
            Y = test_data_YL[idx][:,:,self.parallax:,:].astype(np.float32) / 255.0
            X = np.zeros((1,im_h,im_w-self.parallax,3*(self.parallax+1)))
            X[:,:,:,0:3] = test_data_XL[idx][:,:,self.parallax:,:].astype(np.float32) / 255.0
            pp = 0
            for p in range(3, 3*(self.parallax+1), 3):
                X[:,:,:,p:p+3] = test_data_XR[idx][:,:,self.parallax-pp:im_w-pp,:].astype(np.float32) / 255.0
                pp += 1
            nl_image, x_tensor, psnr_summary = self.sess.run(
                       [self.Y, self.X, summary_merged],
                        feed_dict={self.Y_: Y, self.X: X,
                        self.is_training: False})
            summary_writer.add_summary(psnr_summary, iter_num)
            groundtruth = np.clip(test_data_YL[idx][:,:,self.parallax:,:], 0, 255).astype('uint8').squeeze()
            darkimage = np.clip(  test_data_XL[idx][:,:,self.parallax:,:], 0, 255).astype('uint8').squeeze()
            outputimage = np.clip(255 * nl_image, 0, 255).astype('uint8').squeeze()
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx + 1, psnr))
            sys.stdout.flush()
            psnr_sum += psnr
            save_images(os.path.join(sample_dir, 'test%d_%d.png' % (idx + 1, iter_num)),
                        groundtruth, darkimage, outputimage)
        avg_psnr = psnr_sum / len(test_data_YL)
        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)
        sys.stdout.flush()

    #def denoise(self, data_gt, data_in):
    #    output_clean_image, noisy_image, psnr = self.sess.run([self.Y, self.X, self.eva_psnr],
    #            feed_dict={self.Y_:data_gt, self.X:data_in, self.is_training: False})
    #    return output_clean_image, noisy_image, psnr

    def train(self, data, eval_data_YL, eval_data_XL, eval_data_XR, 
              batch_size, ckpt_dir, epoch, lr, use_gpu, sample_dir,
              eval_every_epoch=2):
        # assert data range is between 0 and 1
        data_num = data["X_tr"].shape[0]
        numBatch = int(data_num / batch_size)
        # load pretrained model
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
            sys.stdout.flush()
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")
            sys.stdout.flush()
        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        if use_gpu == 1:
            writer = tf.summary.FileWriter('./logs-gpu', self.sess.graph)
        else:
            writer = tf.summary.FileWriter('./logs-cpu', self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        sys.stdout.flush()
        start_time = time.time()
        self.evaluate(iter_num, eval_data_YL, eval_data_XL, eval_data_XR,
                      sample_dir=sample_dir, summary_merged=summary_psnr,
                      summary_writer=writer)
        for epoch in xrange(start_epoch, epoch):
            blist = random.sample(range(0, numBatch), numBatch)
            for batch_id in xrange(start_step, numBatch):
                i_s = blist[batch_id] * batch_size
                i_e = (blist[batch_id] + 1 ) * batch_size
                batch_inputs = data["X_tr"][i_s:i_e, ...]
                batch_gt     = data["Y_tr"][i_s:i_e, ...]
                batch_inputs = batch_inputs.astype(np.float32) / 255.0 # normalize the data to 0-1
                batch_gt     = batch_gt.astype(np.float32) / 255.0
                _, loss, summary = self.sess.run([self.train_op, self.loss, merged],
                        feed_dict={self.Y_: batch_gt, self.X:batch_inputs, self.lr: lr[epoch], self.is_training: True})
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                sys.stdout.flush()
                iter_num += 1
                writer.add_summary(summary, iter_num)
            if np.mod(epoch + 1, eval_every_epoch) == 0:
                self.evaluate(iter_num, eval_data_YL, eval_data_XL, eval_data_XR,
                              sample_dir=sample_dir, summary_merged=summary_psnr,
                              summary_writer=writer)  # eval_data value range is 0-255
                self.save(iter_num, ckpt_dir)
        print("[*] Finish training.")
        sys.stdout.flush()

    def save(self, iter_num, ckpt_dir, model_name='cnn_par'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        sys.stdout.flush()
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        sys.stdout.flush()
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, test_files, ckpt_dir, save_dir):
        """Test CNN_PAR"""
        # init variables
        tf.initialize_all_variables().run()
        assert len(test_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        sys.stdout.flush()
        psnr_sum = 0
        print("[*] " + 'noise level: ' + str(self.sigma) + " start testing...")
        sys.stdout.flush()
        for idx in xrange(len(test_files)):
            clean_image = load_images(test_files[idx]).astype(np.float32) / 255.0
            output_clean_image, noisy_image = self.sess.run([self.Y, self.X],
                                                            feed_dict={self.Y_: clean_image, self.is_training: False})
            groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx, psnr))
            sys.stdout.flush()
            psnr_sum += psnr
            save_images(os.path.join(save_dir, 'noisy%d.png' % idx), noisyimage)
            save_images(os.path.join(save_dir, 'denoised%d.png' % idx), outputimage)
        avg_psnr = psnr_sum / len(test_files)
        print("--- Average PSNR %.2f ---" % avg_psnr)
        sys.stdout.flush()
