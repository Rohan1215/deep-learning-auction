from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf


class Trainer(object):

    def __init__(self, config, mode, net, clip_op_lambda):
        self.config = config
        self.mode = mode
        tf.compat.v1.disable_eager_execution()
        
        # Create output-dir
        if not os.path.exists(self.config.dir_name): 
            os.makedirs(self.config.dir_name)

        if self.mode == "train":
            log_suffix = '_' + str(self.config.train.restore_iter) if self.config.train.restore_iter > 0 else ''
            self.log_fname = os.path.join(self.config.dir_name, 'train' + log_suffix + '.txt')
        else:
            log_suffix = "_iter_" + str(self.config.test.restore_iter) + "_m_" + str(self.config.test.num_misreports) + "_gd_" + str(self.config.test.gd_iter)
            self.log_fname = os.path.join(self.config.dir_name, "test" + log_suffix + ".txt")
            
        # Set Seeds for reproducibility
        np.random.seed(self.config[self.mode].seed)
        tf.compat.v1.set_random_seed(self.config[self.mode].seed)
        
        # Init Logger
        self.init_logger()

        # Init Net
        self.net = net
        
        ## Clip Op
        self.clip_op_lambda = clip_op_lambda
        
        # Init TF-graph
        self.init_graph()
        
    def get_clip_op(self, adv_var):
        self.clip_op =  self.clip_op_lambda(adv_var)
        #tf.compat.v1.assign(adv_var, tf.compat.v1.clip_by_value(adv_var, 0.0, 1.0))
        

    def init_logger(self):


        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        handler = logging.FileHandler(self.log_fname, 'w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        self.logger = logger

    def compute_rev(self, pay):
        """ Given payment (pay), computes revenue
            Input params:
                pay: [num_batches, num_agents]
            Output params:
                revenue: scalar
        """
        return tf.compat.v1.reduce_mean(tf.compat.v1.reduce_sum(pay, axis=-1))

    def compute_utility(self, x, alloc, pay):
        """ Given input valuation (x), payment (pay) and allocation (alloc), computes utility
            Input params:
                x: [num_batches, num_agents, num_items]
                a: [num_batches, num_agents, num_items]
                p: [num_batches, num_agents]
            Output params:
                utility: [num_batches, num_agents]
        """
        return tf.compat.v1.reduce_sum(tf.compat.v1.multiply(alloc, x), axis=-1) - pay


    def get_misreports(self, x, adv_var, adv_shape):

        num_misreports = adv_shape[1]
        adv = tf.compat.v1.tile(tf.compat.v1.expand_dims(adv_var, 0), [self.config.num_agents, 1, 1, 1, 1])
        x_mis = tf.compat.v1.tile(x, [self.config.num_agents * num_misreports, 1, 1])
        x_r = tf.compat.v1.reshape(x_mis, adv_shape)
        y = x_r * (1 - self.adv_mask) + adv * self.adv_mask
        misreports = tf.compat.v1.reshape(y, [-1, self.config.num_agents, self.config.num_items])
        return x_mis, misreports

    def init_graph(self):
       
        x_shape = [self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
        adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
        adv_var_shape = [ self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
        u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]

        # Placeholders
        self.x = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=x_shape, name='x')
        self.adv_init = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=adv_var_shape, name='adv_init')
        
        self.adv_mask = np.zeros(adv_shape)
        self.adv_mask[np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents), :] = 1.0
        
        self.u_mask = np.zeros(u_shape)
        self.u_mask[np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents)] = 1.0
        
        with tf.compat.v1.variable_scope('adv_var'):
            self.adv_var = tf.compat.v1.get_variable('adv_var', shape = adv_var_shape, dtype = tf.compat.v1.float32)
            
            
        # Misreports
        x_mis, self.misreports = self.get_misreports(self.x, self.adv_var, adv_shape)
        
        # Get mechanism for true valuation: Allocation and Payment
        self.alloc, self.pay = self.net.inference(self.x)
        
        # Get mechanism for misreports: Allocation and Payment
        a_mis, p_mis = self.net.inference(self.misreports)
        
        # Utility
        utility = self.compute_utility(self.x, self.alloc, self.pay)
        utility_mis = self.compute_utility(x_mis, a_mis, p_mis)
        
        # Regret Computation
        u_mis = tf.compat.v1.reshape(utility_mis, u_shape) * self.u_mask
        utility_true = tf.compat.v1.tile(utility, [self.config.num_agents * self.config[self.mode].num_misreports, 1])
        excess_from_utility = tf.compat.v1.nn.relu(tf.compat.v1.reshape(utility_mis - utility_true, u_shape) * self.u_mask)
        rgt = tf.compat.v1.reduce_mean(tf.compat.v1.reduce_max(excess_from_utility, axis=(1, 3)), axis=1)
    
        #Metrics
        revenue = self.compute_rev(self.pay)
        rgt_mean = tf.compat.v1.reduce_mean(rgt)
        irp_mean = tf.compat.v1.reduce_mean(tf.compat.v1.nn.relu(-utility))

        # Variable Lists
        alloc_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='alloc')
        pay_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='pay')
        var_list = alloc_vars + pay_vars


        
        if self.mode is "train":

            w_rgt_init_val = 0.0 if "w_rgt_init_val" not in self.config.train else self.config.train.w_rgt_init_val

            with tf.compat.v1.variable_scope('lag_var'):
                self.w_rgt = tf.compat.v1.Variable(np.ones(self.config.num_agents).astype(np.float32) * w_rgt_init_val, 'w_rgt')

            update_rate = tf.compat.v1.Variable(self.config.train.update_rate, trainable = False)
            self.increment_update_rate = update_rate.assign(update_rate + self.config.train.up_op_add)
      
            # Loss Functions
            rgt_penalty = update_rate * tf.compat.v1.reduce_sum(tf.compat.v1.square(rgt)) / 2.0        
            lag_loss = tf.compat.v1.reduce_sum(self.w_rgt * rgt)
        

            loss_1 = -revenue + rgt_penalty + lag_loss
            loss_2 = -tf.compat.v1.reduce_sum(u_mis)
            loss_3 = -lag_loss

            reg_losses = tf.compat.v1.get_collection('reg_losses')
            if len(reg_losses) > 0:
                reg_loss_mean = tf.compat.v1.reduce_mean(reg_losses)
                loss_1 = loss_1 + reg_loss_mean

             
            learning_rate = tf.compat.v1.Variable(self.config.train.learning_rate, trainable = False)
        
            # Optimizer
            opt_1 = tf.compat.v1.train.AdamOptimizer(learning_rate)
            opt_2 = tf.compat.v1.train.AdamOptimizer(self.config.train.gd_lr)
            opt_3 = tf.compat.v1.train.GradientDescentOptimizer(update_rate)


            # Train ops
            self.train_op  = opt_1.minimize(loss_1, var_list = var_list)
            self.train_mis_step = opt_2.minimize(loss_2, var_list = [self.adv_var])
            self.lagrange_update    = opt_3.minimize(loss_3, var_list = [self.w_rgt])
            
            # Val ops
            val_mis_opt = tf.compat.v1.train.AdamOptimizer(self.config.val.gd_lr)
            self.val_mis_step = val_mis_opt.minimize(loss_2, var_list = [self.adv_var])       

            # Reset ops
            self.reset_train_mis_opt = tf.compat.v1.variables_initializer(opt_2.variables()) 
            self.reset_val_mis_opt = tf.compat.v1.variables_initializer(val_mis_opt.variables())

            # Metrics
            self.metrics = [revenue, rgt_mean, rgt_penalty, lag_loss, loss_1, tf.compat.v1.reduce_mean(self.w_rgt), update_rate]
            self.metric_names = ["Revenue", "Regret", "Reg_Loss", "Lag_Loss", "Net_Loss", "w_rgt_mean", "update_rate"]
            
            #Summary
            tf.compat.v1.summary.scalar('revenue', revenue)
            tf.compat.v1.summary.scalar('regret', rgt_mean)
            tf.compat.v1.summary.scalar('reg_loss', rgt_penalty)
            tf.compat.v1.summary.scalar('lag_loss', lag_loss)
            tf.compat.v1.summary.scalar('net_loss', loss_1)
            tf.compat.v1.summary.scalar('w_rgt_mean', tf.compat.v1.reduce_mean(self.w_rgt))
            if len(reg_losses) > 0: tf.compat.v1.summary.scalar('reg_loss', reg_loss_mean)

            self.merged = tf.compat.v1.summary.merge_all()
            self.saver = tf.compat.v1.train.Saver(max_to_keep = self.config.train.max_to_keep)
        
        elif self.mode is "test":

            loss = -tf.compat.v1.reduce_sum(u_mis)
            test_mis_opt = tf.compat.v1.train.AdamOptimizer(self.config.test.gd_lr)
            self.test_mis_step = test_mis_opt.minimize(loss, var_list = [self.adv_var])
            self.reset_test_mis_opt = tf.compat.v1.variables_initializer(test_mis_opt.variables())

            # Metrics
            welfare = tf.compat.v1.reduce_mean(tf.compat.v1.reduce_sum(self.alloc * self.x, axis = (1,2)))
            self.metrics = [revenue, rgt_mean, irp_mean]
            self.metric_names = ["Revenue", "Regret", "IRP"]
            self.saver = tf.compat.v1.train.Saver(var_list = var_list)
            

        # Helper ops post GD steps
        self.assign_op = tf.compat.v1.assign(self.adv_var, self.adv_init)
        self.get_clip_op(self.adv_var)
        
    def train(self, generator):
        """
        Runs training
        """
        
        self.train_gen, self.val_gen = generator
        
        iter = self.config.train.restore_iter
        sess = tf.compat.v1.InteractiveSession()
        tf.compat.v1.global_variables_initializer().run()
        train_writer = tf.compat.v1.summary.FileWriter(self.config.dir_name, sess.graph)
        
        if iter > 0:
            model_path = os.path.join(self.config.dir_name, 'model-' + str(iter))
            self.saver.restore(sess, model_path)

        if iter == 0:
            self.train_gen.save_data(0)
            self.saver.save(sess, os.path.join(self.config.dir_name,'model'), global_step = iter)

        time_elapsed = 0.0
        while iter < (self.config.train.max_iter):
             
            # Get a mini-batch
            X, ADV, perm = next(self.train_gen.gen_func)
                
            if iter == 0: sess.run(self.lagrange_update, feed_dict = {self.x : X})
 

            tic = time.time()    
            
            # Get Best Mis-report
            sess.run(self.assign_op, feed_dict = {self.adv_init: ADV})                                        
            for _ in range(self.config.train.gd_iter):
                sess.run(self.train_mis_step, feed_dict = {self.x: X})
                sess.run(self.clip_op)
            sess.run(self.reset_train_mis_opt)

            if self.config.train.data is "fixed" and self.config.train.adv_reuse:
                self.train_gen.update_adv(perm, sess.run(self.adv_var))

            # Update network params
            sess.run(self.train_op, feed_dict = {self.x: X})
                
            if iter==0:
                summary = sess.run(self.merged, feed_dict = {self.x: X})
                train_writer.add_summary(summary, iter) 

            iter += 1

            # Run Lagrange Update
            if iter % self.config.train.update_frequency == 0:
                sess.run(self.lagrange_update, feed_dict = {self.x:X})
                

            if iter % self.config.train.up_op_frequency == 0:
                sess.run(self.increment_update_rate)

            toc = time.time()
            time_elapsed += (toc - tic)
                        
            if ((iter % self.config.train.save_iter) == 0) or (iter == self.config.train.max_iter): 
                self.saver.save(sess, os.path.join(self.config.dir_name,'model'), global_step = iter) 
                self.train_gen.save_data(iter)

            if (iter % self.config.train.print_iter) == 0:
                # Train Set Stats
                summary = sess.run(self.merged, feed_dict = {self.x: X})
                train_writer.add_summary(summary, iter)
                metric_vals = sess.run(self.metrics, feed_dict = {self.x: X})
                fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_vals) for item in tup ])
                log_str = "TRAIN-BATCH Iter: %d, t = %.4f"%(iter, time_elapsed) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
                self.logger.info(log_str)

            if (iter % self.config.val.print_iter) == 0:
                #Validation Set Stats
                metric_tot = np.zeros(len(self.metric_names))         
                for _ in range(self.config.val.num_batches):
                    X, ADV, _ = next(self.val_gen.gen_func) 
                    sess.run(self.assign_op, feed_dict = {self.adv_init: ADV})               
                    for k in range(self.config.val.gd_iter):
                        sess.run(self.val_mis_step, feed_dict = {self.x: X})
                        sess.run(self.clip_op)
                    sess.run(self.reset_val_mis_opt)                                   
                    metric_vals = sess.run(self.metrics, feed_dict = {self.x: X})
                    metric_tot += metric_vals
                    
                metric_tot = metric_tot/self.config.val.num_batches
                fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_tot) for item in tup ])
                log_str = "VAL-%d"%(iter) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
                self.logger.info(log_str)

    def test(self, generator):
        """
        Runs test
        """
        
        # Init generators
        self.test_gen = generator

        iter = self.config.test.restore_iter
        sess = tf.compat.v1.InteractiveSession()
        tf.compat.v1.global_variables_initializer().run()

        model_path = os.path.join(self.config.dir_name,'model-' + str(iter))
        self.saver.restore(sess, model_path)

        #Test-set Stats
        time_elapsed = 0
            
        metric_tot = np.zeros(len(self.metric_names))

        if self.config.test.save_output:
            assert(hasattr(generator, "X")), "save_output option only allowed when config.test.data = Fixed or when X is passed as an argument to the generator"
            alloc_tst = np.zeros(self.test_gen.X.shape)
            pay_tst = np.zeros(self.test_gen.X.shape[:-1])
                    
        for i in range(self.config.test.num_batches):
            tic = time.time()
            X, ADV, perm = next(self.test_gen.gen_func)
            sess.run(self.assign_op, feed_dict = {self.adv_init: ADV})
                    
            for k in range(self.config.test.gd_iter):
                sess.run(self.test_mis_step, feed_dict = {self.x: X})
                sess.run(self.clip_op)

            sess.run(self.reset_test_mis_opt)        
                
            metric_vals = sess.run(self.metrics, feed_dict = {self.x: X})
            
            if self.config.test.save_output:
                A, P = sess.run([self.alloc, self.pay], feed_dict = {self.x:X})
                alloc_tst[perm, :, :] = A
                pay_tst[perm, :] = P
                    
            metric_tot += metric_vals
            toc = time.time()
            time_elapsed += (toc - tic)

            fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_vals) for item in tup ])
            log_str = "TEST BATCH-%d: t = %.4f"%(i, time_elapsed) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
            self.logger.info(log_str)
        
        metric_tot = metric_tot/self.config.test.num_batches
        fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_tot) for item in tup ])
        log_str = "TEST ALL-%d: t = %.4f"%(iter, time_elapsed) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
        self.logger.info(log_str)
            
        if self.config.test.save_output:
            np.save(os.path.join(self.config.dir_name, 'alloc_tst_' + str(iter)), alloc_tst)
            np.save(os.path.join(self.config.dir_name, 'pay_tst_' + str(iter)), pay_tst)
            
        
        
        
