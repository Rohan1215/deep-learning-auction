from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from baseline.baseline import OptRevOneItem
import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

class MyersonNet():
    '''
    MyersonNet: A Neural Network for computing optimal single item auctions
    '''
    def __init__(self, args, train_data, test_data):
        self.args = args
        self.sess = tf.compat.v1.InteractiveSession()
        self.train_data = train_data
        self.test_data = test_data
        self.nn_build()
        
    def nn_build(self):
        num_func      = self.args.num_linear_func
        num_max_units = self.args.num_max_units
        num_agent     = self.args.num_agent
        
        self.x  = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, num_agent])
        self.lr = tf.compat.v1.placeholder(tf.compat.v1.float32)
        
        ######
        # Initialization of the weights
        ######
        np.random.seed(self.args.seed_val)
        self.w_encode1_init = np.random.normal(size = (num_max_units, num_func, num_agent))
        self.w_encode2_init = -np.random.rand(num_max_units, num_func, num_agent) * 5.0
                            
        # Linear weights
        self.w_encode1 = tf.compat.v1.Variable(tf.compat.v1.constant(np.float32(self.w_encode1_init)))
        # Biase weights
        self.w_encode2 = tf.compat.v1.Variable(tf.compat.v1.constant(np.float32(self.w_encode2_init)))
        
        self.revenue, self.alloc, self.vv = self.nn_eval('train')
        self.revenue_test, self.alloc_test, self.vv_test = self.nn_eval('test')
        
    def nn_eval(self, str): 
        num_func      = self.args.num_linear_func
        num_max_units = self.args.num_max_units
        num_agent     = self.args.num_agent
        
        batch_size = tf.compat.v1.shape(self.x)[0]
        
        append_dummy_mat = tf.compat.v1.constant(
                            np.float32(np.append(np.identity(num_agent),
                            np.zeros([num_agent, 1]), 1)))
        ######                    
        # Compute the input of the SPA with reserve zero unit
        ######
        w_encode1_copy = tf.compat.v1.reshape(tf.compat.v1.tile(
                            self.w_encode1, [batch_size, 1, 1]),
                            [batch_size, num_max_units, num_func, num_agent])
        w_encode2_copy = tf.compat.v1.reshape(tf.compat.v1.tile(
                            self.w_encode2, [batch_size, 1, 1]),
                            [batch_size, num_max_units, num_func, num_agent])
        
        x_copy = tf.compat.v1.reshape(tf.compat.v1.tile(self.x, [1, num_func*num_max_units]),
                            [batch_size, num_max_units, num_func, num_agent])
                    
        vv_max_units = tf.compat.v1.reduce_max(tf.compat.v1.multiply(
                        x_copy, tf.compat.v1.exp(w_encode1_copy)) + w_encode2_copy, [2])
        # Compute virtual value
        vv = tf.compat.v1.reduce_min(vv_max_units, [1])
        
        #####
        # Run SPA unit with reserve price 0
        #####
        
        # Compute allocation rate in SPA unit
        if str == 'train':
            w_a = tf.compat.v1.constant(np.float32(np.identity(num_agent+1) * 1000))
            a_dummy = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(tf.compat.v1.matmul(vv, append_dummy_mat), w_a))
        if str == 'test':
            win_agent = tf.compat.v1.argmax(tf.compat.v1.matmul(vv, append_dummy_mat), 1) # The index of agent who win the item
            a_dummy = tf.compat.v1.one_hot(win_agent, num_agent+1)
        
        a = tf.compat.v1.slice(a_dummy, [0, 0], [batch_size, num_agent])
        
        # Compute payment in SPA unit: weighted max of inputs
        w_p = tf.compat.v1.constant(np.float32(np.ones((num_agent, num_agent)) - np.identity(num_agent)))
        spa_tensor1 = tf.compat.v1.reshape(tf.compat.v1.tile(tf.compat.v1.reshape(vv, [-1]), [num_agent]),\
                        [num_agent, -1, num_agent])
        spa_tensor2 = tf.compat.v1.matmul(spa_tensor1, tf.compat.v1.matrix_diag(w_p))
        p_spa = tf.compat.v1.transpose(tf.compat.v1.reduce_max(spa_tensor2, reduction_indices=[2]))
        
        ## Decode the payment
        p_spa_copy = tf.compat.v1.reshape(tf.compat.v1.tile(p_spa, [1, num_func * num_max_units]),\
                        [batch_size, num_max_units, num_func, num_agent])
        p_max_units = tf.compat.v1.reduce_min(tf.compat.v1.multiply(p_spa_copy - w_encode2_copy,\
                        tf.compat.v1.reciprocal(tf.compat.v1.exp(w_encode1_copy))),\
                            reduction_indices=[2])
                            
        p = tf.compat.v1.reduce_max(p_max_units, reduction_indices = [1])
        
        # Compute the revenue
        revenue = tf.compat.v1.reduce_mean(tf.compat.v1.reduce_sum(tf.compat.v1.multiply(a, p),
                    reduction_indices=[1]))
                    
        return revenue, a, vv
       
    def nn_train(self):
        # Parse parameters
        sample_val = self.train_data
        
        num_func = self.args.num_linear_func
        num_max_units = self.args.num_max_units
        num_agent = self.args.num_agent
        batch_size = self.args.batch_size
        data_size = sample_val.shape[0]

        # Loss: 
        loss = - self.revenue

        # Choose gradient descent optimizer update step
        self.train_step = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(loss)

        # Initialize variables
        tf.compat.v1.global_variables_initializer().run()

        # Store weights when training
        num_recordings = int(np.round(self.args.num_iter/self.args.skip_iter)) + 1
        w_encode1_array = np.zeros((num_recordings, num_max_units, num_func, num_agent))
        w_encode2_array = np.zeros((num_recordings, num_max_units, num_func, num_agent))

        w_encode1_array[0,:,:,:] = self.w_encode1_init
        w_encode2_array[0,:,:,:] = self.w_encode2_init

        np.random.seed()
        # Iterate over self.args.num_iter iterations, executing update train_step 
        for i in range(self.args.num_iter):
                perm = np.random.choice(data_size, batch_size, replace=False)
                self.sess.run(self.train_step, 
                    feed_dict={self.x: sample_val[perm,:],
                        self.lr: self.args.learning_rate})  
                                
                if(i % self.args.skip_iter == 0):
                        ind = int(np.round(i/self.args.skip_iter)) + 1
                        w_encode1_array[ind,:,:,:], w_encode2_array[ind,:,:,:]\
                            = self.sess.run((self.w_encode1, self.w_encode2),
                                feed_dict = {self.x: sample_val[perm,:]})
                            
                if((i+1) % 10000 == 0):
                        print('Complete ' + str(i+1) + ' iterations')                        
        
        return w_encode1_array, w_encode2_array


    def nn_test(self, data, mechanism):
        # Parse parameters
        sample_val                       = data
        w_encode1_array, w_encode2_array = mechanism
        num_agent                        = self.args.num_agent
        data_size                        = sample_val.shape[0]
        
        win_index = OptRevOneItem(self.args, data).winner()
        
        a_error = tf.compat.v1.reduce_sum(tf.compat.v1.abs(self.alloc_test - win_index))/data_size/2.0

        num_recordings = w_encode1_array.shape[0]
        rev_array = np.zeros(num_recordings)
        alloc_error_array = np.zeros(num_recordings)
        vv_array = np.zeros((data_size, num_agent))

        for i in range(num_recordings):
            rev_array[i], alloc_error_array[i], vv_array =\
                self.sess.run((self.revenue_test, a_error, self.vv_test),
                    feed_dict={self.w_encode1: w_encode1_array[i,:,:,:],
                        self.w_encode2: w_encode2_array[i,:,:,:],\
                            self.x: sample_val})
                                   
        return (rev_array, alloc_error_array, vv_array)
    