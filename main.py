# Import modules
import numpy as np
import theano
from theano import In
from theano import tensor as T
from theano import function
from theano import shared

# Session variables
floatX = 'float32'

# Data I/O
data = open('yoursourcetext.txt', 'r').read()
chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# Vectorize data
chars_as_vectors = np.zeros((data_size,vocab_size)).astype(dtype=floatX)
for x in range(0,len(chars_as_vectors)):
    char = data[x]
    char_id = char_to_ix[char]
    chars_as_vectors[x,char_id] = 1

# Scalable LSTM module
class LSTM:
    def __init__(self,x_dim,h_dim,temperature = 1.0):
        # LSTM Block Right
        self.lstm_block_right = None
        self.lstm_block_front = None

        # learning rate
        self.learning_rate = np.float32(0.001)

        # dropout probability
        self.p_dropout = np.float32(0.3)

        # log inputs
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.temperature = temperature

        # construct matrices
        self.Wx = shared(2 * np.random.random((h_dim * 4,x_dim)).astype(dtype = floatX) - 1)
        self.Wh = shared(2 * np.random.random((h_dim * 4,h_dim)).astype(dtype = floatX) - 1)
        self.msWx = shared(np.zeros((h_dim * 4,x_dim)).astype(dtype = floatX))
        self.msWh = shared(np.zeros((h_dim * 4,h_dim)).astype(dtype = floatX))
        self.dWx = shared(np.zeros((h_dim * 4,x_dim)).astype(dtype = floatX))
        self.dWh = shared(np.zeros((h_dim * 4,h_dim)).astype(dtype = floatX))

        # construct vectors
        self.B = shared(2*np.ones((h_dim * 4,)).astype(dtype = floatX) - 1)
        self.msB = shared(np.zeros((h_dim * 4,)).astype(dtype = floatX))
        self.dB = shared(np.zeros((h_dim * 4,)).astype(dtype = floatX))

        # dropout vector
        self.dropout_vector = shared(np.random.choice([0, 1], size=(self.h_dim,), p=[self.p_dropout, 1 - self.p_dropout]).astype(dtype=floatX))

        # SYMBOLIC VARIABLES
        input_vector_one = T.vector("input_vector_one",dtype = floatX)
        input_vector_two = T.vector("input_vector_two",dtype = floatX)
        input_vector_three = T.vector("input_vector_three",dtype = floatX)
        input_vector_four = T.vector("input_vector_four",dtype = floatX)
        input_vector_five = T.vector("input_vector_five",dtype = floatX)
        input_vector_six = T.vector("input_vector_six",dtype = floatX)
        input_vector_seven = T.vector("input_vector_seven",dtype = floatX)
        input_vector_eight = T.vector("input_vector_eight",dtype = floatX)
        input_vector_nine = T.vector("input_vector_nine",dtype = floatX)
        input_vector_dc21 = T.vector("input_vector_dc21",dtype = floatX)
        input_vector_dc22 = T.vector("input_vector_dc22",dtype = floatX)
        input_vector_x_sigma = T.vector("input_vector_x_sigma",dtype = floatX)
        input_vector_h_sigma = T.vector("input_vector_h_sigma",dtype = floatX)
        input_vector_h = T.vector("input_vector_h",dtype = floatX)
        input_vector_y = T.vector("input_vector_y",dtype = floatX)
        input_matrix_Wx = T.matrix("input_matrix_Wx",dtype = floatX)
        input_matrix_Wh = T.matrix("input_matrix_Wh",dtype = floatX)
        input_scalar_one = T.scalar("input_scalar_one",dtype = floatX)
        input_scalar_x_dim = T.iscalar()
        input_scalar_h_dim = T.iscalar()
        input_scalar_learning_rate = T.scalar(dtype=floatX)

        # OPTIMIZED FUNCTIONS
        # f_prop dot - Edge
        f_prop_dot_Wx = T.dot(self.Wx,input_vector_one)
        self.f_prop_dotEdge = function([input_vector_one],f_prop_dot_Wx)

        # f_prop dot
        f_prop_dot_Wx = T.dot(self.Wx,input_vector_one)
        f_prop_dot_Wh = T.dot(self.Wh,input_vector_two)
        self.f_prop_dot = function([input_vector_one,input_vector_two],[f_prop_dot_Wx,f_prop_dot_Wh])


        # f_prop dot_dropout - Edge
        f_prop_dot_dropout_Wx = T.dot(self.Wx,input_vector_one * input_vector_two)
        self.f_prop_dot_dropoutEdge = function([input_vector_one,input_vector_two],f_prop_dot_dropout_Wx)

        # f_prop dot_dropout
        f_prop_dot_dropout_Wh = T.dot(self.Wh,input_vector_three)
        self.f_prop_dot_dropoutBODY = function([input_vector_one,input_vector_three,input_vector_two],[f_prop_dot_dropout_Wx,f_prop_dot_dropout_Wh])

        # f_prop pointwise - Edge
        f_prop_combined = input_vector_one + self.B
        f_prop_f_t = T.nnet.sigmoid(f_prop_combined[:input_scalar_h_dim])
        f_prop_i_t = T.nnet.sigmoid(f_prop_combined[input_scalar_h_dim:input_scalar_h_dim * 2])
        f_prop_o_t = T.nnet.sigmoid(f_prop_combined[input_scalar_h_dim * 2:input_scalar_h_dim * 3])
        f_prop_c_in_t= T.tanh(f_prop_combined[input_scalar_h_dim * 3:input_scalar_h_dim * 4])
        f_prop_c_t = f_prop_i_t * f_prop_c_in_t
        f_prop_h_t = f_prop_o_t * T.tanh(f_prop_c_t)
        self.f_prop_pointwiseEDGE = function([input_vector_one,input_scalar_h_dim],
                                         [f_prop_c_t,f_prop_h_t,f_prop_f_t,f_prop_i_t,f_prop_o_t,f_prop_c_in_t])
        # f_prop pointwise - BODY
        f_prop_combined = input_vector_one + input_vector_two + self.B
        f_prop_f_t = T.nnet.sigmoid(f_prop_combined[:input_scalar_h_dim])
        f_prop_i_t = T.nnet.sigmoid(f_prop_combined[input_scalar_h_dim:input_scalar_h_dim * 2])
        f_prop_o_t = T.nnet.sigmoid(f_prop_combined[input_scalar_h_dim * 2:input_scalar_h_dim * 3])
        f_prop_c_in_t = T.tanh(f_prop_combined[input_scalar_h_dim * 3:input_scalar_h_dim * 4])
        f_prop_c_t = f_prop_f_t * input_vector_four + f_prop_i_t * f_prop_c_in_t
        f_prop_h_t = f_prop_o_t * T.tanh(f_prop_c_t)
        self.f_prop_pointwiseBODY = function([input_vector_one,input_vector_two,input_vector_four,input_scalar_h_dim],
                                         [f_prop_c_t,f_prop_h_t,f_prop_f_t,f_prop_i_t,f_prop_o_t,f_prop_c_in_t])
        # dh_t1 (dh_t derivative from softmax section)
        back_prop_s_t = T.nnet.softmax(input_vector_h / np.float32(self.temperature))[0]
        back_prop_ds_t = -input_vector_y * (np.float32(1) / back_prop_s_t) + np.float32(1) / (np.float32(1)-back_prop_s_t) * (np.float32(1) - input_vector_y)
        back_prop_J_t = T.tile(back_prop_s_t,(back_prop_s_t.shape[0],1)).T * T.tile(-back_prop_s_t,(back_prop_s_t.shape[0],1)) + T.nlinalg.diag(back_prop_s_t)
        back_prop_dh_t1 = T.dot(back_prop_J_t,back_prop_ds_t)

        # dh_t+ (dh_t derivative from other lstm-blocks)
        back_prop_dh_t2 = T.dot(input_matrix_Wx.T,input_vector_x_sigma)
        back_prop_dh_t3 = T.dot(input_matrix_Wh.T,input_vector_h_sigma)

        # dc_t (dc_t derivative from other lstm-blocks)
        back_prop_dc_t2 = input_vector_dc21 * input_vector_dc22

        # back_prop pointwise - CORNER
        back_prop_dh_t = back_prop_dh_t1
        back_prop_dc_t = back_prop_dh_t * input_vector_five * (np.float32(1.) -  input_vector_four**2)
        back_prop_sigma_f_t = back_prop_dc_t * input_vector_six * input_vector_nine * (np.float32(1) - input_vector_nine)
        back_prop_sigma_i_t = back_prop_dc_t * input_vector_eight * input_vector_seven * (np.float32(1) - input_vector_seven)
        back_prop_sigma_o_t = back_prop_dh_t * T.tanh(input_vector_four) * input_vector_five * (np.float32(1) - input_vector_five)
        back_prop_sigma_c_in_t = back_prop_dc_t * input_vector_seven * (np.float32(1.) - input_vector_eight**2)
        back_prop_sigma_all = T.concatenate([back_prop_sigma_f_t,back_prop_sigma_i_t,back_prop_sigma_o_t,back_prop_sigma_c_in_t])
        self.back_prop_pointwiseCORNER = function([input_vector_four,input_vector_five,input_vector_six,input_vector_seven,input_vector_eight,input_vector_nine,
                                            input_vector_h,
                                            input_vector_y],
                                            [back_prop_sigma_all,back_prop_dc_t])
        # back_prop pointwise - END
        back_prop_dh_t =  back_prop_dh_t3
        back_prop_dc_t = back_prop_dh_t * input_vector_five * (np.float32(1.) -  input_vector_four**2) + back_prop_dc_t2
        back_prop_sigma_f_t = back_prop_dc_t * input_vector_six * input_vector_nine * (np.float32(1) - input_vector_nine)
        back_prop_sigma_i_t = back_prop_dc_t * input_vector_eight * input_vector_seven * (np.float32(1) - input_vector_seven)
        back_prop_sigma_o_t = back_prop_dh_t * T.tanh(input_vector_four) * input_vector_five * (np.float32(1) - input_vector_five)
        back_prop_sigma_c_in_t = back_prop_dc_t * input_vector_seven * (np.float32(1.) - input_vector_eight**2)
        back_prop_sigma_all = T.concatenate([back_prop_sigma_f_t,back_prop_sigma_i_t,back_prop_sigma_o_t,back_prop_sigma_c_in_t])
        self.back_prop_pointwiseEND = function([input_vector_four,input_vector_five,input_vector_six,input_vector_seven,input_vector_eight,input_vector_nine,
                                            input_matrix_Wh,
                                            input_vector_h_sigma,
                                            input_vector_dc21,
                                            input_vector_dc22],
                                            [back_prop_sigma_all,back_prop_dc_t])

        # back_prop pointwise - EDGE
        back_prop_dh_t = back_prop_dh_t2 * self.dropout_vector
        back_prop_dc_t = back_prop_dh_t * input_vector_five * (np.float32(1.) -  input_vector_four**2)
        back_prop_sigma_f_t = back_prop_dc_t * input_vector_six * input_vector_nine * (np.float32(1) - input_vector_nine)
        back_prop_sigma_i_t = back_prop_dc_t * input_vector_eight * input_vector_seven * (np.float32(1) - input_vector_seven)
        back_prop_sigma_o_t = back_prop_dh_t * T.tanh(input_vector_four) * input_vector_five * (np.float32(1) - input_vector_five)
        back_prop_sigma_c_in_t = back_prop_dc_t * input_vector_seven * (np.float32(1.) - input_vector_eight**2)
        back_prop_sigma_all = T.concatenate([back_prop_sigma_f_t,back_prop_sigma_i_t,back_prop_sigma_o_t,back_prop_sigma_c_in_t])
        self.back_prop_pointwiseEDGE = function([input_vector_four,input_vector_five,input_vector_six,input_vector_seven,input_vector_eight,input_vector_nine,
                                            input_matrix_Wx,
                                            input_vector_x_sigma],
                                            [back_prop_sigma_all,back_prop_dc_t])

        # back_prop pointwise - BODY
        back_prop_dh_t =  back_prop_dh_t2 * self.dropout_vector + back_prop_dh_t3
        back_prop_dc_t = back_prop_dh_t * input_vector_five * (np.float32(1.) -  input_vector_four**2) + back_prop_dc_t2
        back_prop_sigma_f_t = back_prop_dc_t * input_vector_six * input_vector_nine * (np.float32(1) - input_vector_nine)
        back_prop_sigma_i_t = back_prop_dc_t * input_vector_eight * input_vector_seven * (np.float32(1) - input_vector_seven)
        back_prop_sigma_o_t = back_prop_dh_t * T.tanh(input_vector_four) * input_vector_five * (np.float32(1) - input_vector_five)
        back_prop_sigma_c_in_t = back_prop_dc_t * input_vector_seven * (np.float32(1.) - input_vector_eight**2)
        back_prop_sigma_all = T.concatenate([back_prop_sigma_f_t,back_prop_sigma_i_t,back_prop_sigma_o_t,back_prop_sigma_c_in_t])
        self.back_prop_pointwiseBODY = function([input_vector_four,input_vector_five,input_vector_six,input_vector_seven,input_vector_eight,input_vector_nine,
                                            input_matrix_Wx,
                                            input_vector_x_sigma,
                                            input_matrix_Wh,
                                            input_vector_h_sigma,
                                            input_vector_dc21,
                                            input_vector_dc22],
                                            [back_prop_sigma_all,back_prop_dc_t])


        # Calculate gradients - BODY
        back_prop_dWx = T.tile(input_vector_one,(input_scalar_x_dim,1)).T * input_vector_two
        back_prop_dWh = T.tile(input_vector_one,(input_scalar_h_dim,1)).T * input_vector_three
        back_prop_dB = input_vector_one

        # Update Gradients - BODY
        self.back_prop_update_gradientsBODY = function([input_vector_one,input_vector_two,input_vector_three,input_scalar_x_dim,input_scalar_h_dim],
                                                updates = [(self.dWx,self.dWx + back_prop_dWx),(self.dWh,self.dWh + back_prop_dWh),(self.dB,self.dB + back_prop_dB)])

        # Update Gradients - EDGE
        self.back_prop_update_gradientsEDGE = function([input_vector_one,input_vector_two,input_scalar_x_dim],
                                                updates = [(self.dWx,self.dWx + back_prop_dWx),(self.dB,self.dB + back_prop_dB)])
        # MeanSquare
        rms_prop_msWx = np.float32(0.9) * self.msWx + np.float32(0.1) * (self.dWx / input_scalar_one)**2
        rms_prop_msWh = np.float32(0.9) * self.msWh + np.float32(0.1) * (self.dWh / input_scalar_one)**2
        rms_prop_msB = np.float32(0.9) * self.msB + np.float32(0.1) * (self.dB / input_scalar_one)**2

        # Weights
        rms_prop_Wx = self.Wx - input_scalar_learning_rate * self.dWx / input_scalar_one / (T.sqrt(rms_prop_msWx) + np.float32(1e-8))
        rms_prop_Wh = self.Wh - input_scalar_learning_rate * self.dWh / input_scalar_one / (T.sqrt(rms_prop_msWh) + np.float32(1e-8))
        rms_prop_B = self.B - input_scalar_learning_rate * self.dB / input_scalar_one / (T.sqrt(rms_prop_msB) + np.float32(1e-8))

        # Update weights - BODY
        self.back_prop_update_weightsBODY = function([input_scalar_one,input_scalar_learning_rate],updates=[(self.msWx,rms_prop_msWx),(self.msWh,rms_prop_msWh),(self.msB,rms_prop_msB),
                                                                                 (self.Wx,rms_prop_Wx),(self.Wh,rms_prop_Wh),(self.B,rms_prop_B),
                                                                                 (self.dWx,T.zeros_like(self.dWx,dtype=floatX)),
                                                                                 (self.dWh,T.zeros_like(self.dWh,dtype=floatX)),
                                                                                 (self.dB,T.zeros_like(self.dB,dtype=floatX))])

        # Update weights - EDGE
        self.back_prop_update_weightsEDGE = function([input_scalar_one,input_scalar_learning_rate],updates=[(self.msWx,rms_prop_msWx),(self.msB,rms_prop_msB),
                                                                                 (self.Wx,rms_prop_Wx),(self.B,rms_prop_B),
                                                                                 (self.dWx,T.zeros_like(self.dWx,dtype=floatX)),
                                                                                 (self.dB,T.zeros_like(self.dB,dtype=floatX))])
    def forward_propagateBODY(self,x_t,c_t_minus_one,h_t_minus_one):
        # log inputs
        self.x_t = x_t
        self.c_t_minus_one = c_t_minus_one
        self.h_t_minus_one = h_t_minus_one

        # forward propagation
        f_prop_dot_output = self.f_prop_dot(x_t,h_t_minus_one)
        self.f_prop_pointwise_output = self.f_prop_pointwiseBODY(f_prop_dot_output[0],f_prop_dot_output[1],c_t_minus_one,self.h_dim)
    def forward_propagateEDGE(self,x_t):
        # log inputs
        self.x_t = x_t
        self.c_t_minus_one = np.zeros((self.h_dim,)).astype(dtype=floatX)
        self.h_t_minus_one = np.zeros((self.h_dim,)).astype(dtype=floatX)
        # forward propagation
        f_prop_dot_outputWx = self.f_prop_dotEdge(x_t)
        self.f_prop_pointwise_output = self.f_prop_pointwiseEDGE(f_prop_dot_outputWx,self.h_dim)
    def forward_propagate_dropoutBODY(self,x_t,c_t_minus_one,h_t_minus_one,dropout_vector):
        # log inputs
        self.x_t = x_t
        self.c_t_minus_one = c_t_minus_one
        self.h_t_minus_one = h_t_minus_one

        # forward propagation
        f_prop_dot_output = self.f_prop_dot_dropoutBODY(x_t,h_t_minus_one,dropout_vector)
        self.f_prop_pointwise_output = self.f_prop_pointwiseBODY(f_prop_dot_output[0],f_prop_dot_output[1],c_t_minus_one,self.h_dim)
    def forward_propagate_dropoutEDGE(self,x_t,dropout_vector):
        # log inputs
        self.x_t = x_t
        self.c_t_minus_one = np.zeros((self.h_dim,)).astype(dtype=floatX)
        self.h_t_minus_one = np.zeros((self.h_dim,)).astype(dtype=floatX)
        # forward propagation
        f_prop_dot_outputWx = self.f_prop_dot_dropoutEdge(x_t,dropout_vector)
        self.f_prop_pointwise_output = self.f_prop_pointwiseEDGE(f_prop_dot_outputWx,self.h_dim)
    def backward_propagateCORNER(self,y_t):
        # corner block
        # log y_t
        self.y_t = y_t

        # pointwise operations
        self.node_error_all = self.back_prop_pointwiseCORNER(self.f_prop_pointwise_output[0],self.f_prop_pointwise_output[4],
                                                       self.c_t_minus_one,self.f_prop_pointwise_output[3],
                                                       self.f_prop_pointwise_output[5],self.f_prop_pointwise_output[2],
                                                       self.f_prop_pointwise_output[1],y_t)
        # matrix operations
    def backward_propagateEND(self,y_t):
        # log y_t
        self.y_t = y_t

        # pointwise operations
        self.node_error_all = self.back_prop_pointwiseEND(self.f_prop_pointwise_output[0],self.f_prop_pointwise_output[4],
                                                        self.c_t_minus_one,self.f_prop_pointwise_output[3],
                                                        self.f_prop_pointwise_output[5],self.f_prop_pointwise_output[2],
                                                        self.lstm_block_right.Wh.get_value(),
                                                        self.lstm_block_right.node_error_all[0],
                                                        self.lstm_block_right.f_prop_pointwise_output[2],
                                                        self.lstm_block_right.node_error_all[1])
    def backward_propagateEDGE(self):
        # pointwise operations
        self.node_error_all = self.back_prop_pointwiseEDGE(self.f_prop_pointwise_output[0],self.f_prop_pointwise_output[4],
                                                        self.c_t_minus_one,self.f_prop_pointwise_output[3],
                                                        self.f_prop_pointwise_output[5],self.f_prop_pointwise_output[2],
                                                        self.lstm_block_front.Wx.get_value(),
                                                        self.lstm_block_front.node_error_all[0])
    def backward_propagateBODY(self):
        # edge block
        # pointwise operations
        self.node_error_all = self.back_prop_pointwiseBODY(self.f_prop_pointwise_output[0],self.f_prop_pointwise_output[4],
                                                        self.c_t_minus_one,self.f_prop_pointwise_output[3],
                                                        self.f_prop_pointwise_output[5],self.f_prop_pointwise_output[2],
                                                        self.lstm_block_front.Wx.get_value(),
                                                        self.lstm_block_front.node_error_all[0],
                                                        self.lstm_block_right.Wh.get_value(),
                                                        self.lstm_block_right.node_error_all[0],
                                                        self.lstm_block_right.f_prop_pointwise_output[2],
                                                        self.lstm_block_right.node_error_all[1])
    def update_gradientsBODY(self):
        self.back_prop_update_gradientsBODY(self.node_error_all[0],self.x_t,self.h_t_minus_one,self.x_dim,self.h_dim)
    def update_gradientsEDGE(self):
        self.back_prop_update_gradientsEDGE(self.node_error_all[0],self.x_t,self.x_dim)
    def update_weightsBODY(self,batch_size):
        self.back_prop_update_weightsBODY(batch_size,self.learning_rate)
    def update_weightsEDGE(self,batch_size):
        self.back_prop_update_weightsEDGE(batch_size,self.learning_rate)

# Recurrent neural network (LSTM wrapper)
class RNN:
    def __init__(self,rnn_structure,x_h_dim,softmax_temperature=1.0):
        # log inputs
        self.x_h_dim = x_h_dim
        self.network_width = rnn_structure[1]
        self.network_depth = rnn_structure[0]
        self.temperature = softmax_temperature
        input_vector_one = T.vector(dtype = floatX)
        input_vector_two = T.vector(dtype = floatX)
        input_scalar_one = T.scalar(dtype = floatX)

        # Error
        error_output = T.sum(T.nnet.binary_crossentropy(T.nnet.softmax(input_vector_one / np.float32(self.temperature))[0],input_vector_two)) / input_vector_one.shape[0]
        self.calc_error = function([input_vector_one,input_vector_two],error_output)

        # Softmax
        softmax_output = T.nnet.softmax(input_vector_one / np.float32(self.temperature))[0]
        self.softmax = function([input_vector_one],softmax_output)


        # create block network
        self.lstm_network = list()
        for x in range(0,self.network_depth):
            self.lstm_network.append(list())

        # fill in block lstm_network with LSTM blocks
        for x in range(0,self.network_depth):
            for y in range(0,self.network_width):
                self.lstm_network[x].append(LSTM(x_h_dim[x],x_h_dim[x+1],temperature=softmax_temperature))

        # assign block connections
        # right-facing connections
        for x in range(0,self.network_depth):
            for y in range(0,self.network_width - 1):
                self.lstm_network[x][y].lstm_block_right = self.lstm_network[x][y+1]
        # forward-facing connections
        for x in range(0,self.network_depth - 1):
            for y in range(0,self.network_width):
                self.lstm_network[x][y].lstm_block_front = self.lstm_network[x+1][y]
    def forward_propagate(self,x_matrix):
        # t = 1
        self.lstm_network[0][0].forward_propagateEDGE(x_matrix[0])
        for x in range(1,self.network_depth):
            self.lstm_network[x][0].forward_propagateEDGE(self.lstm_network[x - 1][0].f_prop_pointwise_output[1])
        # t = 1+
        for x in range(1,self.network_width):
            self.lstm_network[0][x].forward_propagateBODY(x_matrix[x],self.lstm_network[0][x-1].f_prop_pointwise_output[0],self.lstm_network[0][x-1].f_prop_pointwise_output[1])
            for y in range(1,self.network_depth):
                self.lstm_network[y][x].forward_propagateBODY(self.lstm_network[y-1][x].f_prop_pointwise_output[1],self.lstm_network[y][x-1].f_prop_pointwise_output[0],self.lstm_network[y][x-1].f_prop_pointwise_output[1])
    def forward_propagate_dropout(self,x_matrix):
        # t = 1
        self.lstm_network[0][0].forward_propagateEDGE(x_matrix[0])
        for x in range(1,self.network_depth):
            self.lstm_network[x][0].forward_propagate_dropoutEDGE(self.lstm_network[x - 1][0].f_prop_pointwise_output[1],self.lstm_network[x-1][0].dropout_vector.get_value())
        # t = 1+
        for x in range(1,self.network_width):
            self.lstm_network[0][x].forward_propagateBODY(x_matrix[x],self.lstm_network[0][x-1].f_prop_pointwise_output[0],self.lstm_network[0][x-1].f_prop_pointwise_output[1])
            for y in range(1,self.network_depth):
                self.lstm_network[y][x].forward_propagate_dropoutBODY(self.lstm_network[y-1][x].f_prop_pointwise_output[1],self.lstm_network[y][x-1].f_prop_pointwise_output[0],self.lstm_network[y][x-1].f_prop_pointwise_output[1],self.lstm_network[y-1][x].dropout_vector.get_value())
    def backward_propagate(self,y_matrix):
        # Corner node error
        self.lstm_network[self.network_depth-1][self.network_width - 1].backward_propagateCORNER(y_matrix[self.network_width-1])
        # Edge node error
        for x in range(1,self.network_depth):
            self.lstm_network[self.network_depth - x - 1][self.network_width - 1].backward_propagateEDGE()
        # End node error
        for x in range(1,self.network_width):
            self.lstm_network[self.network_depth-1][self.network_width - x - 1].backward_propagateEND(y_matrix[self.network_width-x-1])
        # Body node error
        for x in range(1,self.network_width):
            for y in range(1,self.network_depth):
                    self.lstm_network[self.network_depth - y - 1][self.network_width - x - 1].backward_propagateBODY()
        # WEIGHT ERROR and UPDATE WEIGHTS
        for x in range(0,self.network_width-1):
            for y in range(0,self.network_depth):
                self.lstm_network[self.network_depth-y-1][self.network_width - x - 1].update_gradientsBODY()
        for y in range(0,self.network_depth):
            self.lstm_network[self.network_depth-y-1][0].update_gradientsEDGE()
    def update_weights(self,batch_size):
        # WEIGHT ERROR and UPDATE WEIGHTS
        for x in range(0,self.network_width-1):
            for y in range(0,self.network_depth):
                self.lstm_network[self.network_depth-y-1][self.network_width - x - 1].update_weightsBODY(batch_size)
        for y in range(0,self.network_depth):
            self.lstm_network[self.network_depth-y-1][0].update_weightsEDGE(batch_size)
    def error(self):
        error = np.float32(0)
        output = self.lstm_network[self.network_depth-1][self.network_width-1].f_prop_pointwise_output[1]
        y_vector = self.lstm_network[self.network_depth-1][self.network_width-1].y_t
        error += self.calc_error(output,y_vector)
        return error
    def train(self,data,batch_size,display_progress = False):
        self.error_log = list()
        # iterating through total number of batches
        for x in range(0,int((len(data)/(self.network_width + batch_size)))):
            if(display_progress):
                print(x)
            # iterating through batch size
            for y in range(0,batch_size):
                self.forward_propagate_dropout(data[y + x * batch_size :y + x * batch_size + self.network_width])
                self.backward_propagate(data[y + x * batch_size + 1:y + x * batch_size + 1 + self.network_width])
                self.update_dropout_vectors()
            self.update_weights(batch_size)
            #self.update_dropout_vectors()
        self.error_log.append(self.error())
    def discretize_prediction(self,prediction_vector):
        index_max = np.argmax(prediction_vector)
        prediction_vector  = np.zeros((len(prediction_vector,))).astype(dtype = floatX)
        prediction_vector[index_max] = 1
        return prediction_vector
    def predict(self,kindling_text,prediction_length,calculate_perplexity):
        # Calculate prediction vectors
        kindling_text_trunc = np.copy(kindling_text[:self.network_width])
        prediction_text = np.copy(kindling_text_trunc)
        for x in range(0,prediction_length):
            self.forward_propagate(kindling_text_trunc[x:x+self.network_width])
            prediction = self.softmax(self.lstm_network[self.network_depth - 1][self.network_width - 1].f_prop_pointwise_output[1])
            kindling_text_trunc = np.row_stack((kindling_text_trunc,self.discretize_prediction(prediction)))
            prediction_text = np.row_stack((prediction_text,prediction))
        test_perplexity = 0
        # Calculate perplexity
        if(calculate_perplexity):
            test_perplexity = self.perplexity(kindling_text[self.network_width:prediction_length + self.network_width],prediction_text[self.network_width:prediction_length + self.network_width])
        # Return prediction vectors + perplexity
        return (kindling_text_trunc,test_perplexity)
    def beam_max(self,prediction_vector,x_matrix,beam_length):
        order = np.argsort(prediction_vector)[::-1]
        identity = np.identity(len(prediction_vector)).astype(dtype=floatX)
        # Beam branches
        beam_1 = identity[order[0]]
        beam_2 = identity[order[1]]
        beam_3 = identity[order[2]]
        beams = np.vstack((beam_1,beam_2,beam_3))
        # Create beam tree-leaves
        self.forward_propagate(np.vstack((x_matrix,beam_1))[1:])
        beam_1_max = np.max(np.sum(self.beam_search(np.vstack((x_matrix,beam_1))[1:],np.array([prediction_vector[order[0]]]),beam_length - 1),axis=1))
        self.forward_propagate(np.vstack((x_matrix,beam_2))[1:])
        beam_2_max = np.max(np.sum(self.beam_search(np.vstack((x_matrix,beam_2))[1:],np.array([prediction_vector[order[1]]]),beam_length - 1),axis=1))
        self.forward_propagate(np.vstack((x_matrix,beam_3))[1:])
        beam_3_max = np.max(np.sum(self.beam_search(np.vstack((x_matrix,beam_3))[1:],np.array([prediction_vector[order[2]]]),beam_length - 1),axis=1))
        # beam_max
        beam_max = np.argsort([beam_1_max,beam_2_max,beam_3_max])[::-1][0]
        return(beams[beam_max])
    def beam_search(self,x_matrix,predict_tree,beam_length):
        self.forward_propagate(x_matrix)
        prediction_vector = self.softmax(self.lstm_network[self.network_depth - 1][self.network_width - 1].f_prop_pointwise_output[1])
        predict_max_3 = np.sort(prediction_vector)[::-1][:3]
        predict_index_max = np.argsort(prediction_vector)[::-1]
        # Recursion ends:
        if(beam_length==0):
            predict = np.hstack((np.repeat(predict_tree,3,axis=0),np.repeat(predict_max_3,len(predict_tree))))
            return(predict)# beam branches
        else:
            identity = np.identity(len(prediction_vector)).astype(dtype=floatX)
            # Predict beams
            beam_1 = identity[predict_index_max[0]]
            beam_2 = identity[predict_index_max[1]]
            beam_3 = identity[predict_index_max[2]]
            # Prepare predict branches
            predict_branch_1 = np.hstack((np.repeat(predict_tree,3,axis=0),np.repeat(predict_max_3[0],len(predict_tree))))
            predict_branch_2 = np.hstack((np.repeat(predict_tree,3,axis=0),np.repeat(predict_max_3[1],len(predict_tree))))
            predict_branch_3 = np.hstack((np.repeat(predict_tree,3,axis=0),np.repeat(predict_max_3[2],len(predict_tree))))
            # Fill predict branches via recursion
            predict_branch_1 = self.beam_search(np.vstack((x_matrix,beam_1))[1:],predict_branch_1,beam_length-1)
            predict_branch_2 = self.beam_search(np.vstack((x_matrix,beam_2))[1:],predict_branch_2,beam_length-1)
            predict_branch_3 = self.beam_search(np.vstack((x_matrix,beam_3))[1:],predict_branch_3,beam_length-1)
            # Combine and return
            predict_tree = np.vstack((predict_branch_1,predict_branch_2,predict_branch_3))
            return(predict_tree)
    def predict_beam_search(self,kindling_text,prediction_length,beam_length,calculate_perplexity):
        # Calculate prediction vectors
        kindling_text_trunc = np.copy(kindling_text[:self.network_width])
        prediction_text = np.copy(kindling_text_trunc)
        for x in range(0,prediction_length):
            self.forward_propagate(kindling_text_trunc[x:x+self.network_width])
            prediction = self.softmax(self.lstm_network[self.network_depth - 1][self.network_width - 1].f_prop_pointwise_output[1])
            kindling_text_trunc = np.row_stack((kindling_text_trunc,self.beam_max(prediction,kindling_text_trunc[x:x+self.network_width],beam_length)))
            prediction_text = np.row_stack((prediction_text,prediction))
        test_perplexity = 0
        # Calculate perplexity
        if(calculate_perplexity):
            test_perplexity = self.perplexity(kindling_text[self.network_width:prediction_length + self.network_width],prediction_text[self.network_width:prediction_length + self.network_width])
        # Return prediction vectors + perplexity
        return (kindling_text_trunc,test_perplexity)
    def update_learning_rate(self,learning_rate):
        # WEIGHT ERROR and UPDATE WEIGHTS
        for x in range(0,self.network_depth):
            for y in range(0,self.network_width):
                lstm = self.lstm_network[x][y]
                lstm.learning_rate = np.float32(learning_rate)
    def perplexity(self,x_matrix,prediction):
        sample_size = len(x_matrix)
        inter_perplexity = 1.0
        for x in range(0,sample_size):
            arg = np.argmax(x_matrix[x])
            prob = prediction[x][arg]
            inter_perplexity *= prob
        return np.float32(inter_perplexity**(-1/sample_size))
    def update_dropout_vectors(self):
        for x in range(0,self.network_depth):
            for y in range(0,self.network_width):
                lstm = self.lstm_network[x][y]
                p_dropout = lstm.p_dropout
                h_dim = lstm.h_dim
                lstm.dropout_vector.set_value(np.random.choice([0, 1], size=(h_dim,), p=[p_dropout,1-p_dropout]).astype(dtype=floatX))

# Ix-to-char
def unit_to_char(data):
    char = ""
    for x in range(0,len(data)):
        char_id = np.argmax(data[x])
        char += ix_to_char[char_id]
    return char
# Save RNN to CSV
def save_rnn_csv(rnn, folder_name):
    import os
    import pickle
    file_prefix = folder_name + '/'
    # create folder if it doesn't exist already
    if(os.path.exists(folder_name) == False):
        os.makedirs(folder_name)
    # clear folder
    files = os.listdir(folder_name)
    for file in files:
        os.remove(folder_name + '/' + file)
    pickle.dump(chars,open(file_prefix + "chars.obj","wb"))
    for x in range(0,rnn.network_depth):
        for y in range(0,rnn.network_width):
            lstm = rnn.lstm_network[x][y]
            # x-matrices
            pickle.dump(lstm.Wx.get_value(),open(file_prefix + str(x)+str(y)+'Wx.obj','wb'))

            # h-matrices
            pickle.dump(lstm.Wh.get_value(),open(file_prefix + str(x)+str(y)+'Wh.obj','wb'))

            # b-vectors
            pickle.dump(lstm.B.get_value(),open(file_prefix + str(x)+str(y)+'B.obj','wb'))
# Load RNN from CSV
def load_rnn_csv(rnn, folder_name):
    import os
    import pickle
    file_prefix = folder_name + '/'
    chars = pickle.load(open(file_prefix + "chars.obj","rb"))
    for x in range(0,rnn.network_depth):
        for y in range(0,rnn.network_width):
            lstm = rnn.lstm_network[x][y]
            # x-matrices
            lstm.Wx.set_value(pickle.load(open(file_prefix + str(x) + str(y) + "Wx.obj",'rb')))

            # h-matrices
            lstm.Wh.set_value(pickle.load(open(file_prefix + str(x) + str(y) + "Wh.obj",'rb')))

            # b-vectors
            lstm.B.set_value(pickle.load(open(file_prefix + str(x) + str(y) + "B.obj",'rb')))
    return chars


# Initialize RNN
self = RNN((3,5),(vocab_size,32,32,vocab_size),softmax_temperature=1.0)

# Train RNN for 50 epochs and print training results
####################
for x in range(0,50):
    self.train(chars_as_vectors,128)
    print(unit_to_char(self.predict(chars_as_vectors[:self.network_width],100,False)))