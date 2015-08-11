# -*- coding: utf-8 -*-
import theano, theano.tensor as T
import numpy as np
import theano_lstm


from theano_lstm import LSTM, RNN, StackedCells, Layer, create_optimization_updates, masked_loss

model = StackedCells(input_size=20, celltype=RNN, layers =[10] * 1)
model.layers.append(Layer(10, 1, activation = T.tanh))

def oneStep(gfs_tm2,gfs_tm1,gfs_t,pm25_tm2,pm25_tm1,*prev_hiddens):
    input_x=gfs_tm2+gfs_tm1+gfs_t+pm25_tm2+pm25_tm1
    new_states = model.forward(input_x, prev_hiddens)
    #错位之后返回
    return [new_states[-1]]+new_states[:-1]

gfs=T.matrix()
initial_predict=T.matrix()
steps=T.iscalar()
    
result, updates = theano.scan(oneStep,
                            n_steps=steps,
                            sequences=[dict(input=gfs, taps=[-2,-1,-0])],
                            outputs_info=[dict(initial=initial_predict, taps=[-2,-1])] + [dict(initial=layer.initial_hidden_state, taps=[-1]) for layer in model.layers if hasattr(layer, 'initial_hidden_state')])
                            
                            
pred_fun = theano.function(inputs=[gfs,initial_predict,steps],outputs =result[0],updates=updates,allow_input_downcast=True)

gfs=np.arange(24).reshape(4,6)
pm25in=np.arange(1,5).reshape(4,1)
a=pred_fun(gfs,pm25in,1)

#test
model.layers[0].linear_matrix.get_value()
model.layers[0].linear_matrix.get_value().shape
