import sys;
import copy;
import numpy as np;
import scipy.io as sio
import scipy.signal as sg
np.random.seed(0);
import matplotlib.pyplot as plt

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# generate large training dataset

m = 500;

sequence_dim = 100;
hidden_dim = 200
output_dim = 1
ts=m-sequence_dim-1
pred_len=500
alpha = 0.01


a=np.linspace(0,m,m+1)

X=np.zeros((sequence_dim,ts))
x=np.sin(a/5)+np.cos(a/25)
y=np.zeros((ts,1))
layer=np.zeros((ts,1))
error=np.zeros((ts,1))
pred=list()

for i in range(ts):

    X[:,i]=x[i:i+sequence_dim];
    y[i]=x[i+sequence_dim]

# input variables and initialization

Wix = 2*np.random.random((hidden_dim, sequence_dim))-1
Wih = 2*np.random.random((hidden_dim, hidden_dim))-1
Wfx = 2*np.random.random((hidden_dim, sequence_dim))-1
Wfh = 2*np.random.random((hidden_dim, hidden_dim))-1
Wox = 2*np.random.random((hidden_dim, sequence_dim))-1
Woh = 2*np.random.random((hidden_dim, hidden_dim))-1
Wcx = 2*np.random.random((hidden_dim, sequence_dim))-1
Wch = 2*np.random.random((hidden_dim, hidden_dim))-1
Wy = 2*np.random.random((output_dim,hidden_dim)) - 1

Wix_update = np.zeros_like(Wix)
Wih_update = np.zeros_like(Wih)
Wfx_update = np.zeros_like(Wfx)
Wfh_update = np.zeros_like(Wfh)
Wox_update = np.zeros_like(Wox)
Woh_update = np.zeros_like(Woh)
Wcx_update = np.zeros_like(Wcx)
Wch_update = np.zeros_like(Wch)
Wy_update = np.zeros_like(Wy)

layer_0 = np.zeros((sequence_dim,1))
a = np.zeros((hidden_dim,1))
layer_i = np.zeros((hidden_dim,1))
layer_f = np.zeros((hidden_dim,1))
layer_o = np.zeros((hidden_dim,1))
layer_a = np.zeros((hidden_dim,1))
layer_h_minus = np.zeros((hidden_dim,1))
layer_h = np.zeros((hidden_dim,1))
C_minus = np.zeros((hidden_dim,1))
C = np.zeros((hidden_dim,1))
output = np.zeros((output_dim,1))

b_i = np.zeros((1,1))
b_f = np.zeros((1,1))
b_o = np.zeros((1,1))
b_a = np.zeros((1,1))
b_h = np.zeros((1,1))

b_i_update = np.zeros_like(b_i)
b_f_update = np.zeros_like(b_f)
b_o_update = np.zeros_like(b_o)
b_a_update = np.zeros_like(b_a)
b_h_update = np.zeros_like(b_h)

layer_0_delta = np.zeros_like(layer_0)
layer_i_delta = np.zeros_like(layer_i)
layer_f_delta = np.zeros_like(layer_f)
layer_o_delta = np.zeros_like(layer_o)
layer_a_delta = np.zeros_like(layer_a)
layer_h_delta = np.zeros_like(layer_h)
C_delta = np.zeros_like(C)
C_minus_delta = np.zeros_like(C_minus)
output_error = np.zeros_like(output)

layer_i_delta_z = np.zeros_like(layer_i_delta)
layer_f_delta_z = np.zeros_like(layer_f_delta)
layer_o_detla_z = np.zeros_like(layer_o_delta)
layer_a_delta_z = np.zeros_like(layer_a_delta)


# Training

for j in range(100):

    for i in range (ts-1):

        # Forward pass

        layer_0 = X[:,[i]]
        a = np.dot(Wcx,layer_0)+np.dot(Wch,layer_h_minus) + b_a
        layer_a = np.tanh(a)
        layer_i = sigmoid(np.dot(Wix,layer_0)+np.dot(Wih,layer_h_minus) + b_i)
        layer_f = sigmoid(np.dot(Wfx,layer_0)+np.dot(Wfh,layer_h_minus) + b_f)
        layer_o = sigmoid(np.dot(Wox,layer_0)+np.dot(Woh,layer_h_minus) + b_o)

        C = np.multiply(layer_i,layer_a) + np.multiply(C_minus, layer_f)

        layer_h = np.multiply(layer_o, np.tanh(C))

        output = np.dot(Wy,layer_h)

        # Backward

        error[i] = 1/2*(y[i]-output)**2    
        output_error = y[i]-output

        layer_h_delta = np.dot(np.transpose(Wy),output_error)
        layer_o_delta = np.multiply(np.multiply(layer_h_delta,np.tanh(C)),np.multiply(layer_o,(1-layer_o)))
        C_delta = np.multiply(layer_h_delta, layer_o, (1-np.tanh(C)**2))
        layer_i_delta = np.multiply(np.multiply(C_delta,layer_a),np.multiply(layer_i,(1-layer_i)))
        layer_f_delta = np.multiply(np.multiply(C_delta,C_minus),np.multiply(layer_f,(1-layer_f)))
        layer_a_delta = np.multiply(np.multiply(C_delta,layer_i),(1-np.tanh(a)**2))
        C_minus_delta = np.multiply(C_delta,layer_f)

        b_i_update = np.sum(layer_i_delta)
        b_f_update = np.sum(layer_f_delta)
        b_o_update = np.sum(layer_o_delta)
        b_a_update = np.sum(layer_a_delta)
        b_h_update = np.sum(layer_h_delta)

        b_i = b_i_update * alpha
        b_f = b_f_update * alpha
        b_o = b_o_update * alpha
        b_a = b_a_update * alpha
        b_h = b_h_update * alpha

        Wix_update = np.transpose(np.dot(layer_0,np.transpose(layer_i_delta)))
        Wih_update = np.transpose(np.dot(layer_h_minus,np.transpose(layer_i_delta)))
        Wfx_update = np.transpose(np.dot(layer_0,np.transpose(layer_f_delta)))
        Wfh_update = np.transpose(np.dot(layer_h,np.transpose(layer_f_delta)))
        Wox_update = np.transpose(np.dot(layer_0,np.transpose(layer_o_delta)))
        Woh_update = np.transpose(np.dot(layer_h,np.transpose(layer_o_delta)))
        Wcx_update = np.transpose(np.dot(layer_0,np.transpose(C_minus_delta)))
        Wch_update = np.transpose(np.dot(layer_h,np.transpose(C_minus_delta)))
        Wy_update = np.transpose(np.dot(layer_h,np.transpose(output_error)))

        Wix += Wix_update*alpha
        Wih += Wih_update*alpha
        Wfx += Wfx_update*alpha
        Wfh += Wfh_update*alpha
        Wox += Wox_update*alpha
        Woh += Woh_update*alpha
        Wcx += Wcx_update*alpha
        Wch += Wch_update*alpha
        Wy += Wy_update*alpha
                
        layer_h_minus = layer_h
        C_minus = C

    print(j)
# Predict

base = X[:,[ts-1]]
#layer_h_minus *= 0
#C_minus *= 0

for j in range(pred_len):

    layer_0 = base
    a = np.dot(Wcx,layer_0)+np.dot(Wch,layer_h_minus)
    layer_a = np.tanh(a)
    layer_i = sigmoid(np.dot(Wix,layer_0)+np.dot(Wih,layer_h_minus))
    layer_f = sigmoid(np.dot(Wfx,layer_0)+np.dot(Wfh,layer_h_minus))
    layer_o = sigmoid(np.dot(Wox,layer_0)+np.dot(Woh,layer_h_minus))

    C = np.multiply(layer_i,layer_a) + np.multiply(C_minus, layer_f)

    layer_h = np.multiply(layer_o, np.tanh(C))

    output = np.dot(Wy,layer_h)
    layer_h_minus=layer_h
    C_minus=C
    
    pred.append(output)
    add=output[0,0]
    base[0:-1]=base[1:]
    base[-1]=add
    
# plt.show()
pred = np.asarray(pred)
pred=pred[:,0,0]


plt.plot(pred)
plt.show()

    
plt.plot(error)
plt.show()
