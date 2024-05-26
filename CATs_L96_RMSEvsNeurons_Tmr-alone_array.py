import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Lambda
from tensorflow.keras.optimizers import RMSprop, SGD, Adagrad, Adadelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MinMaxScaler
from datetime import datetime
import netCDF4 as nc
import gc

# + -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# + -+-+-+-+-+-+-+-+-+-+-+- Model definitions -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# + -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def L96_twolevel(y,t,K=36,J=10,h=1,F=10,c=10,b=10):
    X = y[:K]
    Y = y[K:].reshape((J,K))
    dydt = np.zeros(y.shape)
    dydt[:K] = - np.roll(X,-1)*(np.roll(X,-2)-np.roll(X,1)) - \
    X -(h*c)*np.mean(Y,axis=0) + F
    dydt[K:] = -c*(b*np.roll(Y,(1,0))*(np.roll(Y,(2,0))-np.roll(Y,(-1,0))) + \
                   Y - (h/J)*np.tile(X,(J,1))).reshape(K*J)
    return list(dydt)

def L96_onelevel(y,t,F=10):
    dydt = -np.roll(y,-1)*(np.roll(y,-2)-np.roll(y,1)) - y + F
    return list(dydt)

# + -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# + -+-+-+-+-+-+-+-+-+-+-+-+-+-+- Data +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-
# + -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# params
K = 8; J = 4; F = 10
h = 1; c = 10; b = 10 # default: b=10
t0 = 0
tend = t0+1000
dt = 0.1
nt = int((tend-t0)/dt)
t = np.linspace(t0,tend,nt+1)

# initialization
np.random.seed(10)
Xinit = np.random.normal(0.25*F,0.5*F,K) # resolved

np.random.seed(20)
ss_init = np.random.normal(0,1,J*K) # unresolved
Yinit = np.append(Xinit, ss_init) # resolved + unresolved

# integrate the Lorenz models over t
# Perfect Model
args_twolevel = (K,J,h,F,c,b)
states_twolevel = odeint(L96_twolevel,Yinit,t,args=args_twolevel)

# Imperfect Model
# Here the imperfections are due to (1) the absence of small-scales, and
# (2) error in the forcing term
F_imp = 3
args_onelevel = (F_imp,)
states_onelevel = odeint(L96_onelevel,Xinit,t,args=args_onelevel)

# + -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# + -+-+-+-+-+-+-+-+-+-+-+-+- degree-1 polynomial fitting +-+-+-+-+-+-+-+-+-+-+-+-
# + -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Fit degree 1 polynomial
slope, intercept = np.polyfit(states_twolevel[:8000,:K].ravel(), 
                              states_twolevel[:8000,K:2*K].ravel(), 1)
p_deg1 = np.poly1d([slope, intercept]) # create a 1d polynomial object
print('Slope:',slope)
print('Intercept:',intercept)

#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#+-+-+-+-+-+-+-+ degree-1 polynomial fitting to normalized data +-+-+-+-+-+-+-+-
#+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
normalization = 'standard' # values: standard/min-max

if normalization=='standard':
    scaler_ref = StandardScaler()
    scaler_ref.fit(states_twolevel[:8000,:])

elif normalization=='min-max':
    scaler_ref = MinMaxScaler()
    scaler_ref.fit(states_twolevel[:8000,:])

else:
    ValueError('Normalization type not recognised')

states_twolevel_norm = scaler_ref.transform(states_twolevel)
slope_norm, intercept_norm = np.polyfit(states_twolevel_norm[:8000,:K].ravel(),
                                        states_twolevel_norm[:8000,K:2*K].ravel(), 1)

print('Slope - Normalized:',slope_norm)
print('Intercept - Normalized:',intercept_norm)

# + -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# + -+-+-+-+-+-+- Bootstrap to compute 95% confidence interval +-+-+-+-+-+-+-+-+-+
# + -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

# draw bootstrap sample from a 1D dataset
def draw_random_samples(data):
    return np.random.choice(data, size=len(data))

# perform pair bootstrpping
def bootstrap(x, size=10000):
    out = np.empty(size)
    for i in range(size):
        indices = np.arange(np.shape(x)[0])
        bs_indices = draw_random_samples(indices)
        resampled_dat = x[bs_indices,:K]
        out[i] = np.sqrt(np.nanmean(resampled_dat**2, axis=(0,1)))
    return out

# + -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# + -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+- Forecast -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
# + -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Set forecast time step = dt = training time step (this is required at this 
# point during `training`. You can change it later in the RK4 integration during
# the forecast step. A single model time unit (MTU), i.e., 1 MTU, is approx. 
# equal to 5 `atmospheric days`

bs_size = 1000
n_neurons = np.linspace(5,100,20)
rmse_bs_tmr_alone = np.zeros((len(n_neurons),bs_size))

for nn in range(len(n_neurons)):
    neurons = round(n_neurons[nn])
    n_mtu = 1 # intended forecast period in terms of MTU  
    dt_fcast = dt # forecast time step
    nts_fcast = round(n_mtu/dt_fcast)  # number of forecast time steps
    t_fcast = np.linspace(0,n_mtu,nts_fcast+1)

    # h for RK4 integration
    # This is the RK4 integration time step. Keep it small for convergence.
    h_rk4 = 0.05 
    n_steps_rk4 = round(n_mtu/h_rk4)
    print(f"Number of integrating time steps are: {n_steps_rk4}")

    tmp = t.shape[0]-nts_fcast
    catalogue0 = np.zeros((K*tmp,K))
    catalogue1 = np.zeros((K*tmp,K))
    for i in range(K):
        # initial condition
        catalogue0[tmp*i:tmp*(i+1),:] = np.roll(states_onelevel[:-nts_fcast,:],
                                                i,axis=1) 
        # final condition
        catalogue1[tmp*i:tmp*(i+1),:] = np.roll(states_onelevel[nts_fcast:,:],
                                                i,axis=1)  
    
    # Normalize the catalogues
    normalization = 'standard' # values: standard/min-max

    if normalization=='standard':
        scaler_cat0 = StandardScaler()
        scaler_cat0.fit(catalogue0)
        catalogue0_normal = scaler_cat0.transform(catalogue0)
    
        scaler_cat1 = StandardScaler()
        scaler_cat1.fit(catalogue1)
        catalogue1_normal = scaler_cat1.transform(catalogue1)

    elif normalization=='min-max':
        scaler_cat0 = MinMaxScaler()
        scaler_cat0.fit(catalogue0)
        catalogue0_normal = scaler_cat0.transform(catalogue0)
    
        scaler_cat1 = MinMaxScaler()
        scaler_cat1.fit(catalogue1)
        catalogue1_normal = scaler_cat1.transform(catalogue1)

    else:
        ValueError('Normalization type not recognised')
        
    # + -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # + -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ CATs +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    # + -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    class CATS(tf.keras.Model):
    
        # constructor
        def __init__(self, nvar_ref, nvar_model, J, isreg=False, lmbda=0.01, 
                     act_fn='linear', isbias=False, alpha=1, beta=0, 
                     initializer=tf.keras.initializers.GlorotUniform(),**kwargs):
        
            super(CATS, self).__init__(**kwargs)

            # input layer
            inputs1 = tf.keras.Input(shape=(nvar_ref,))
    
            #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            #+-+ Real to Model space tranformation -- T_rm +-+
            #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   
            if isreg:
                #h1_rm = Dense(neurons, activation=act_fn, use_bias=isbias, 
                #              kernel_regularizer=tf.keras.regularizers.L2(l2=lmbda),
                #              kernel_initializer=initializer)(inputs1)
                #h2_rm = Dense(100, activation=act_fn, use_bias=isbias,
                #              kernel_regularizer=tf.keras.regularizers.L2(l2=lmbda),
                #              kernel_initializer=initializer)(h1_rm)
                #h3_rm = Dense(100, activation=act_fn, use_bias=isbias,
                #              kernel_regularizer=tf.keras.regularizers.L2(l2=lmbda),
                #              kernel_initializer=initializer)(h2_rm)
                #h4_rm = Dense(100, activation=act_fn, use_bias=isbias,
                #              kernel_regularizer=tf.keras.regularizers.L2(l2=lmbda),
                #              kernel_initializer=initializer)(h3_rm)
                out_rm = Dense(nvar_model, activation=act_fn, use_bias=isbias,
                               kernel_regularizer=tf.keras.regularizers.L2(l2=lmbda),
                               kernel_initializer=tf.keras.initializers.zeros(),
                               trainable=False)(inputs1)
            
            else:
                #h1_rm = Dense(neurons, activation=act_fn, use_bias=isbias, 
                #              kernel_initializer=initializer)(inputs1) 
                #h2_rm = Dense(100, activation=act_fn, use_bias=isbias, 
                #              kernel_initializer=initializer)(h1_rm)
                #h3_rm = Dense(100, activation=act_fn, use_bias=isbias, 
                #              kernel_initializer=initializer)(h2_rm)
                #h4_rm = Dense(100, activation=act_fn, use_bias=isbias, 
                #              kernel_initializer=initializer)(h3_rm)
                out_rm = Dense(nvar_model, activation=act_fn, use_bias=isbias, 
                               kernel_initializer=tf.keras.initializers.zeros(),
                               trainable=False)(inputs1)
            
            # This adds the linear component
            outputs_trm = tf.keras.layers.Add()([inputs1[:,:nvar_model], out_rm])
        
            T_rm = tf.keras.Model(inputs=inputs1, outputs=outputs_trm)
        
            print(T_rm.summary())
        
            # store T_rm as a property of cats
            self.T_rm = T_rm
        
            #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            #+-+ Set the custom layer as a one layer model +-+
            #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        
            # Input layer
            inputs2 = tf.keras.Input(shape=(nvar_model,))
        
            # Output layer
            outputs_custom = ModelLayer(nvar_model, 
                                        input_shape=(nvar_model,))(inputs2)
        
            # Define the analog forecast model
            model_fcast = tf.keras.Model(inputs=inputs2, outputs=outputs_custom)
        
            print(model_fcast.summary())
        
            # store analog_fcast as a property of gmaps 
            self.model_fcast = model_fcast
        
            #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            #+-+ Model to Real space tranformation -- T_mr +-+
            #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        
            # Input layer
            inputs3 = tf.keras.Input(shape=(nvar_model,))
        
            if isreg:
                h1_mr = Dense(neurons, activation=act_fn, use_bias=isbias, 
                              kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.L2
                              (l2=lmbda))(inputs3)
                h2_mr = Dense(neurons, activation=act_fn, use_bias=isbias, 
                              kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.L2
                              (l2=lmbda))(h1_mr)
                #h3_mr = Dense(100, activation=act_fn, use_bias=isbias, 
                #              kernel_initializer=initializer,
                #              kernel_regularizer=tf.keras.regularizers.L2
                #(l2=lmbda))(h2_mr)
                #h4_mr = Dense(100, activation=act_fn, use_bias=isbias, 
                #              kernel_initializer=initializer,
                #              kernel_regularizer=tf.keras.regularizers.L2
                #(l2=lmbda))(h3_mr)
                out_mr = Dense(nvar_ref, activation=act_fn, use_bias=isbias, 
                               kernel_initializer=initializer,
                               kernel_regularizer=tf.keras.regularizers.L2
                               (l2=lmbda))(h2_mr)
            else:
                h1_mr = Dense(neurons, activation=act_fn, use_bias=isbias, 
                              kernel_initializer=initializer)(inputs3)
                h2_mr = Dense(neurons, activation=act_fn, use_bias=isbias, 
                              kernel_initializer=initializer)(h1_mr)
                #h3_mr = Dense(100, activation=act_fn, use_bias=isbias, 
                #              kernel_initializer=initializer)(h2_mr)
                #h4_mr = Dense(100, activation=act_fn, use_bias=isbias, 
                #              kernel_initializer=initializer)(h3_mr)
                out_mr = Dense(nvar_ref, activation=act_fn, use_bias=isbias, 
                               kernel_initializer=initializer)(h2_mr)
    
            # compute the linear part of the forecast
            alpha_t = tf.constant(alpha, dtype=tf.float32)
            beta_t = tf.constant(beta, dtype=tf.float32)   # alpha/beta tensors
            linear_approx_ss = alpha_t*(inputs3) + beta_t  #(None, nvar_model)
            linear_approx_ss_J = tf.tile(linear_approx_ss, [1, J]) # (None, J*nvar_model)
            linear_part = tf.concat([inputs3, linear_approx_ss_J],-1)
            print('Linear part shape:',linear_part.shape)
    
            # final output of the model
            outputs_tmr = tf.keras.layers.Add()([linear_part,out_mr])
        
            T_mr = tf.keras.Model(inputs=inputs3,outputs=outputs_tmr)
        
            print(T_mr.summary())
        
            # store T_mr as a property of gmaps 
            self.T_mr = T_mr
        
            #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            #+-+-+-+-+-+- setup the full model +-+-+-+-+-+-+-+
            #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            cat = tf.keras.Model(inputs=inputs1, outputs=T_mr(model_fcast(
                T_rm(inputs1))))
            self.cat = cat
        
        def train(self, x, y, loss='mae', optimizer='adam',metric='mae',
                  epochs=100, validation_split=0.2, batch_size=512):         
            
            # Configuring the learning process
            self.cat.compile(loss=loss, optimizer=optimizer, metrics=[metric])
            
            # This is the training part of the network:
            history = self.cat.fit(x, y, epochs=epochs, 
                                   validation_split=validation_split,
                                   batch_size=batch_size)
        
            return self.cat, history
        
        # Save weights
        def save_weights(self, path, case, **kwargs,):
            self.T_rm.save_weights(path+'L96_trm.h5')
            self.model_fcast.save_weights(path+'L96_analog_fcast.h5')
            self.T_mr.save_weights(path+'L96_tmr.h5')
        
        # Load weights
        def load_weights(self, path, case, **kwargs):
            self.T_rm.load_weights(path+'L96_trm.h5',by_name=True)
            self.model_fcast.load_weights(path+'L96_analog_fcast.h5',by_name=True)
            self.T_mr.load_weights(path+'L96_tmr.h5',by_name=True)

        # Countparams
        def count_params(self, **kwargs):
            return self.T_rm.count_params() + self.model_fcast.count_params() + \
        self.T_mr.count_params()
    
        # Predict
        def r2m(self, data):
            return self.T_rm(data)
        
        def m2r(self, data):
            return self.T_mr(data)
        
        def custom_layer(self, data):
            return self.model_fcast(data)
        
        def forecast(self,data,batch_size=1):
            out1 = self.T_rm.predict(data,batch_size=batch_size)
            out2 = self.model_fcast.predict(out1,batch_size=batch_size)
            out3 = self.T_mr.predict(out2,batch_size=batch_size)
            return out3

    #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ Custom Layer +-+-+-+-+-+-+-+-+-+-+-+-+-+-
    #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # One-level function definition to use with tensors
    @tf.function
    def L96_onelevel_tf(y,F=10):
        dydt = -tf.roll(y,shift=-1,axis=-1)*(tf.roll(y,shift=-2,axis=-1) - \
                                             tf.roll(y,shift=1,axis=-1)) - \
        y + F
        return dydt

    # One-level parameterized function definition to use with tensors 
    @tf.function
    def L96_onelevel_param_tf(y,h=1,F=10,c=10):
        # Only linear X-Y relationship in this parameterized version
        py = slope*y+ intercept
        dydt = -tf.roll(y,shift=-1,axis=-1)*(tf.roll(y,shift=-2,axis=-1) - \
                                             tf.roll(y,shift=1,axis=-1)) - \
        y + F - h*c*py
        return dydt

    @tf.function
    def RK4_solve(tensor_obj):
        # tensor_obj has shape [None,K] and contains the initial conditions.
        # The L96_onelevel_tf function returns RHS of the L96 system of ODEs. 
        # The output also has the shape [None,K]
            
        args_param = {'h':h,'F':F_imp,'c':c}
        for n in tf.range(n_steps_rk4):
            k1 = L96_onelevel_param_tf(tensor_obj, **args_param)
            stage_2 = tf.math.add(tensor_obj, tf.math.multiply(h_rk4/2, k1))
            k2 = L96_onelevel_param_tf(stage_2, **args_param)
            stage_3 = tf.math.add(tensor_obj, tf.math.multiply(h_rk4/2, k2))
            k3 = L96_onelevel_param_tf(stage_3, **args_param)
            stage_4 = tf.math.add(tensor_obj, tf.math.multiply(h_rk4, k3))
            k4 = L96_onelevel_param_tf(stage_4, **args_param)
            update =  tf.math.add(tf.math.add(tf.math.add(k1,2*k2),2*k3),k4)
            tensor_obj = tf.math.add(tensor_obj, 
                                     tf.math.multiply(h_rk4/6,update))           
                
        return tensor_obj
            
    # custom layer with custom gradient
    @tf.custom_gradient
    def model_output(X_in): # (None, K)
        #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
        #+-+-+-+-+- Inverse Transform X_in +-+-+-+-+-+- 
        #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
        if normalization == 'min-max'.lower():
            max_tensor = tf.expand_dims(tf.convert_to_tensor(
                scaler_cat0.data_max_, dtype=tf.float32), axis=0)  # (1,K)
            min_tensor = tf.expand_dims(tf.convert_to_tensor(
                scaler_cat0.data_min_, dtype=tf.float32), axis=0)  # (1,K)
            out1 = tf.math.multiply(X_in, tf.math.subtract(
                max_tensor, min_tensor)) # (None, K)
            out1 = tf.math.add(out1, min_tensor)   # (None, K)
        
        elif normalization == 'standard'.lower():
            mean_tensor = tf.expand_dims(tf.convert_to_tensor(
                scaler_cat0.mean_, dtype=tf.float32), axis=0) # (1,K)
            std_tensor = tf.expand_dims(tf.convert_to_tensor(
                np.sqrt(scaler_cat0.var_),dtype=tf.float32), axis=0) # (1,K)
            out1 = tf.math.multiply(X_in, std_tensor)   # (None,K)
            out1 = tf.math.add(out1, mean_tensor)       # (None,K)
    
        else:
            raise ValueError('Normalization type not recognised')
    
        #out1 = out1[:,:K]                 # (None, K)
        print('Out1 shape:', out1.shape)
    
        #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
        #+- Time integration using the imperfect model +- 
        #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    
        out2 = RK4_solve(out1)
        print('Shape of out2:', out2.shape) 
    
        #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
        #+-+-+-+-+- Forward Transform out2 +-+-+-+-+-+-+- 
        #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    
        # Now transform it back
        if normalization == 'min-max'.lower():
            max_tensor = tf.expand_dims(tf.convert_to_tensor(
                scaler_cat1.data_max_, dtype=tf.float32), axis=0) # (1,K)
            min_tensor = tf.expand_dims(tf.convert_to_tensor(
                scaler_cat1.data_min_, dtype=tf.float32), axis=0) # (1,K)
            out3 = tf.math.divide(tf.math.subtract(out2, min_tensor), 
                                  tf.math.subtract(max_tensor,
                                                   min_tensor)) # (None, K)
            
        elif normalization == 'standard'.lower():
            mean_tensor = tf.expand_dims(tf.convert_to_tensor(
                scaler_cat1.mean_, dtype=tf.float32), axis=0)  # (1,K)
            std_tensor = tf.expand_dims(tf.convert_to_tensor(
                np.sqrt(scaler_cat1.var_), dtype=tf.float32), 
                                        axis=0)  # (1,K)
            out3 = tf.math.divide(tf.math.subtract(out2, mean_tensor), 
                                  std_tensor)   # (None,K)
    
            print('Shape of out3:', out3.shape) 
    
        # compute gradient
        def model_grad(upstream):
            ## perform forward computation ##
            # number of nearest neighbours
            nnn = 5
    
            # convert catalogue0 and catalogue1 to tensors
            #cat0_tensor = tf.convert_to_tensor(catalogue0_normal, 
            #                                   dtype=tf.float32)  # (catsize,36)
            #cat1_tensor = tf.convert_to_tensor(catalogue1_normal, 
            #                                   dtype=tf.float32)  # (catsize,36)
            cat0_tensor = tf.convert_to_tensor(catalogue0, 
                                               dtype=tf.float32)  # (catsize,36)
            cat1_tensor = tf.convert_to_tensor(catalogue1, 
                                               dtype=tf.float32)  # (catsize,36)
        
            # compute the distance between X_in and catalogue0
            #X_in_expand = tf.expand_dims(X_in,axis=1)  # (None,1,K)
            X_in_expand = tf.expand_dims(out1,axis=1)  # (None,1,K)
        
            temp = tf.math.subtract(X_in_expand,cat0_tensor)    # (None,catsize,36)
            temp_sq = tf.math.square(temp)                      # (None,catsize,36)
            dist = tf.math.sqrt(tf.math.reduce_mean(temp_sq, 
                                                        axis=2))  # (None,catsize)
    
            # get the nearest neighbours
            sort_order = tf.argsort(dist,axis=1)[:,:nnn]      # (None,nnn)
    
            A = tf.gather(cat0_tensor, sort_order, axis=0)    # (None,nnn,36)
            B = tf.gather(cat1_tensor, sort_order, axis=0)    # (None,nnn,36)
        
            # regression coefficient for the initial condition x_in
            # Tikhonov Regularization amplitude (for collinearity issue)
            epsilon = tf.constant(1e-3) 
            AA_T = tf.linalg.matmul(A,tf.transpose(A, perm=[0,2,1])) + \
            epsilon*tf.eye(num_rows=nnn, num_columns=nnn)#(None,nnn,nnn)
        
            temp = tf.linalg.solve(AA_T,A) # (None,nnn,3)
            grad_local = tf.linalg.matmul(tf.transpose
                                          (B,perm=[0,2,1]),temp) # (None,3,3)
            grad = tf.squeeze(tf.expand_dims(upstream,axis=1)@grad_local,
                              axis=1)
            return grad
    
        return out3, model_grad
    
    # inherited from keras layers.Layer class
    class ModelLayer(tf.keras.layers.Layer): 
        def __init__(self,units,**kwargs):
            super(ModelLayer,self).__init__(**kwargs) 
            self.units = units
            #self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), 
            #                         trainable=False)

        def build(self, input_shape): 
            self.kernel = self.add_weight(name = 'kernel', \
                                          shape = (input_shape[-1], self.units),\
                                          initializer = 'identity', \
                                          trainable = False)
            super(ModelLayer, self).build(input_shape)
    
        def call(self,inputs):
            #batch_size = tf.shape(inputs)[0]
            #self.batch_size = batch_size
            return model_output(inputs@self.kernel)
    
        def compute_output_shape(self,input_shape):
            return (input_shape[0],self.units)
    
    # custom loss funtion
    def custom_loss_mae(y_truth, y_pred):
        # based on MAE (less sensitive to the outliers)
        loss = tf.math.reduce_mean(tf.math.abs(y_pred[:,:K]-y_truth[:,:K]), 
                                   axis=-1)
        return loss

    def custom_loss_mse(y_truth, y_pred):
        # based on mse (more sensitive to outliers)
        loss = tf.math.reduce_mean(tf.math.square(y_pred[:,:K]-\
                                                  y_truth[:,:K]), axis=-1)
        return loss

    #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    #+-+-+-+-+-+-+-+-+-+-+-+-+- Train CATs parameters +-+-+-+-+-+-+-+-+-+-+-
    #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # Train the Neural Network
    start = datetime.now()
    lmbda = 1/80
    isreg = False
 
    # CATS definition parameters
    def_params = {'isreg':False, 'lmbda':lmbda**2, 
                  'act_fn':'tanh', 'isbias':True,
                  #'alpha':0, 'beta':0}
                  'alpha':slope_norm, 'beta':intercept_norm}

    # CATS training hyperparameters
    hyperparams = {'loss':custom_loss_mae,'optimizer':'adam','metric':'mae',
                   'epochs':100,'validation_split':0.2,'batch_size':512,}

    # standardize the training data
    if normalization=='standard':
        scaler = StandardScaler()
        scaler.fit(states_twolevel)
        train_data = scaler.transform(states_twolevel)

    elif normalization=='min-max':
        scaler = MinMaxScaler()
        scaler.fit(states_twolevel)
        train_data = scaler.transform(states_twolevel)

    else:
        ValueError('Normalization type not recognised')

    x_train = train_data[:-nts_fcast,:]
    y_train = train_data[nts_fcast:,:]

    # Create a CATS object
    cat_l96 = CATS(K*J+K, K, J, **def_params)

    model, history = cat_l96.train(x_train,y_train,**hyperparams)

    print('Training time:',datetime.now()-start)

    #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    #+-+-+-+-+-+-+-+-+ Bootstrapped RMSE -- T_mr-alone CATs +-+-+-+-+-+-+-+-+-+-
    #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    n_tot = np.shape(x_train)[0]
    n_train = round(n_tot*0.8) # change 0.8 to the ratio used in the training
    n_test = n_tot-n_train  
    states_ANN = np.zeros((n_test,K*J+K)) # output using the imperfect model

    for kk in range(n_test):
        states_ANN[kk,:] = cat_l96.forecast(np.reshape(x_train[n_train+kk,:], 
                                                       [1,-1]), batch_size=1)
    
    states_ANN_scaled = scaler.inverse_transform(states_ANN)
    residual = states_twolevel[nts_fcast+n_train:
                               n_train+nts_fcast+n_test,:] - states_ANN_scaled
    print('CATs RMSE:',np.sqrt(np.nanmean(residual[:,:K]**2, axis=(0,1))))
    
    rmse_bs_tmr_alone[nn,:] = bootstrap(residual,size=bs_size)
    del states_ANN

# + -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# + -+-+-+-+-+-+-+-+-+-+-+-+- Save bootstrapped RMSEs +-+-+-+-+-+-+-+-+-+-+-+-+-+-
# + -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
try:ncfile.close()
except:pass  

fname = f"CATs_L96_RMSE_vs_Neurons_Tmr-alone_"+ \
f"1HL_{int(round(n_neurons[0]))}-{int(round(n_neurons[-1]))}Neurons"+\
f"_tau={n_mtu}_LRcorrected.nc"
ncfile = nc.Dataset(fname,mode='w',format='NETCDF4')

# Add dimensions
nneurons = ncfile.createDimension('nneurons',len(n_neurons))
nbs = ncfile.createDimension('nbs',bs_size)

# Add title
ncfile.title='RMSE vs Neurons Dataset'

# Add variables
neurons = ncfile.createVariable('neurons', np.float32, ('nneurons',))
neurons.long_name = 'Number of Neurons'

rmse_tmr_alone = ncfile.createVariable('rmse_tmr_alone',\
                                        np.float32, ('nneurons','nbs'))
rmse_tmr_alone.long_name = 'RMSE for T_mr-alone configuration '


# write the data
neurons[:] = n_neurons
rmse_tmr_alone[:,:] = rmse_bs_tmr_alone

# close
ncfile.close()
