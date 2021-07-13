from keras.models import Model
from keras.layers import Dense,Activation,Input

def body_net(num_joints = 5):
    latent_input = Input(2,num_joints)
    x = Dense(10)(latent_input)
    x = Dense(2)(x)
    x = Activation('tanh')(x)
    model = Model(latent_input, x)
    return model

def motor_net(num_joints=5):
    motor_input = Input(1,num_joints)
    x = Dense(10)(motor_input)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    model = Model(motor_input, x)
    return model