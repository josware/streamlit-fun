import streamlit as st

st.title('Customizable Neural Network')


num_neurons = st.sidebar.slider('Number of neurons in hidden layer:', 1, 64)
num_epochs = st.sidebar.slider('Number of epochs:', 1, 10)
str_activation = st.sidebar.text_input('Activation Function')

str_activation = ("relu", str_activation)[str_activation == ""]


#"The number of neurons is " + str(num_neurons)
#"The number of epochs is " + str(num_epochs)
#"The activation function is " + str_activation


if st.button('Train the model'):
    'Model is training...'
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import ModelCheckpoint
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    def preprocess_image(images):
        images = images / 255
        return images

    X_train = preprocess_image(X_train)
    X_test = preprocess_image(X_test)

    model = Sequential()
    model.add(InputLayer((28,28)))
    model.add(Flatten())
    model.add(Dense(num_neurons,str_activation))
    model.add(Dense(10)) #10 classes
    model.add(Softmax()) #get probability 
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    save_cp= ModelCheckpoint('model', save_best_only=True)
    history_cp = tf.keras.callbacks.CSVLogger('history.csv', separator=',')
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, callbacks=[save_cp, history_cp])

if st.button('Evaluate the model'):
    'Model is evaluating...'
    import pandas as pd
    import matplotlib.pyplot as plt

    history = pd.read_csv('history.csv')
    fig = plt.figure()
    plt.plot(history['epoch'], history['accuracy'])
    plt.plot(history['epoch'], history['val_accuracy'])
    plt.title('Model accuracy vs epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train','Val'])
    fig
