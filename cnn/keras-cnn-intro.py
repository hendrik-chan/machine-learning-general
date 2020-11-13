#https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/

# example of calculation 1d convolutions
from numpy import asarray
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D

'''
	○ Using keras for 1D Convolutional layer
		○ Let's say we have 1-d input with 8 elements. 
		○ Keras requires input to be 3-D for a 1-D convolutional layer
			§ First dimension: number of input samples : we only have 1
			§ Second dimension: length of sample: we have 8 elements
			§ Third dimension: number of channels: we have 1
		○ Create 1 1-D conv layer. Filter = 1, shape = 3 (note normally its 3x3 here).
		input_shape is 8x1 (why ah?)
		# create model
		model = Sequential()
		model.add(Conv1D(1, 3, input_shape=(8, 1)))
		○ We manually define the weights here, as we want to detect vertical bumps in data. High input value surrounded by low values.
		○ Weight must be row x height x channel
		# define a vertical line detector
		weights = [asarray([[[0]],[[1]],[[0]]]), asarray([0.0])]
		# store the weights in the model
		model.set_weights(weights)
		○ If we print the weights:
		# confirm they were stored
        print(model.get_weights())

        ○ Note: Note that the feature map has six elements, whereas our input has eight elements. This is an artefact of how the filter was applied to the input sequence. 
        ::we can also use SAME padding
'''

# define input data
data = asarray([0, 0, 0, 1, 1, 0, 0, 0])
data = data.reshape(1, 8, 1)
# create model
model = Sequential()
model.add(Conv1D(1, 3, input_shape=(8, 1)))
# define a vertical line detector
weights = [asarray([[[0]],[[1]],[[0]]]), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
# confirm they were stored
print(model.get_weights())


# apply filter to input data
#		○ We use. predict to perform the convolution
yhat = model.predict(data)
print(yhat)


print('two-d convolutional layer \n')
# define input data
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = asarray(data)
data = data.reshape(1, 8, 8, 1)


from tensorflow.keras.layers import Conv2D
# define input data
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data = asarray(data)
data = data.reshape(1, 8, 8, 1)
# create model
model = Sequential()
model.add(Conv2D(1, (3,3), input_shape=(8, 8, 1)))
# define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
# confirm they were stored
print(model.get_weights())
# apply filter to input data
yhat = model.predict(data)
for r in range(yhat.shape[1]):
	# print each column in the row
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])