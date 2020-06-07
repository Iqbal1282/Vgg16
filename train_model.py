from utils import load_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
import tensorflow as tf 




feature, label = load_data()
categories = ['daisy', 'dandelion', 'rose','sunflower', 'tulip']

x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.1, shuffle=True)

x_train, x_test = tf.cast(x_train, tf.float32), tf.cast(x_test, tf.float32)

train_dataset= tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = train_dataset.batch(batch_size= 1)




model = tf.keras.models.load_model('myVggModel.h5')

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adadelta()


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


@tf.function
def train_step(images, labels):
	with tf.GradientTape() as tape:
		predictions = model(images, training= True)
		loss = loss_object(y_true= labels, y_pred= predictions)

	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(grads_and_vars = zip(gradients, model.trainable_variables))

	train_loss(loss)
	train_accuracy(labels, predictions)



for epoch in range(1):
	train_loss.reset_states()
	train_accuracy.reset_states()

	step = 0

	for images, labels in train_dataset:
		step+=1 
		train_step(images, labels)

		if step%10 ==0:

			print('=> epoch: %i, loss: %.4f, train_accuracy: %.4f'%(epoch+1,
				train_loss.result(), train_accuracy.result()))



model.save('myVggModel.h5')



