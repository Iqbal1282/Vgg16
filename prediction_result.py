from utils import load_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
import tensorflow as tf 
import numpy as np 




feature, label = load_data()
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.1, shuffle=True)





categories = ['daisy', 'dandelion', 'rose','sunflower', 'tulip']

model = tf.keras.models.load_model('myVggModel.h5')


prediction = model(x_test[0:9])

plt.figure(figsize=(8,8))

for i in range(9):
	plt.subplot(3,3,i+1)
	plt.imshow(x_test[i])
	plt.xlabel('Pedicted:%s\n Actual: %s'%(categories[np.argmax(prediction[i])],
		categories[y_test[i]]))

	plt.xticks([])

plt.show()

