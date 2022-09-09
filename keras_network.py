from keras.datasets.mnist import load_data
from keras.layers import Conv2D, Flatten, Dense
from keras.losses import mse
from keras.models import Sequential
from keras.optimizers import Adam


def reformat(array):
    result = []
    for arr in array:
        for i in range(10):
            if arr[i] == 1.0:
                result.append(i)
    return result


(trainX, trainy), (testX, testy) = load_data()

n_samples = 10000
trainX, trainy = trainX[:n_samples].reshape(n_samples, 28, 28, 1), trainy[:n_samples]
# print(trainX.shape)
# print(trainy.shape)
model = Sequential()

kernel_size = 2
filters = 3

model.add(Conv2D(
    kernel_size=kernel_size,
    filters=filters,
    activation='relu',
    input_shape=(28, 28, 1),
))
model.add(Conv2D(
    kernel_size=kernel_size,
    filters=filters,
    activation='relu',
))
model.add(Conv2D(
    kernel_size=kernel_size,
    filters=filters,
    activation='relu',
))
model.add(Conv2D(
    kernel_size=kernel_size,
    filters=filters,
    activation='relu'),
)
# model.add(Conv2D(kernel_size=5, filters=50, activation='relu'))
# model.add(Conv2D(kernel_size=5, filters=50, activation='relu'))
# model.add(Conv2D(kernel_size=5, filters=50, activation='relu'))
model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer=Adam(), loss=mse)

# model.summary()
for i in range(4):
    filters, biases = model.layers[i].get_weights()
    print(f"Layer: {i}")
    print(filters)
    print()
# print(model.get_weights())
# print("weight paths:")
# print(model.get_weight_paths())
# print("weight paths over")
# model.fit(x=trainX, y=trainy, epochs=10)

# model.summary()

num_predictions = 2
pred = model.predict(testX[:num_predictions].reshape(num_predictions, 28, 28))

# print(reformat(pred))
# print(testy[:num_predictions])
