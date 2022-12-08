# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")
# import the necessary packages
from preprocessing import load_az_dataset, load_mnist_dataset
from tensorflow.keras.optimizers.legacy import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from CNN import FirstCNN

if __name__ == "__main__":

    # load the A-Z and MNIST datasets, respectively
    print("[INFO] loading datasets...")
    (azData, azLabels) = load_az_dataset('A_Z Handwritten Data.csv')
    (digitsData, digitsLabels) = load_mnist_dataset()

    # DATA FORMATTING
    # 0-9 are indexes 0-9 and a-z are indexes 10-32
    azLabels += 10
    data = np.vstack([azData, digitsData])
    labels = np.hstack([azLabels, digitsLabels])
    # resize to 32x32 size images
    data = [cv2.resize(image, (32, 32)) for image in data]
    data = np.array(data, dtype="float32")
    # set images to b/w and add an extra column for bias
    data = np.expand_dims(data, axis=-1)
    data /= 255.0
    labelNames = ["0","1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C","D", "E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

    # convert the labels from integers to vectors
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    (trainX, testX, trainY, testY) = train_test_split(data,
                                                      labels, test_size=0.20, stratify=labels, random_state=42)
    #INITIALIZATION
    print("[INFO] compiling model")
    opt = SGD(lr=0.01)
    model = FirstCNN.build(32, 32, 1, 36)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics =["accuracy"])

    #TRAINING
    print("[INFO] training network...")
    H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=5, verbose=1)

    #SAVING
    print("[INFO] serializing network")
    model.save("alphanumeric_weights.hdf5")

    #EVALUATION
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 5), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 5), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 5), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, 5), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()