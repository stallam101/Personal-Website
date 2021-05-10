import numpy as np
import cv2
from flask import Flask, render_template, request, redirect
import keras

app = Flask(__name__)

@app.route("/")
def index():
        return render_template("index.html")

@app.route("/train", methods = ["GET", "POST"])
def train():
    data=[]
    if request.method == "POST":
        allimg = request.files.getlist("img")
        for img in allimg:
            imgarr = np.fromstring(img.read(), np.uint8)
            imgarr = cv2.imdecode(imgarr, cv2.IMREAD_COLOR)
            resimgarr = cv2.resize(imgarr, (100,100))
            data.append(resimgarr)
        data = np.array(data)

        labels = request.form.getlist("label")
        labels = np.array(labels).astype(np.int32)

        data = data / 255

        model = keras.Sequential([
            keras.layers.Conv2D(filters=32, kernel_size=(3,3), input_shape=(100,100,3)),
            keras.layers.Maxpooling2D(pool_size=(2,2)),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(data, labels, epochs=5)
        predimg = request.files["predimg"]
        predarr = np.fromstring(predimg.read(), np.uint8)
        predarr = cv2.imdecode(predarr, cv2.IMREAD_COLOR)
        respred = cv2.resize(predarr, (100, 100))
        predtest = np.reshape(respred, (1, 100, 100, 3))

        finalpred = model.predict(predtest)
        for finalpredictions in finalpred:
            if finalpredictions[0] > finalpredictions[1]:
                return request.form["indicate1"]
            else:
                return request.form["indicate0"]

    else:
        return render_template("train.html")



if __name__ == "__main__":
    app.run()