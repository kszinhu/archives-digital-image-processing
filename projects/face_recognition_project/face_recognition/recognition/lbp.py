from .base import Recognizer


from typing import Any, Dict
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread
from skimage.transform import resize
from numpy import array, random, dstack

import matplotlib.pyplot as plt
import cv2 as cv
import pdb


class LBPRecognizer(Recognizer):
    name = "LBP"
    __recognizer = cv.face.LBPHFaceRecognizer_create()

    def train(self, x_train: Any, y_train: Any):
        self.__recognizer.train(x_train, y_train)

    def predict(self, x_test: Any):
        for image in x_test:
            prediction, confidence = self.__recognizer.predict(image)
            yield prediction, confidence

    def evaluate(self, output: bool = True, save_model: bool = False) -> Dict[str, float]:
        faces = []
        labels = []
        metrics = {"f1_score": [], "accuracy_score": []}

        data = self.dataset._loaded_dataset if self.dataset._loaded_dataset else self.dataset.load_dataset()
        for image_path, label in data:
            readable_image = imread(image_path)
            faces.append(readable_image)
            labels.append(int(label))

        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        for i in range(10):
            x_train, x_test, y_train, y_test = self._extract(random_state=i)

            self.train(x_train, y_train)

            predictions, confidence = [], []

            for prediction, conf in self.predict(x_test):
                predictions.append(prediction)
                confidence.append(conf)

            metrics["f1_score"].append(f1_score(y_test, array(predictions), average="macro"))
            metrics["accuracy_score"].append(accuracy_score(y_test, array(predictions)))

            if output:
                sample = random.randint(0, len(x_test))
                # grab the face image with predicted name and display it, x_test is numpy array
                readable_image = dstack([x_test[sample]] * 3)
                readable_image = resize(readable_image, (250, 250))

                predicted_name = label_encoder.inverse_transform([predictions[sample]])[0]
                real_name = label_encoder.classes_[y_test[sample]]

                plt.title(f"Predicted: {predicted_name}, Actual: {real_name}")
                plt.imshow(readable_image, cmap="gray")
                plt.show()

        if save_model:
            """
            Note: The model is saved in the database folder and size is large (around ~2 GB)
            """

            if self.dataset._database_path is None:
                raise ValueError("You must provide a path for the database")
            self.__recognizer.save(self.dataset._database_path / "face-recognizer-LBP-model.yml")

        mean_metrics = {key: sum(metric_values) / len(metric_values) for key, metric_values in metrics.items()}

        return mean_metrics
