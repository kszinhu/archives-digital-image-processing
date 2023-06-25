from .base import Recognizer
from face_recognition.metrics import Metrics
from face_recognition.utils import secho

from typing import Any, Dict
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread
from skimage.transform import resize
from numpy import random, dstack

import matplotlib.pyplot as plt
import cv2 as cv


class LBPRecognizer(Recognizer):
    name = "LBP"
    __recognizer = cv.face.LBPHFaceRecognizer_create()
    __metricCalculator = Metrics()

    def train(self, x_train: Any, y_train: Any):
        self.__recognizer.train(x_train, y_train)

    def predict(self, x_test: Any):
        for image in x_test:
            prediction, confidence = self.__recognizer.predict(image)
            yield prediction, confidence

    def evaluate(self, output: bool = True, save_model: bool = False) -> Dict[str, float]:
        faces = []
        labels = []
        metrics = {}

        data = self.dataset._loaded_dataset if self.dataset._loaded_dataset else self.dataset.load_dataset()
        for image_path, label in data:
            readable_image = imread(image_path)
            faces.append(readable_image)
            labels.append(int(label))

        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        for current_random_state in range(10):
            secho(f"Training model with random state: {current_random_state}", message_type="INFO")
            x_train, x_test, y_train, y_test = self._extract(random_state=current_random_state, split_only_test=False)

            self.train(x_train, y_train)

            predictions, confidence = [], []

            for prediction, conf in self.predict(x_test):
                predictions.append(prediction)
                confidence.append(conf)

            for metric, value in self.__metricCalculator.evaluate(
                y_test,
                predictions,
                requested_metrics=[
                    ("accuracy_score", None),
                    ("precision_score", None),
                    ("recall_score", None),
                    ("f1_score", None),
                ],
            ):
                metrics[metric] = metrics.get(metric, []) + [value]

            secho(f"accuracy_score: {metrics['accuracy_score'][-1]*100}%", message_type="INFO")
            if metrics["accuracy_score"][-1] != 1.0:
                wrong_prediction = random.choice(
                    [i for i, prediction in enumerate(predictions) if prediction != y_test[i]]
                )
                predicted_name = label_encoder.classes_[predictions[wrong_prediction]]
                real_name = label_encoder.classes_[y_test[wrong_prediction]]

                secho(f"One wrong prediction: {predicted_name} instead of {real_name}", message_type="INFO")
                if output:
                    # show the image with wrong prediction and the actual prediction
                    readable_image = dstack([x_test[wrong_prediction]] * 3)

                    plt.title(f"Predicted: {predicted_name}, Actual: {real_name}")
                    plt.imshow(readable_image, cmap="gray")
                    plt.show(block=True)
            elif output:
                sample = random.randint(0, len(x_test))
                # grab the face image with predicted name and display it, x_test is numpy array
                readable_image = dstack([x_test[sample]] * 3)
                readable_image = resize(readable_image, (250, 250))

                predicted_name = label_encoder.classes_[predictions[sample]]
                real_name = label_encoder.classes_[y_test[sample]]

                plt.title(f"Predicted: {predicted_name}, Actual: {real_name}")
                plt.imshow(readable_image, cmap="gray")
                plt.show(block=True)

        if save_model:
            """
            Note: The model is saved in the database folder and size is large (around ~2 GB)
            """

            if self.dataset._database_path is None:
                raise ValueError("You must provide a path for the database")
            self.__recognizer.save(self.dataset._database_path / "face-recognizer-LBP-model.yml")

        mean_metrics = {key: sum(metric_values) / len(metric_values) for key, metric_values in metrics.items()}

        return mean_metrics
