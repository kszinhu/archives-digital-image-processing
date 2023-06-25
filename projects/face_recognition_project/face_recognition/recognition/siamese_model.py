from face_recognition.dataset import Dataset
from .base import Recognizer
from face_recognition.metrics import Metrics
from face_recognition.utils import secho

from typing import Any, Dict, Tuple, List
from numpy import unique, where, array, ndarray, random, mean
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread
from keras import backend as k
from keras.layers import Input, Flatten, Dropout, Dense, Lambda
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import plot_model

import tensorflow as tf
import matplotlib.pyplot as plt
import pdb

rms = RMSprop()
# Fix: https://github.com/slundberg/shap/issues/1907#issuecomment-1169377709
tf.compat.v1.disable_eager_execution()


class SiameseRecognizer(Recognizer):
    name = "Siamese"
    __recognizer: Model = None  # type: ignore
    __metricCalculator = Metrics()

    def __init__(self, dataset: Dataset, **kwargs: Dict[str, Any]):
        super().__init__(dataset, **kwargs)

        # get the shape of dataset
        sample_image = (
            imread(self.dataset._loaded_dataset[0][0])
            if self.dataset._loaded_dataset
            else imread(self.dataset.load_dataset()[0][0])
        )

        self.__recognizer = self.__create_model(input_shape=sample_image.shape)
        self.__recognizer.compile(optimizer=rms, loss=self.__contrastive_loss_with_margin(margin=1))

    def __create_model(self, input_shape: Tuple[int, int]) -> Any:
        input = Input(shape=input_shape, name="base_input")

        x = Flatten(name="flatten_input")(input)
        x = Dense(128, activation="relu", name="first_base_dense")(x)
        x = Dropout(0.3, name="first_dropout")(x)
        x = Dense(128, activation="relu", name="second_base_dense")(x)
        x = Dropout(0.3, name="second_dropout")(x)
        x = Dense(128, activation="relu", name="third_base_dense")(x)

        # create the base model with layers shared by both branches
        base_model = Model(inputs=input, outputs=x)

        # add input layer for each of the two branches of the network
        # receives pairs[:, 0] and pairs[:, 1] less one dimension
        input_left = Input(input_shape)
        input_right = Input(input_shape)

        vectors_output_left = base_model(input_left)
        vectors_output_right = base_model(input_right)

        # finally, add the euclidean distance layer
        output = Lambda(
            self.__euclidean_distance, name="output_layer", output_shape=self.__euclidean_distance_output_shape
        )([vectors_output_left, vectors_output_right])

        # assemble the final model
        model = Model(inputs=[input_left, input_right], outputs=output)

        # plot the model
        plot_model(model, to_file="siamese_model_plot.png", show_shapes=True)

        return model

    def train(self, x_train: Any, y_train: Any):
        pairs, labels = self.__create_pairs(x_train, y_train)
        return self.__recognizer.fit([pairs[:, 0], pairs[:, 1]], labels, epochs=100)

    def predict(self, x_test: Any, y_test: Any = None) -> Any:
        pairs, _labels = self.__create_pairs(x_test, y_test)
        return self.__recognizer.predict([pairs[:, 0], pairs[:, 1]])

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

            predictions = self.predict(x_test, y_test)

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

            if output:
                plt.figure(figsize=(10, 10))
                plt.subplot(1, 2, 1)
                plt.title("Actual")
                plt.imshow(x_test[0].reshape(64, 64), cmap="gray")

                plt.subplot(1, 2, 2)
                plt.title("Predicted")
                plt.imshow(x_test[1].reshape(64, 64), cmap="gray")

                plt.show()

        if save_model:
            if self.dataset._database_path is None:
                raise ValueError("You must provide a path for the database")

            self.__recognizer.save(self.dataset._database_path / "face-recognizer-siamese-model.h5")

        mean_metrics = {key: sum(metric_values) / len(metric_values) for key, metric_values in metrics.items()}

        return mean_metrics

    def __create_pairs(self, images_dataset: List[Any], labels_dataset: List[Any]) -> Tuple[ndarray, ndarray]:
        unique_labels = unique(labels_dataset)
        label_wise_indices = dict()
        for label in unique_labels:
            label_wise_indices.setdefault(
                label, [index for index, curr_label in enumerate(labels_dataset) if label == curr_label]
            )

        pair_images = []
        pair_labels = []
        for index, image in enumerate(images_dataset):
            pos_indices = label_wise_indices.get(labels_dataset[index])
            pos_image = images_dataset[random.choice(pos_indices)]  # type: ignore
            pair_images.append((image, pos_image))
            pair_labels.append(1)

            neg_indices = where(labels_dataset != labels_dataset[index])
            neg_image = images_dataset[random.choice(neg_indices[0])]
            pair_images.append((image, neg_image))
            pair_labels.append(0)
        return array(pair_images), array(pair_labels)

    def __euclidean_distance(self, vectors):
        (featA, featB) = vectors
        sum_squared = k.sum(k.square(featA - featB), axis=1, keepdims=True)
        return k.sqrt(k.maximum(sum_squared, k.epsilon()))

    def __euclidean_distance_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def __contrastive_loss_with_margin(self, margin):
        def contrastive_loss(y_true, y_pred):
            square_pred = k.square(y_pred)
            margin_square = k.square(k.maximum(margin - y_pred, 0))
            return y_true * square_pred + (1 - y_true) * margin_square

        return contrastive_loss
