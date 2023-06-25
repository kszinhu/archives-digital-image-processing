from face_recognition.dataset import Dataset
from .base import Recognizer
from face_recognition.metrics import Metrics
from face_recognition.utils import secho

from typing import Any, Dict, Tuple, List, Optional
from numpy import unique, where, array, ndarray, random, mean
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread
from keras import backend as k
from keras.layers import (
    Input,
    Flatten,
    MaxPooling2D,
    Dropout,
    Dense,
    Lambda,
    BatchNormalization,
    Conv2D,
)
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import plot_model

import tensorflow as tf
import matplotlib.pyplot as plt
import pdb

# Fix: https://github.com/slundberg/shap/issues/1907#issuecomment-1169377709
# tf.compat.v1.disable_eager_execution()


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
        optimizer = RMSprop(lr=0.1)

        self.__recognizer = self.__create_model(input_shape=sample_image.shape)
        self.__recognizer.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        self.__recognizer.summary()

    def __create_model(self, input_shape: Tuple[int, int], model_name: Optional[str] = None) -> Any:
        input = Input(shape=(input_shape[0], input_shape[1], 1), name="input_layer")

        x = BatchNormalization()(input)
        x = Conv2D(64, kernel_size=3, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=3, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.2)(x)

        x = Conv2D(128, kernel_size=3, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, kernel_size=3, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.2)(x)

        x = Conv2D(256, kernel_size=3, activation="relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.2)(x)

        x = Flatten()(x)
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Dense(5, activation="softmax")(x)

        # create the base model with layers shared by both branches
        base_model = Model(inputs=input, outputs=x, name=model_name)

        # add input layer for each of the two branches of the network
        # receives pairs[:, 0] and pairs[:, 1] less one dimension
        input_left = Input(input_shape)
        input_right = Input(input_shape)

        tower_1 = base_model(input_left)
        tower_2 = base_model(input_right)

        # finally, add the euclidean distance layer
        merge_layer = Lambda(self.__euclidean_distance, name="merge_layer")([tower_1, tower_2])
        normal_layer = BatchNormalization()(merge_layer)
        output_layer = Dense(1, activation="sigmoid")(normal_layer)

        # assemble the final model
        siamese_model = Model(inputs=[input_left, input_right], outputs=output_layer)

        # plot the model
        plot_model(siamese_model, to_file="siamese_model_plot.png", show_shapes=True)

        return siamese_model

    def train(self, x_train: Any, y_train: Any):
        pairs, labels = self.__create_pairs(x_train, y_train)
        return self.__recognizer.fit([pairs[:, 0], pairs[:, 1]], labels, batch_size=256, epochs=20)

    def predict(self, x_test: Any, y_test: Any = None) -> Any:
        pairs, labels = self.__create_pairs(x_test, y_test)
        return self.__recognizer.evaluate([pairs[:, 0], pairs[:, 1]], labels)

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

        for current_random_state in range(1):
            secho(f"Training model with random state: {current_random_state}", message_type="INFO")
            x_train, x_test, y_train, y_test = self._extract(random_state=current_random_state, split_only_test=False)

            t_pred = self.train(x_train, y_train)

            if output:
                self.__plt_metric(t_pred.history, metric="accuracy", title="Model accuracy")

            predictions = self.predict(x_test, y_test)

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

    def __plt_metric(self, history, metric, title, has_valid=True):
        """Plots the given 'metric' from 'history'.

        Arguments:
            history: history attribute of History object returned from Model.fit.
            metric: Metric to plot, a string value present as key in 'history'.
            title: A string to be used as title of plot.
            has_valid: Boolean, true if valid data was passed to Model.fit else false.

        Returns:
            None.
        """
        plt.plot(history[metric])
        if has_valid:
            plt.plot(history[metric])
            plt.legend(["train"], loc="upper left")
        plt.title(title)
        plt.ylabel(metric)
        plt.xlabel("epoch")
        plt.show()
