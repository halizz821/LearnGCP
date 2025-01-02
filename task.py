#
"""This training script trains binary classifier on Sentinel-2 satellite images.
The model is a fully convolutional neural network that predicts whether a power
plant is turned on or off.

A Sentinel-2 image consists of 13 bands. Each band contains the data for a
specific range of the electromagnetic spectrum.

A JPEG image consists of three channels: Red, Green, and Blue. For Sentinel-2
images, these correspond to Band 4 (red), Band 3 (green), and Band 2 (blue).
These bands contain the raw pixel data directly from the satellite sensors.
For more information on the Sentinel-2 dataset:
https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2
"""

from __future__ import annotations

import argparse

import tensorflow as tf

#################################################### 
#These define the input features (BANDS), the target label (LABEL), and batch size for data processing 
#and model training.

BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]
LABEL = "is_powered_on"
BATCH_SIZE = 64

#########################################################################
#Purpose: Accepts a --bucket argument for specifying the GCS bucket where the TFRecords
#are stored and where the trained model will be saved.
def get_args() -> dict:
    """Parses args."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, type=str, help="GCS Bucket")
    args = parser.parse_args()
    return args


def parse_tfrecord(example_proto: bytes, features_dict: dict) -> dict:
    # Reads and parses individual TFRecord examples (a single Example) into a dictionary of features and labels.
    # TFRecord files store serialized tf.train.Example objects.
    # Each Example contains structured data (e.g., image pixels, labels) in a compact, binary format.
    """Parses a single tf.train.Example."""

    return tf.io.parse_single_example(example_proto, features_dict) #is used to parse a single serialized 
                                                                    #tf.train.Example of TFRecord into a dictionary
                                                                    #of tensors, based on a specified feature schema defined in
                                                                    #features_dict .


def create_features_dict() -> dict:
    """Creates dict of features."""
    # Defines how each band and the label are formatted for TensorFlow. In other words, it help to parse the serialized
    #Examples and convert it to a format that can be proxessed by TensorFlow.
    #For example,
    #"B1": tf.io.FixedLenFeature(shape=[33, 33], dtype=tf.float32)
    # This represents a 33x33 patch of spectral data for Band 1.
    features_dict = {
        name: tf.io.FixedLenFeature(shape=[33, 33], dtype=tf.float64) for name in BANDS
    }

    features_dict[LABEL] = tf.io.FixedLenFeature(shape=[1, 1], dtype=tf.float64)

    return features_dict


def get_feature_and_label_vectors(
    inputs: dict, features_dict: dict
) -> tuple[tf.Tensor, int]:
    """Formats data."""
    
    # This code extracts and processes features and labels from a
    # parsed input dictionary, preparing them for use in machine learning pipelines
    # the parsed data into a usable format for TensorFlow:
    # Transposes band data to (x, y, bands) for convolutional layers.
    # Extracts the label (is_powered_on).

    label_value = tf.cast(inputs.pop(LABEL), tf.int32)
    #inputs.pop(LABEL) get and Removes the value associated with the key LABEL from the dictionary inputs.
    #After this, the label is no longer part of inputs
    #tf.cast(..., tf.int32) Converts the extracted label to the tf.int32 data type, ensuring compatibility 
    #with TensorFlow operations.
    
    features_vec = [inputs[name] for name in BANDS]
    # (bands, x, y) -> (x, y, bands)
    features_vec = tf.transpose(features_vec, [1, 2, 0])
    return features_vec, label_value


############################ Dataset Creation

def create_datasets(bucket: str) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Creates training and validation datasets."""

    train_data_dir = f"gs://{bucket}/geospatial_training.tfrecord.gz"
    eval_data_dir = f"gs://{bucket}/geospatial_validation.tfrecord.gz"
    features_dict = create_features_dict()

    training_dataset = (
        tf.data.TFRecordDataset(train_data_dir, compression_type="GZIP")
        .map(lambda example_proto: parse_tfrecord(example_proto, features_dict)) # parse each Example (patch of image)
        .map(lambda inputs: get_feature_and_label_vectors(inputs, features_dict)) # convert the parsed example to a format liked bt tensorflow
        .batch(64)
    )

    validation_dataset = (
        tf.data.TFRecordDataset(eval_data_dir, compression_type="GZIP")
        .map(lambda example_proto: parse_tfrecord(example_proto, features_dict))
        .map(lambda inputs: get_feature_and_label_vectors(inputs, features_dict))
        .batch(64)
    )

    return training_dataset, validation_dataset

##############################################################################################
#Model Definition


def create_model(training_dataset: tf.data.Dataset) -> tf.keras.Model:
    """Creates model."""

    feature_ds = training_dataset.map(lambda x, y: x) # ingore y=label
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
    normalizer.adapt(feature_ds) # computes the mean and variance of the features in feature_ds

    inputs = tf.keras.Input(shape=(None, None, 13)) # None in the shape parameter of tf.keras.Input makes the model
                                                    # flexible to handle inputs with variable spatial dimensions.
    x = normalizer(inputs)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=33, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    args = get_args()
    training_dataset, validation_dataset = create_datasets(args.bucket)
    model = create_model(training_dataset)
    model.fit(training_dataset, validation_data=validation_dataset, epochs=20)
    model.save(f"gs://{args.bucket}/model_output")  # save the model in the bucket


if __name__ == "__main__":
    main()
