# Import necessary libraries
"""Imports the necessary libraries required for the project"""
import datasets
import pandas
import transformers
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer

# Use DistillROBERTa's tokenizer to convert the tokens into dense vectors
tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")

def tokenize(examples):
    """Converts the text of each example to "input_ids", a sequence of integers
    representing 1-hot vectors for each token in the text"""
    return tokenizer(examples["text"],
                     truncation=True,
                     max_length=64,
                     padding="max_length")
# Define the train function
def train(model_path="model.keras", train_path="train.csv", dev_path="dev.csv"):
    """Trains the model on the training set using the parameters specified"""
    # We convert the csv files to higgingface datasets
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path, "validation": dev_path})

    # We extract the labels
    labels = hf_dataset["train"].column_names[1:]
    num_labels = len(labels)

    print(f"Training model with {num_labels} labels: {labels}")

    def gather_labels(example):
        # We convert the integer labels to floats
        return {"labels": [float(example[l]) for l in labels]}

    # We map it to the dataset
    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    # Convert huggingface datasets into data frames for training and dev
    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="labels",
        batch_size=16,
        shuffle=True)
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="labels",
        batch_size=16)

    # We use DistillROBERTa model and define the type of problem and
    # the number of labels
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "distilroberta-base",
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )
    # Learning rate schedule for slowing down the learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=5e-6,
        decay_steps=1000,
        decay_rate=0.9
    )
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.metrics.F1Score(average="micro", threshold=0.35)]
    )

    # Fit the model
    # We use callbacks to stop over fitting 
    # We use a patience of 5 and use validation f1 score as the metric
    # We maximize the validation f1 score, hence mode= "max"
    # We after earlystopping we return the best weights
    model.fit(
        train_dataset,
        epochs=15,
        validation_data=dev_dataset,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_f1_score",
                patience=5,
                mode='max',
                min_delta=0.0001,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor="val_f1_score",
                mode="max",
                save_best_only=True,
                save_weights_only=True
            )])

def predict(model_path="model.keras", input_path="dev.csv"):
    """Predicts labels on new data using the trained model"""
    # First, let's determine the number of labels from the input file
    df = pandas.read_csv(input_path)
    # We substract 1 as we account for the "text" column
    num_labels = len(df.columns) - 1

    # We load the model
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "distilroberta-base",
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    # We load the weights
    try:
        model.load_weights(model_path)
    except Exception as e:
        print(f"Error loading weights: {e}")
        print(f"Expected number of labels: {num_labels}")
        return

    # Again use tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=64, padding="max_length")

    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    tf_dataset = hf_dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        batch_size=16,
        shuffle=False,
    )

    # Make predictions
    outputs = model.predict(tf_dataset)
    logits = outputs['logits']
    probs = tf.nn.sigmoid(logits)
    predictions = (probs.numpy() > 0.5).astype(int)

    # Update the predictions in the dataframe
    df.iloc[:, 1:] = predictions

    # Save the predictions to a zip file
    df.to_csv(
        "submission.zip",
        index=False,
        compression=dict(method='zip', archive_name='submission.csv')
    )

    # Created/Modified files during execution:
    for file_name in ["submission.zip"]:
        print(file_name)
# Train the model
train()
# Predict new data with the trained model
predict(input_path="test-in.csv")
