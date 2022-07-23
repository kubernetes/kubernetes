# Predicting Income with the Census Income Dataset
## Overview
The [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/Census+Income) contains over 48,000 samples with attributes including age, occupation, education, and income (a binary label, either `>50K` or `<=50K`). The dataset is split into roughly 32,000 training and 16,000 testing samples.

Here, we use the [wide and deep model](https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html) to predict the income labels. The **wide model** is able to memorize interactions with data with a large number of features but not able to generalize these learned interactions on new data. The **deep model** generalizes well but is unable to learn exceptions within the data. The **wide and deep model** combines the two models and is able to generalize while learning exceptions.

For the purposes of this example code, the Census Income Data Set was chosen to allow the model to train in a reasonable amount of time. You'll notice that the deep model performs almost as well as the wide and deep model on this dataset. The wide and deep model truly shines on larger data sets with high-cardinality features, where each feature has millions/billions of unique possible values (which is the specialty of the wide model).

---

The code sample in this directory uses the high level `tf.estimator.Estimator` API. This API is great for fast iteration and quickly adapting models to your own datasets without major code overhauls. It allows you to move from single-worker training to distributed training, and it makes it easy to export model binaries for prediction.

The input function for the `Estimator` uses `tf.contrib.data.TextLineDataset`, which creates a `Dataset` object. The `Dataset` API makes it easy to apply transformations (map, batch, shuffle, etc.) to the data. [Read more here](https://www.tensorflow.org/programmers_guide/datasets).

The `Estimator` and `Dataset` APIs are both highly encouraged for fast development and efficient training.

## Running the code
First make sure you've [added the models folder to your Python path](/official/#running-the-models); otherwise you may encounter an error like `ImportError: No module named official.wide_deep`.

### Setup
The [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/Census+Income) that this sample uses for training is hosted by the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/). We have provided a script that downloads and cleans the necessary files.

```
python data_download.py
```

This will download the files to `/tmp/census_data`. To change the directory, set the `--data_dir` flag.

### Training
You can run the code locally as follows:

```
python wide_deep.py
```

The model is saved to `/tmp/census_model` by default, which can be changed using the `--model_dir` flag.

To run the *wide* or *deep*-only models, set the `--model_type` flag to `wide` or `deep`. Other flags are configurable as well; see `wide_deep.py` for details.

The final accuracy should be over 83% with any of the three model types.

### TensorBoard

Run TensorBoard to inspect the details about the graph and training progression.

```
tensorboard --logdir=/tmp/census_model
```

## Inference with SavedModel
You can export the model into Tensorflow [SavedModel](https://www.tensorflow.org/programmers_guide/saved_model) format by using the argument `--export_dir`:

```
python wide_deep.py --export_dir /tmp/wide_deep_saved_model
```

After the model finishes training, use [`saved_model_cli`](https://www.tensorflow.org/programmers_guide/saved_model#cli_to_inspect_and_execute_savedmodel) to inspect and execute the SavedModel.

Try the following commands to inspect the SavedModel:

**Replace `${TIMESTAMP}` with the folder produced (e.g. 1524249124)**
```
# List possible tag_sets. Only one metagraph is saved, so there will be one option.
saved_model_cli show --dir /tmp/wide_deep_saved_model/${TIMESTAMP}/

# Show SignatureDefs for tag_set=serve. SignatureDefs define the outputs to show.
saved_model_cli show --dir /tmp/wide_deep_saved_model/${TIMESTAMP}/ \
    --tag_set serve --all
```

### Inference
Let's use the model to predict the income group of two examples:
```
saved_model_cli run --dir /tmp/wide_deep_saved_model/${TIMESTAMP}/ \
--tag_set serve --signature_def="predict" \
--input_examples='examples=[{"age":[46.], "education_num":[10.], "capital_gain":[7688.], "capital_loss":[0.], "hours_per_week":[38.]}, {"age":[24.], "education_num":[13.], "capital_gain":[0.], "capital_loss":[0.], "hours_per_week":[50.]}]'
```

This will print out the predicted classes and class probabilities. Class 0 is the <=50k group and 1 is the >50k group.

## Additional Links

If you are interested in distributed training, take a look at [Distributed TensorFlow](https://www.tensorflow.org/deploy/distributed).

You can also [run this model on Cloud ML Engine](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction), which provides [hyperparameter tuning](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction#hyperparameter_tuning) to maximize your model's results and enables [deploying your model for prediction](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction#deploy_a_model_to_support_prediction).
