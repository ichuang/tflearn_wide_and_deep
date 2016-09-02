tflearn_wide_and_deep
=====================

Pedagogical example realization of wide & deep networks, using
[TensorFlow](https://www.tensorflow.org/) and
[TFLearn](http://tflearn.org/).

(Also see: [Pedagogical example of seq2seq RNN](https://github.com/ichuang/tflearn_seq2seq))

This is a re-implementation of the google paper on [Wide & Deep
Learning for Recommender Systems](http://arxiv.org/abs/1606.07792),
using the combination of a wide linear model, and a deep feed-forward
neural network, for binary classification (image from the Tensorflow
Tutorial):

![wide_and_deep](https://www.tensorflow.org/versions/r0.10/images/wide_n_deep.svg)

This example realization is based on Tensorflow's [Wide and Deep Learning Tutorial](https://www.tensorflow.org/versions/r0.10/tutorials/wide_and_deep/index.html),
but implemented in [TFLearn](http://tflearn.org/).  Note that despite
the closeness of names, [TFLearn](http://tflearn.org/) is distinct
from TF.Learn (previously known as scikit flow, sometimes referred to
as
[tf.contrib.learn](https://www.tensorflow.org/versions/r0.9/tutorials/tflearn/index.html)).

This implementation explicitly presents the construction of layers in the deep part of the
network, and allows direct access to changing the layer architecture, and customization
of methods used for regression and optimization.

In contrast, the TF.Learn tutorial offers more sophistication, but
hides the layer architecture behind a black box function,
[tf.contrib.learn.DNNLinearCombinedClassifier](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/estimators/dnn_linear_combined.py#L41).


Basic Usage
===========

    usage: tflearn_wide+deep.py [-h] [--model_type MODEL_TYPE]
                                [--run_name RUN_NAME]
                                [--load_weights LOAD_WEIGHTS] [--n_epoch N_EPOCH]
                                [--snapshot_step SNAPSHOT_STEP]
                                [--wide_learning_rate WIDE_LEARNING_RATE]
                                [--deep_learning_rate DEEP_LEARNING_RATE]
                                [--verbose [VERBOSE]] [--noverbose]
    
    optional arguments:
      -h, --help            show this help message and exit
      --model_type MODEL_TYPE
                            Valid model types: {'wide', 'deep', 'wide+deep'}.
      --run_name RUN_NAME   name for this run (defaults to model type)
      --load_weights LOAD_WEIGHTS
                            filename with initial weights to load
      --n_epoch N_EPOCH     Number of training epoch steps
      --snapshot_step SNAPSHOT_STEP
                            Step number when snapshot (and validation testing) is
                            done
      --wide_learning_rate WIDE_LEARNING_RATE
                            learning rate for the wide part of the model
      --deep_learning_rate DEEP_LEARNING_RATE
                            learning rate for the deep part of the model
      --verbose [VERBOSE]   Verbose output
      --noverbose

Dataset
=======

The dataset is the same [Census income
data](https://archive.ics.uci.edu/ml/datasets/Census+Income) used in
Tensorflow's [Wide and Deep Learning
Tutorial](https://www.tensorflow.org/versions/r0.10/tutorials/wide_and_deep/index.html).
The goal is to predict whether a given individual has an income of
over 50,000 dollars or not, based on 5 continuous variables (`age`,
`education_num`, `capital_gain`, `capital_loss`, `hours_per_week`) and 9 categorical variables.  

We simplify the approach used for categorical variables, and do not
use sparse tensors or anything fancy; instead, for the sake of a
simple demonstration, we map category strings to integers, using
pandas, then use embedding layers (whose weights are learned by
training).  That part of the code is excerpted here:

```python
        cc_input_var = {}
        cc_embed_var = {}
        flat_vars = []
        for cc, cc_size in self.categorical_columns.items():
            cc_input_var[cc] = tflearn.input_data(shape=[None, 1], name="%s_in" % cc,  dtype=tf.int32)
            # embedding layers only work on CPU!  No GPU implementation in tensorflow, yet!
            cc_embed_var[cc] = tflearn.layers.embedding_ops.embedding(cc_input_var[cc],    cc_size,  8, name="deep_%s_embed" % cc)
            flat_vars.append(tf.squeeze(cc_embed_var[cc], squeeze_dims=[1], name="%s_squeeze" % cc))
```

Notice how TFLearn provides input layers, which automatically construct placeholders for input data feeds.

Layer Architecture
==================

The wide model is realized using a single fully-connected layer, with no bias, and width equal to the number of inputs:

```python
        network = tflearn.fully_connected(network, n_inputs, activation="linear", name="wide_linear", bias=False)	# x*W (no bias)
        network = tf.reduce_sum(network, 1, name="reduce_sum")	# batched sum, to produce logits
        network = tf.reshape(network, [-1, 1])
```

The deep model is realized with two fully connected layers, with an
input constructed by concatenating the wide inputs with the embedded
categorical variables:

```python
	n_nodes=[100, 50]
        network = tf.concat(1, [wide_inputs] + flat_vars, name="deep_concat")
        for k in range(len(n_nodes)):
            network = tflearn.fully_connected(network, n_nodes[k], activation="relu", name="deep_fc%d" % (k+1))
        network = tflearn.fully_connected(network, 1, activation="linear", name="deep_fc_output", bias=False)
```

For the combined wide+deep model, the probability that the outcome is
"1" (versus "0"), for input "x" is given by Equation 3 of the [google research
paper](http://arxiv.org/abs/1606.07792), as

![prediction_formula](https://github.com/ichuang/tflearn_wide_and_deep/raw/master/images/prediction_formula.png "")

Note that the wide and deep models share a single central bias variable:

```python
        with tf.variable_op_scope([wide_inputs], None, "cb_unit", reuse=False) as scope:
            central_bias = tflearn.variables.variable('central_bias', shape=[1],
                                                      initializer=tf.constant_initializer(np.random.randn()),
                                                      trainable=True, restore=True)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/cb_unit', central_bias)
```

The wide and deep networks are combined according to the formula:

```python
        wide_network = self.wide_model(wide_inputs, n_cc)
        deep_network = self.deep_model(wide_inputs, n_cc)
        network = tf.add(wide_network, deep_network)
        network = tf.add(network, central_bias, name="add_central_bias")
```

Regression is done separately for the wide and deep networks, and for the central bias:

```python
            trainable_vars = tf.trainable_variables()
            tv_deep = [v for v in trainable_vars if v.name.startswith('deep_')]
            tv_wide = [v for v in trainable_vars if v.name.startswith('wide_')]

            wide_network_with_bias = tf.add(wide_network, central_bias, name="wide_with_bias")
            tflearn.regression(wide_network_with_bias, 
                               placeholder=Y_in,
                               optimizer='sgd', 
                               loss='binary_crossentropy',
                               metric="binary_accuracy",
                               learning_rate=learning_rate[0],
                               validation_monitors=vmset,
                               trainable_vars=tv_wide,
                               op_name="wide_regression",
                               name="Y")

            deep_network_with_bias = tf.add(deep_network, central_bias, name="deep_with_bias")
            tflearn.regression(deep_network_with_bias, 
                               placeholder=Y_in,
                               optimizer='adam', 
                               loss='binary_crossentropy',
                               metric="binary_accuracy",
                               learning_rate=learning_rate[1],
                               trainable_vars=tv_deep,
                               op_name="deep_regression",
                               name="Y")

            tflearn.regression(network, 
                               placeholder=Y_in,
                               optimizer='adam', 
                               loss='binary_crossentropy',
                               metric="binary_accuracy",
                               learning_rate=learning_rate[0],	# use wide learning rate
                               trainable_vars=[central_bias],
                               op_name="central_bias_regression",
                               name="Y")
```

and the confusion matrix is computed at each valiation step, using a
validation monitor which pushes the result as a summary to
TensorBoard:

```python
        with tf.name_scope('Monitors'):
            predictions = tf.cast(tf.greater(network, 0), tf.int64)
            Ybool = tf.cast(Y_in, tf.bool)
            pos = tf.boolean_mask(predictions, Ybool)
            neg = tf.boolean_mask(predictions, ~Ybool)
            psize = tf.cast(tf.shape(pos)[0], tf.int64)
            nsize = tf.cast(tf.shape(neg)[0], tf.int64)
            true_positive = tf.reduce_sum(pos, name="true_positive")
            false_negative = tf.sub(psize, true_positive, name="false_negative")
            false_positive = tf.reduce_sum(neg, name="false_positive")
            true_negative = tf.sub(nsize, false_positive, name="true_negative")
            overall_accuracy = tf.truediv(tf.add(true_positive, true_negative), tf.add(nsize, psize), name="overall_accuracy")
        vmset = [true_positive, true_negative, false_positive, false_negative, overall_accuracy]
```

Performance Comparisons
=======================

How does wide-only compare with wide+deep, or, for that matter, with deep only?

Wide Model
----------

Run this for the wide model:

    python tflearn_wide_and_deep.py --verbose --n_epoch=2000 --model_type=wide --snapshot_step=500 --wide_learning_rate=0.0001 

The tensorboard plots should show the accuracy and loss, as well as the four confusion matrix entries, e.g.:

![tensorboard_confusion_matrix](https://github.com/ichuang/tflearn_wide_and_deep/raw/master/images/tensorboard_wide_only_cmat.png "")

The tail end of the console output should look something like this:

```
Training Step: 2000  | total loss: 0.82368
| wide_regression | epoch: 2000 | loss: 0.82368 - binary_acc: 0.7489 | val_loss: 0.58739 - val_acc: 0.7813 -- iter: 32561/32561
--
============================================================  Evaluation
  logits: (16281,), min=-2.59761142731, max=116.775054932
Actual IDV
0    12435
1     3846

Predicted IDV
0    14726
1     1555

Confusion matrix:
actual           0     1
predictions
0            11800  2926
1              635   920
```

Note that the accuracy is (920+11800)/16281 = 78.1%


Deep Model
----------

Run this:

    python tflearn_wide_and_deep.py --verbose --n_epoch=2000 --model_type=deep --snapshot_step=250 --run_name="deep_run" --deep_learning_rate=0.001

And the result should look something like:

```
Training Step: 2000  | total loss: 0.31951
| deep_regression | epoch: 2000 | loss: 0.31951 - binary_acc: 0.8515 | val_loss: 0.31093 - val_acc: 0.8553 -- iter: 32561/32561
--
============================================================  Evaluation
  logits: (16281,), min=-12.0320196152, max=4.89985847473
Actual IDV
0    12435
1     3846

Predicted IDV
0    12891
1     3390


Confusion matrix:
actual           0     1
predictions
0            11485  1406
1              950  2440

```

Giving a final accuracy of (2440+11485)/16281 = 85.53%

Wide+Deep Model
---------------

Now how does the combined model perform?  Run this:

    python tflearn_wide_and_deep.py --verbose --n_epoch=2000 --model_type=wide+deep --snapshot_step=250 \
        --run_name="wide+deep_run"  --wide_learning_rate=0.00001 --deep_learning_rate=0.0001 

And the output should give something like this:

```
Training Step: 2000  | total loss: 1.33436
| wide_regression | epoch: 1250 | loss: 0.56108 - binary_acc: 0.7800 | val_loss: 0.55753 - val_acc: 0.7780 -- iter: 32561/32561
| deep_regression | epoch: 1250 | loss: 0.30490 - binary_acc: 0.8576 | val_loss: 0.30492 - val_acc: 0.8576 -- iter: 32561/32561
| central_bias_regression | epoch: 1250 | loss: 0.46839 - binary_acc: 0.8158 | val_loss: 0.46368 - val_acc: 0.8176 -- iter: 32561/32561
--
============================================================  Evaluation
  logits: (16281,), min=-14.6657066345, max=74.5122756958
Actual IDV
0    12435
1     3846

Predicted IDV
0    15127
1     1154

Confusion matrix:
actual           0     1
predictions
0            12296  2831
1              139  1015
============================================================
```

(Note how TFLearn shows losses and accuracy numbers for all three regressions).  The final accuracy for the combined wide+deep model is 81.76%

It is striking, though, that the deep model evidently gives 85.76%
accuracy, whereas the wide model gives 77.8% accuracy, at least for
the run recorded above.  The combined model has performance inbetween.

On more complicated datasets, perhaps the outcome would be different.

Testing
=======

Unit tests are provided, implemented using [pytest](http://doc.pytest.org/en/latest/).  Run these using:

    py.test tflearn_wide_and_deep.py

Installation
============

* Requires TF 0.10 or better
* Requires TFLearn installed from github (with [PR#308](https://github.com/tflearn/tflearn/pull/308))
