
import os
import os.path
print("Current env:", os.getenv('CONDA_DEFAULT_ENV'))
## create an experiment in the workspace


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from config import new_size
from plotdata import plot_training
from config import Config_classification

from azureml.core import Workspace, Experiment
ws = Workspace.from_config()
experiment = Experiment(workspace=ws,
                        name = 'dl-repro')


run = experiment.start_logging()



data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.1)
])


image_size = (new_size.get('width'), new_size.get('height'))
run.log('image_size',image_size)
batch_size = Config_classification.get('batch_size')
run.log('batch_size' , batch_size)
save_model_flag = Config_classification.get('Save_Model')
epochs = Config_classification.get('Epochs')
run.log('Epochs' , epochs)

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='bin_accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc')
]



dir_fire = 'frames/Training/Fire/'
dir_no_fire = 'frames/Training/No_Fire/'

# Count images in each directory
fire = len([name for name in os.listdir(dir_fire) if os.path.isfile(os.path.join(dir_fire, name))])
no_fire = len([name for name in os.listdir(dir_no_fire) if os.path.isfile(os.path.join(dir_no_fire, name))])
total = fire + no_fire
weight_for_fire = (1 / fire) * total / 2.0
weight_for_no_fire = (1 / no_fire) * total / 2.0

print("Weight for class fire : {:.2f}".format(weight_for_fire))
print("Weight for class No_fire : {:.2f}".format(weight_for_no_fire))

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "frames/Training", validation_split=0.2, subset="training", seed=1337, image_size=image_size,
    batch_size=batch_size, shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "frames/Training", validation_split=0.2, subset="validation", seed=1337, image_size=image_size,
    batch_size=batch_size, shuffle=True
)
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

import matplotlib.pyplot as plt

for images, labels in train_ds.take(1):  # Assuming `train_ds` is the dataset
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"Label: {int(labels[i])}")
        plt.axis('off')
plt.savefig("train_ds_images_glimpse.png")
run.log_image('train_ds images glimpse',
              path = 'train_ds_images_glimpse.png')



input_shape = image_size + (3,)
num_classes = 2

inputs = keras.Input(shape=input_shape)
x = inputs
#x = data_augmentation(inputs)
x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)

#x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
x = layers.Conv2D(8, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

previous_block_activation = x

for size in [8]:
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    
    residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
    x = layers.add([x, residual])
    previous_block_activation = x

x = layers.SeparableConv2D(8, 3, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)

activation = "sigmoid" if num_classes == 2 else "softmax"
units = 1 if num_classes == 2 else num_classes
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs, name="model_fire")

# Visualize the model architecture
keras.utils.plot_model(model, show_shapes=False,to_file="model_architecture.png")
run.log_image("Model Architecture",path = "model_architecture.png")

callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5")]

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, 
              loss="binary_crossentropy", 
              metrics=METRICS)
run.log('Learning Rate',0.001)

history = model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds, verbose=1)

# Log Loss and Accuracy for each epoch (training and validation)
for epoch in range(epochs):
    run.log('Training Loss', history.history['loss'][epoch])
    run.log('Training Accuracy', history.history['bin_accuracy'][epoch])
    
    run.log('Validation Loss', history.history['val_loss'][epoch])
    run.log('Validation Accuracy', history.history['val_bin_accuracy'][epoch])
    run.log('recall',history.history['recall'][epoch])
    run.log('precision',history.history['precision'][epoch])

file_model_fire = 'Output/Models/model_fire_resnet_weighted_40_no_metric_simple'
model.save(file_model_fire)

file_model_fire_azure = 'outputs/'
model.save(file_model_fire_azure)
run.upload_file(name = 'model_repro.pb',path_or_stream='outputs/saved_model.pb')


"""
#################################
 Classification after training the Model, modules and methods in this file evaluate the performance of the trained
 model over the test dataset
 Test Data: Item (8) on https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs 
 Tensorflow Version: 2.3.0
 GPU: Nvidia RTX 2080 Ti
 OS: Ubuntu 18.04
################################
"""
#########################################################
# import libraries
#most important metric for me is the FN
#my positive class is fire
#my negative class is no fire
#my Main metric is the recall

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from plotdata import plot_confusion_matrix
from config import Config_classification
from config import new_size

batch_size = Config_classification.get('batch_size')
image_size = (new_size.get('width'), new_size.get('height'))
epochs = Config_classification.get('Epochs')


test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "frames/Test", seed=1337, image_size=image_size, batch_size=batch_size, shuffle=True
    )

model_fire = load_model('Output/Models/model_fire_resnet_weighted_40_no_metric_simple')

_ = model_fire.evaluate(test_ds, batch_size=batch_size)

best_model_fire = load_model('Output/Models/h5model/keras/save_at_36.h5')
results_eval = best_model_fire.evaluate(test_ds, batch_size=batch_size)

# Evaluate the model and extract confusion matrix elements
results_eval = best_model_fire.evaluate(test_ds, batch_size=batch_size)
run.log('loss_best_model',results_eval[0])
run.log('accuracy_best_model',results_eval[1])

# Extract the individual confusion matrix elements (True Positives, False Positives, True Negatives, False Negatives)
tp = _[1]
fp = _[2]
tn = _[3]
fn = _[4]

# Log the confusion matrix as a table
run.log_table('Confusion Matrix', {
    ' ': ['Actual No Fire', 'Actual Fire'],  # Row headers
    'Predicted No Fire': [tn, fn],           # True Negatives, False Negatives
    'Predicted Fire': [fp, tp]               # False Positives, True Positives
})



# run.log('loss_best_model',results_eval[0])
# run.log('accuracy_best_model',results_eval[1])
# run.log('loss_base_model',_[0])
# run.log('accuracy_base_model',_[1])

# Assuming you already have these metrics
training_loss = history.history['loss'][-1]
validation_loss = history.history['val_loss'][-1]
training_accuracy = history.history['bin_accuracy'][-1]
validation_accuracy = history.history['val_bin_accuracy'][-1]


# Log the table with Loss and Accuracy for Training, Validation, and Test sets
run.log_table("Performance", {
    "Dataset": ["Training", "Validation", "Test"],
    "Loss": [training_loss, validation_loss, results_eval[0]],
    "Accuracy": [training_accuracy, validation_accuracy, results_eval[1]]
})



run.complete()


print("Experiment finished and logged successfully")
