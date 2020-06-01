import tensorflow as tf

use_NAS = False
if use_NAS:
  IMG_SIZE = 224 # 299 for Inception, 224 for NASNetMobile
  IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
else:
  IMG_SIZE = 299 # 299 for Inception, 224 for NASNetMobile
  IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
  
def load_image(image_path):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
  if use_NAS:
    img = tf.keras.applications.nasnet.preprocess_input(img)
  else:
    img = tf.keras.applications.inception_v3.preprocess_input(img)
  return img, image_path

def create_model(image_batch):
  tf.keras.backend.clear_session()

  if use_NAS:
    # Create the base model from the pre-trained model 
    base_model = tf.keras.applications.NASNetMobile(input_shape=IMG_SHAPE,
                                                  include_top=False,
                                                  weights='imagenet')
  else:
    # Create the base model from the pre-trained model 
    base_model = tf.keras.applications.InceptionResNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
  feature_batch = base_model(image_batch)
    
  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
  feature_batch_average = global_average_layer(feature_batch)
  prediction_layer = tf.keras.layers.Dense(3)
  prediction_batch = prediction_layer(feature_batch_average)

  model = tf.keras.Sequential([
                               base_model,
                               global_average_layer,
                               prediction_layer])

  base_learning_rate = 0.0001
  model.compile(optimizer=tf.keras.optimizers.Nadam(lr=base_learning_rate),
                loss=tf.keras.losses.MeanAbsoluteError(),
                metrics=['mae', 'mse', 'accuracy'])
  return model