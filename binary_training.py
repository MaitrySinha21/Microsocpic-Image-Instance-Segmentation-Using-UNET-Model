"""
code : swati sinha & maitry sinha
"""
from binary_generator import *
from data import data_size, batch
import tensorflow as tf
from unet_model import binary_unet_small
# from unet_model import jacard_coef, jacard_coef_loss, dice_coef, dice_coef_loss
import os

size = data_size()
seed = 2020
batch_size = batch()


root_dir = 'nucleus_data'
model_path = 'Flask_nucleus/model/nucleus_small_256_unet_model.hdf5'


class Training:
    def __init__(self, model_path, root_dir, batch_size, size):
        self.batch_size = batch_size
        self.model_path = model_path
        self.directory = root_dir
        self.size = size
        self.train_img_path = "{}/data/training_data/train_images/".format(self.directory)
        self.train_mask_path = "{}/data/training_data/train_masks/".format(self.directory)
        self.val_img_path = "{}/data/training_data/val_images/".format(self.directory)
        self.val_mask_path = "{}/data/training_data/val_masks/".format(self.directory)
        self.train_img_gen = trainGenerator(self.train_img_path, self.train_mask_path,
                                            batch_size=self.batch_size, img_size=self.size)
        self.val_img_gen = trainGenerator(self.val_img_path, self.val_mask_path,
                                          batch_size=self.batch_size, img_size=self.size)
        self.num_train_imgs = len(os.listdir(os.path.join(self.train_img_path, 'train')))
        self.num_val_imgs = len(os.listdir('{}/data/training_data/val_images/val/'.format(self.directory)))
        self.steps_per_epoch = self.num_train_imgs // self.batch_size
        self.val_steps_per_epoch = self.num_val_imgs // self.batch_size

    def print_gen(self):
        print('No. of train images     :', self.num_train_imgs)
        print('No. of validation images:', self.num_val_imgs)
        print('training steps per epoch:', self.steps_per_epoch)
        print('validation steps per epoch:', self.val_steps_per_epoch)

    def model_save(self):
        mod = binary_unet_small(size=(self.size, self.size, 1))
        os.makedirs('unet_models', exist_ok=True)
        mod.save(self.model_path)
        print('model saved at....:{}'.format(self.model_path))

    def train(self, epochs=10, lr=2e-4, training_type='saved_model'):
        if training_type == 'saved_model':
            model = tf.keras.models.load_model(self.model_path, compile=True)
            print('saved model loaded..')
        elif training_type == 'new_model':
            self.model_save()
            model = tf.keras.models.load_model(self.model_path, compile=True)
            print('new model loaded...')

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss='binary_crossentropy', metrics='accuracy')
        hist = model.fit(self.train_img_gen, steps_per_epoch=self.steps_per_epoch,
                         epochs=epochs,
                         verbose=1, validation_data=self.val_img_gen,
                         validation_steps=self.val_steps_per_epoch)
        model.save(self.model_path)
        print('model trained and saved..')
        return hist


# train = Training(model_path=model_path, root_dir=root_dir, batch_size=batch_size, size=size)
# train.print_gen()
# hist = train.train(epochs=30, lr=2e-4, training_type='saved_model')
# plot_history(hist)
