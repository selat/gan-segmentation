# For running in display-less mode
import matplotlib as mpl
mpl.use('Agg')
import tensorflow as tf
from keras import backend as keras
import cv2
from keras.models import *
from keras.layers.merge import concatenate
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import skimage.io as skio
import argparse
import os
import matplotlib.pyplot as plt

with tf.device('/gpu:0'):
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    keras.set_session(tf.Session(config=config))
    
def load_data(rows, cols, crops_num=30, start_index=1):
    imgs_train, imgs_mask_train = [], []
    
    for i in range(start_index, 1000 + start_index, 10):
        img = skio.imread('data/images/{:05}.png'.format(i))
        mask = skio.imread('data/labels/{:05}.png'.format(i))
        for j in range(crops_num):
            startrow = np.random.randint(img.shape[0] - rows)
            startcol = np.random.randint(img.shape[1] - cols)
            img_crop = img[startrow:startrow + rows, startcol:startcol + cols]
            mask_crop = mask[startrow:startrow + rows, startcol:startcol + cols]
            bmask = cv2.inRange(mask_crop, np.array([128, 64, 128]), np.array([128, 64, 128]))
            bmask = np.expand_dims(bmask, axis=2)
            imgs_train.append(img_crop)
            imgs_mask_train.append(bmask)
    
    imgs_train, imgs_mask_train = np.array(imgs_train, dtype=np.float32), np.array(imgs_mask_train, dtype=np.float32)
    imgs_train /= 127.5
    imgs_train -= 1.0
    imgs_mask_train /= 255
    return imgs_train, imgs_mask_train

def get_unet_small(rows, cols):
    inputs = Input((rows, cols,3))

    conv1 = Conv2D(8, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    print "conv1 shape:",conv1.shape
    conv1 = Conv2D(8, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    print "conv1 shape:",conv1.shape
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print "pool1 shape:",pool1.shape

    conv2 = Conv2D(16, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    print "conv2 shape:",conv2.shape
    conv2 = Conv2D(16, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    print "conv2 shape:",conv2.shape
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print "pool2 shape:",pool2.shape

    conv3 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    print "conv3 shape:",conv3.shape
    conv3 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    print "conv3 shape:",conv3.shape
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print "pool3 shape:",pool3.shape

    conv4 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model

def generate_report(args, history, experiment_name):
    plt.plot(np.arange(1, len(history.history['acc']) + 1), history.history['acc'])
    plt.plot(np.arange(1, len(history.history['acc']) + 1), history.history['loss'])
    plt.plot(np.arange(1, len(history.history['acc']) + 1), history.history['val_loss'])
    plt.plot(np.arange(1, len(history.history['acc']) + 1), history.history['val_acc'])
    plt.legend(['acc', 'loss', 'val_locc', 'val_acc'])
    os.mkdir(os.path.join('experiments', experiment_name))
    plt.savefig(os.path.join('experiments', experiment_name, 'plot.png'), figsize=(20, 10))
    with open(os.path.join('experiments', experiment_name, 'report.md'), 'w') as f:
        f.write('# {}\n'.format(experiment_name))
        f.write('Arguments: {}\n'.format(args))
        f.write('\n![plot](plot.png)\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, dest='size', required=True, help='crop size')
    parser.add_argument('--model_name', type=str, dest='model_name', required=True, help='name of the model')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--crops_num', type=int, default=30, help='number of crops')
    parser.add_argument('--experiment_name', type=str, required=True, help='name of the experiment')
    parser.add_argument('--batch_size', type=int, default=100, help='size of the training batch')
    args = parser.parse_args()
    imgs_train, imgs_mask_train = load_data(args.size, args.size, crops_num=args.crops_num)
    imgs_val, imgs_mask_val = load_data(args.size, args.size, crops_num=args.crops_num, start_index=3001)
    model = get_unet_small(args.size, args.size)
    model_checkpoint = ModelCheckpoint(args.model_name + '.hdf5', monitor='loss',verbose=1, save_best_only=True)
    print('Fitting model...')
    history = model.fit(imgs_train, imgs_mask_train, batch_size=args.batch_size, epochs=args.epochs,
                        verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
    generate_report(args, history, args.experiment_name)
   
if __name__ == '__main__':
    main()