# For running in display-less mode
import matplotlib as mpl
mpl.use('Agg')
import tensorflow as tf
from keras import backend as keras
from keras.models import *
from keras.layers.merge import concatenate
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import skimage.io as skio
import argparse
import os
import matplotlib.pyplot as plt
import skimage.transform
import random

import imgaug as ia
from imgaug import augmenters as iaa

with tf.device('/gpu:0'):
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    keras.set_session(tf.Session(config=config))

def simple_generator(path_images, path_masks, crops_num=1, batch_size=5, crop_size=128):
    files_images = sorted(os.listdir(path_images))[:20000]
    files_masks = sorted(os.listdir(path_masks))[:20000]
    while True:
        images = np.zeros((batch_size * crops_num, crop_size, crop_size, 3), dtype=np.float32)
        masks = np.zeros((batch_size * crops_num, crop_size, crop_size, 1), dtype=np.float32)
        for i in range(batch_size):
            cur_id = np.random.randint(len(files_images))
            img = skio.imread(os.path.join(path_images, files_images[cur_id]))
            mask = skio.imread(os.path.join(path_masks, files_masks[cur_id]))
            for j in range(crops_num):
                startrow = np.random.randint(img.shape[0] - crop_size)
                startcol = np.random.randint(img.shape[1] - crop_size)
                img_crop = img[startrow:startrow + crop_size, startcol:startcol + crop_size]
                mask_crop = mask[startrow:startrow + crop_size, startcol:startcol + crop_size]
                mask_crop = np.all(mask_crop == (128, 64, 128), axis=-1).astype(np.float32)
                mask_crop = np.expand_dims(mask_crop, axis=2)
                images[i * crops_num + j] = img_crop
                masks[i * crops_num + j] = mask_crop
#         seq = iaa.Sequential([
#             iaa.Fliplr(0.5),
#             iaa.Affine(scale=(0.5, 1.5), shear=(-32, 32))], random_order=True)
#         ia.seed(1)
#         images = seq.augment_images(images)
#         ia.seed(1)
#         masks = seq.augment_images(masks)

#         seq_img = iaa.Sequential([
#             iaa.ContrastNormalization((0.5, 1.5))],
#             random_order=True)
#         images = seq_img.augment_images(images)
        images /= 127.5
        images -= 1.0
        yield images, masks

def get_unet(rows, cols, layers_num):
    prev_layer = inputs = Input((rows, cols, 3))
    cur_filters_num = 16
    convs = []
    for l in range(layers_num):
        conv = Conv2D(cur_filters_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(prev_layer)
        conv = Conv2D(cur_filters_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
        convs.append(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        prev_layer = pool
        cur_filters_num *= 2

    conv = Conv2D(cur_filters_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(prev_layer)
    conv = Conv2D(cur_filters_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    drop = Dropout(0.5)(conv)

    prev_layer = drop
    for l in range(layers_num):
        up = concatenate([UpSampling2D(size=(2, 2))(prev_layer), convs[-l - 1]], axis=3)
        cur_filters_num //= 2
        conv = Conv2D(cur_filters_num, 3, activation='relu', padding='same')(up)
        conv = Conv2D(cur_filters_num, 3, activation='relu', padding='same')(conv)
        prev_layer = conv

    conv = Conv2D(1, 1, activation='sigmoid')(prev_layer)
    model = Model(inputs=inputs, outputs=conv)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model

def show_sample_segmentation(rows, model, path):
    tmp_model = get_unet_small(1024, 2048)
    for i in range(len(model.layers)):
        tmp_model.layers[i].set_weights(model.layers[i].get_weights())
    state = random.getstate()
    random.seed(13)
    figure, plots = plt.subplots(rows, 2, gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    figure.set_size_inches(20, 27)
    for i in range(rows):
        img_id = random.randint(1, 24967)
        super_input = skio.imread('data/images/{:05}.png'.format(img_id))
        super_input = skimage.transform.resize(super_input, (1024, 2048), preserve_range=True)
        tmp = super_input.astype(np.float32)
        tmp /= 127.5
        tmp -= 1.0
        res = tmp_model.predict(np.array([tmp]))
        plots[i][0].axis('off')
        plots[i][0].imshow(super_input / 255)
        res = np.squeeze(res[0], axis=2)
        plots[i][1].axis('off')
        plots[i][1].imshow(res)
    plt.savefig(os.path.join(path, 'examples.png'))
    random.setstate(state)

def generate_report(model, args, history, experiment_name):
    plt.plot(np.arange(1, len(history.history['acc']) + 1), history.history['acc'])
    plt.plot(np.arange(1, len(history.history['acc']) + 1), history.history['loss'])
    plt.plot(np.arange(1, len(history.history['acc']) + 1), history.history['val_loss'])
    plt.plot(np.arange(1, len(history.history['acc']) + 1), history.history['val_acc'])
    plt.legend(['acc', 'loss', 'val_locc', 'val_acc'])
    path = os.path.join('experiments', experiment_name)
    try:
        os.mkdir(path)
    except OSError:
        pass
    plt.savefig(os.path.join(path, 'plot.png'), figsize=(20, 10))
    show_sample_segmentation(5, model, path)
    with open(os.path.join(path, 'report.md'), 'w') as f:
        f.write('# {}\n'.format(experiment_name))
        f.write('Arguments: {}\n'.format(args))
        f.write('\n![plot](plot.png)\n')
        f.write('# Segmentation examples\n')
        f.write('![examples](examples.png)\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, dest='size', required=True, help='crop size')
    parser.add_argument('--model_name', type=str, dest='model_name', required=True, help='name of the model')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--crops_num', type=int, default=30, help='number of crops')
    parser.add_argument('--experiment_name', type=str, required=True, help='name of the experiment')
    parser.add_argument('--batch_size', type=int, default=100, help='size of the training batch')
    parser.add_argument('--steps_per_epoch', type=int, default=50, help='steps per epoch')
    args = parser.parse_args()
    model = get_unet(args.size, args.size, 4)
    model_checkpoint = ModelCheckpoint(args.model_name + '.hdf5', monitor='loss',verbose=0, save_best_only=True)
    print('Fitting model...')
    gen = simple_generator('data/images', 'data/labels', crops_num=args.crops_num,
                           batch_size=args.batch_size, crop_size=args.size)
    val_gen = simple_generator('data/images', 'data/labels', crops_num=args.crops_num,
                               batch_size=args.batch_size, crop_size=args.size)
    history = model.fit_generator(
        generator=gen,
        validation_data=val_gen.next(),
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        callbacks=[model_checkpoint])
    generate_report(model, args, history, args.experiment_name)
   
if __name__ == '__main__':
    main()