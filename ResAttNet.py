
import argparse
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config
from utils.model import ResidualAttentionNetwork

ap = argparse.ArgumentParser()
ap.add_argument("--model_save", required=False, type=bool, help="Whether to save the generated model weight")
args = ap.parse_args()

model_save = True
if args.model_save:
    model_save = args.model_save

def load_cifar10():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, 
                                                                          train_labels, 
                                                                          test_size=0.1, 
                                                                          shuffle=True)
    train_images = train_images/255
    test_images = test_images/255
    val_images = val_images/255
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


def main(model_save=False):
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_cifar10()
    tf.keras.backend.clear_session()
    MODEL_NAME = config.MODEL_NAME
    p = config.p 
    t = config.t 
    r = config.r 

    resattnet = ResidualAttentionNetwork(p=p, t=t, r=r, name=MODEL_NAME)
    
    lr = config.LEARNING_RATE
    decay = config.DECAY
    momentum = config.MOMENTUM
    optimizer = tf.keras.optimizers.SGD(lr=lr, decay=decay, momentum=momentum)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = ['accuracy']
    resattnet.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    EPOCHS = config.EPOCHS
    BATCH_SIZE = config.BATCH_SIZE
    hist = resattnet.fit(
        x=train_images,
        y=train_labels,
        validation_data=(test_images, test_labels),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True
    )

    print('[*] train finish')
    result = resattnet.evaluate(test_images, test_labels)
    print(f'test loss : {result[0]}, test accuracy : {result[1]}')

    if model_save:
        resattnet.save_weights('./gan_save/ckpt')

if __name__ == '__main__':
    main(model_save)