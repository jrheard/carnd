import csv
import numpy as np
import glob
import random

from keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Lambda

import PIL
from PIL import Image
from PIL import ImageEnhance

with open('data/driving_log.csv') as f:
    driving_log = csv.DictReader(f)
    log_lines = [line for line in driving_log]

# Load all the image files into memory - they're just like ~330Mb, it's fine.
# This makes each epoch much faster.
# I used cprofile to see where the time was going in my epochs, and found that
# a lot of it was being spent in this load_img() call, which was a lot of duplicated
# work each epoch. Removing it sped up my epochs by ~40%.
image_files = {
    path: load_img('data/' + path.strip())
    for line in log_lines
    for path in (line['center'], line['left'], line['right'])
}


# This is the comma.ai model from https://github.com/commaai/research/blob/master/train_steering_model.py .
def get_model():
    model = Sequential([

        # Normalize all pixel values to the range from -1 to 1.
        # Dan in Slack points out that having "0 mean" is helpful in neural nets,
        # and also that they deal best with smaller numbers.
        Lambda(lambda x: x/127.5 - 1., input_shape=(90, 300, 3), output_shape=(90, 300, 3)),

        Conv2D(16, 8, 8, subsample=(4, 4), border_mode='same', activation='relu', name='conv_1'),
        Conv2D(16, 5, 5, subsample=(2, 2), border_mode='same', activation='relu', name='conv_2'),
        Conv2D(16, 5, 5, subsample=(2, 2), border_mode='same', activation='relu', name='conv_3'),

        Flatten(name='flatten_1'),
        # Comma uses a 0.2 dropout here, but 0.5 seemed like a good idea to me. No particular reason.
        Dropout(0.5),

        # I originally went with Comma's 512, but @xslittlegrass in slack was able to make their model
        # work using only 63 parameters, so I figured it'd be a good idea to see if shrinking this
        # layer down hurt anything. It didn't!
        Dense(24, activation='relu', name='fc_1'),
        Dropout(0.5),

        # Output layer, no activation function.
        Dense(1, name='output_1'),
    ])

    model.compile(optimizer="adam", loss="mse")
    model.summary()
    return model

def flip_image(image, angle):
    return image.transpose(PIL.Image.FLIP_LEFT_RIGHT), -angle

def tweak_image_brightness(image):
    # Tweak the image's brightness - got the idea from https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.ndechdida
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(random.uniform(0.2, 1.8))

def pick_randomly_positioned_crop(image, angle):
    # Crop out the sky and the hood of the car - and include a little random variance on the actual
    # specific location of the crop window in the image, in order to make the model more robust to
    # being in different positions in the lane.
    # got the idea from https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.mcibyrg1q
    default_top = 50
    default_left = 10
    crop_width = 300
    crop_height = 90

    x_delta = random.uniform(-10, 10)
    y_delta = random.uniform(-10, 10)

    cropped_image = image.crop((
        default_left + x_delta,
        default_top + y_delta,
        default_left + crop_width + x_delta,
        default_top + crop_height + y_delta,
    ))

    # Per @ksakmann's article, adjust the projected steering angle
    # based on our horizontal displacement from center.
    # I played around with this value a lot - 0.002 was the only value
    # that worked reliably well for me.
    angle_shift_per_horizontal_pixel = 0.002
    adjusted_angle = angle + (x_delta * angle_shift_per_horizontal_pixel)

    return cropped_image, adjusted_angle

def randomly_tweak_log_line(log_line):
    angle = float(log_line['steering'])

    # use the side cameras' images to train the model on how to recover from being offcenter
    # tried a bunch of different values here, but 0.06 was the one that worked best for me
    # got the idea from https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.mcibyrg1q
    # his value is ~0.06 as well
    side_camera_offset = 0.06

    paths_and_angles = [
        (log_line['center'], angle),
        (log_line['left'], angle + side_camera_offset),
        (log_line['right'], angle - side_camera_offset),
    ]

    path, angle = random.choice(paths_and_angles)

    image = image_files[path]

    image = tweak_image_brightness(image)

    return pick_randomly_positioned_crop(image, angle)

PERCENTAGE_OF_ZERO_STEERING_FRAMES_TO_DROP = 0.5

def training_data_generator():
    batch_size = 128

    while True:
        # The input dataset includes a ton of frames where the steering angle is zero, so in order to avoid
        # training a model with a bias against turning when it's necessary, we drop 50% of our zero-angle
        # inputs on the ground. I experimented with the percentage of zero-angle inputs to drop, and found
        # that 50% worked the best for me.
        # Got the idea from https://medium.com/@ValipourMojtaba/my-approach-for-project-3-2545578a9319#.h8b2fsdi1
        lines_this_batch = [
                line for line in log_lines
                if float(line['steering']) != 0
                or random.random() > PERCENTAGE_OF_ZERO_STEERING_FRAMES_TO_DROP
            ]

        for i in range(0, len(lines_this_batch), batch_size):
            images_and_angles = [
                randomly_tweak_log_line(line)
                for line in lines_this_batch[i:i+batch_size]
            ]

            images, angles = zip(*images_and_angles)

            # Flip the image horizontally, to avoid the training dataset's bias for turning left.
            # Lots of other people seem to prefer to do this with a 50% chance, but I prefer
            # training the system on a dataset that includes all of the unflipped images _and_
            # a corresponding flipped image for each of the original images.
            flipped_images, flipped_angles = zip(*[flip_image(image, angle) for (image, angle) in images_and_angles])

            yield (
                np.asarray([
                    np.asarray(image) for image in
                    list(images) + list(flipped_images)
                ]),
                np.asarray(list(angles) + list(flipped_angles))
            )

if __name__ == '__main__':
    model = get_model()
    model.fit_generator(
        training_data_generator(),
        2 * len(log_lines), # generator will produce fewer samples per batch than this because of PERCENTAGE_OF_ZERO_STEERING_FRAMES_TO_DROP
        # Some people found that having a small number of epochs worked for them, but
        # I was able to get the most reliable+robust models by having a large number of epochs.
        # When I only used 10 or 15 epochs and left the other hyperparameters constant,
        # I would often get models that drove into the lake or couldn't make the left turn after the bridge.
        60,
        verbose=2,
    )

    with open('model.json', 'w') as f:
        f.write(model.to_json())

    model.save_weights('model.h5'.format(i))

    import gc; gc.collect()
