import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import tensorflow as tf

class ImageDatasetGenerator:

    @staticmethod
    def generate_image(path, num_images, width, height, img_format: str):

        img_paths = []

        if not os.path.isdir(path):
            if os.path.exists(path):
                raise IOError(f"Provided path {path} is not a directory")
            else:
                pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        for i in range(num_images):
            r_img = np.random.randint(0, 255, size=(width, height, 3), dtype=np.uint8)
            img_path = os.path.join(path, f"image_{i}.{img_format}")
            plt.imsave(img_path, r_img)
            img_paths.append(img_path)

        return img_paths




    @staticmethod
    def generate_image_classes(path, num_classes, num_images, width, height, image_format):
        img_paths = []
        img_labels = []

        for i in range(num_classes):
            class_img_paths = ImageDatasetGenerator.generate_image(os.path.join(path, f"class_{i}"),
                                                 num_images,
                                                 width,
                                                 height,
                                                 image_format)
            img_paths.extend(class_img_paths)
            img_labels.extend([i for l in range(num_images)])

        return img_paths, img_labels

    @staticmethod
    def make_dataset(img_paths, img_labels, img_width, img_height, batch_size, buffer_size):

        def parse_image(filename):
            image = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [img_width, img_height])
            return image

        def configure_for_performance(ds):
            ds = ds.shuffle(buffer_size=buffer_size)
            ds = ds.batch(batch_size)
            # ds = ds.repeat()
            ds = ds.prefetch(buffer_size=buffer_size)
            return ds

        # classes = os.listdir(path)
        # filenames = glob(path + '/*/*')
        # random.shuffle(filenames)
        # labels = [classes.index(name.split('/')[-2]) for name in filenames]

        filenames_ds = tf.data.Dataset.from_tensor_slices(img_paths)
        images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        labels_ds = tf.data.Dataset.from_tensor_slices(img_labels)
        ds = tf.data.Dataset.zip((images_ds, labels_ds))
        ds = configure_for_performance(ds)

        return ds
