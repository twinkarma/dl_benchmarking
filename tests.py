import shutil
from unittest import TestCase
from dlbenchmark.image import ImageDatasetGenerator


class TestImageFileTransfer(TestCase):

    def setUp(self):
        pass


    def test_image_file_transfer(self):
        # Generate image file
        img_dir = "testdir"

        num_images = 30
        image_size = 30
        image_format = "jpg"
        shutil.rmtree(img_dir)
        img_paths = ImageDatasetGenerator.generate_image(img_dir, num_images, image_size, image_size, image_format)

class TestImageDataset(TestCase):

    def test_image_dataset(self):
        # Generate image file
        img_dir = "testdir"

        num_clases = 10
        num_images = 30
        image_size = 30
        batch_size = 4
        buffer_size = 10
        image_format = "jpg"
        shutil.rmtree(img_dir)
        img_paths, img_labels = ImageDatasetGenerator.generate_image_classes(img_dir, num_clases, num_images, image_size, image_size, image_format)

        img_dataset = ImageDatasetGenerator.make_dataset(img_paths,
                                                         img_labels,
                                                         image_size,
                                                         image_size,
                                                         batch_size,
                                                         buffer_size)

        for batch in img_dataset:
            img, label = batch
            print(img.shape)
            print(label.shape)
            myimg_array = img.numpy()


