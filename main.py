import os
import shutil
import sys
import time
from dlbenchmark.timer import Timer
from dlbenchmark.image import ImageDatasetGenerator

def run_benchmark_on_dir(img_dir, output_file_path=None, test_id=None):
    out_status_str = "" if output_file_path is None else f" with output to {output_file_path}"
    print(f"Running dataset read benchmark on {img_dir} {out_status_str}")

    if not test_id:
        test_id = ""

    num_images_per_class = [100, 250, 500, 1000, 5000, 7500]
    image_sizes = [50, 100, 250, 500, 750]
    num_images_per_class = [10, 20]
    image_sizes = [15, 30]
    batch_sizes = [32, 64, 128, 256]
    buffer_sizes = [10, 100, 500, 1000]
    num_epoch_repeats = 5
    num_classes = 10
    image_format = "jpg"


    for num_image_per_class in num_images_per_class:
        for image_size in image_sizes:
            # Generate image files

            # Delete dir
            if os.path.exists(img_dir):
                print(f"Directory {img_dir} already exists, deleting")
                shutil.rmtree(img_dir)

            # Creates image and labels at location
            print("Generating images")
            img_paths, img_labels = ImageDatasetGenerator.generate_image_classes(img_dir, num_classes, num_image_per_class, image_size,
                                                                                 image_size, image_format)

            total_images = len(img_paths)

            # Pre-loading tf.dataset code so it doesn't affect timing of the first entry
            img_dataset = ImageDatasetGenerator.make_dataset(img_paths,
                                                             img_labels,
                                                             image_sizes[0],
                                                             image_sizes[0],
                                                             batch_sizes[0],
                                                             buffer_sizes[0])
            batch = img_dataset.take(1)

            print("Running dataset test")
            for batch_size in batch_sizes:
                for buffer_size in buffer_sizes:
                    for epoch in range(num_epoch_repeats):
                        # Start timing epoch
                        timer = Timer()
                        timer.start()
                        img_dataset = ImageDatasetGenerator.make_dataset(img_paths,
                                                                         img_labels,
                                                                         image_size,
                                                                         image_size,
                                                                         batch_size,
                                                                         buffer_size)

                        for batch in img_dataset:
                            # Iterate through the dataset for one epoch
                            img, label = batch

                        epoch_read_time = timer.stop()
                        prefix = f"Ran test:{test_id} {total_images} images of size {image_size}x{image_size} with batch:{batch_size} buffer:{buffer_size} epoch no:{epoch} at {epoch_read_time}s"
                        print(prefix)
                        if output_file_path:
                            write_header = not os.path.exists(output_file_path)
                            with open(output_file_path, "a") as file:
                                if write_header:
                                    file.write(f"id,num_clases,num_images,image_size,batch_size,buffer_size,epoch_no,epoch_read_time\n")
                                file.write(f"\"{test_id}\",{num_classes},{total_images},{image_size},{batch_size},{buffer_size},{epoch},{epoch_read_time}\n")


            # Delete dir afterwards
            if os.path.exists(img_dir):
                print(f"Deleting test image directory {img_dir}")
                shutil.rmtree(img_dir)

def main():

    if len(sys.argv) < 2:
        print("Must define a path to put images")
        return 1

    image_dir = sys.argv[1]
    out_path = None
    id = None

    if len(sys.argv) > 2:
        out_path = sys.argv[2]

    if len(sys.argv) > 3:
        id = sys.argv[3]

    run_benchmark_on_dir(image_dir, out_path, id)


if __name__ == "__main__":
    main()
