import glob
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from image_classifier import ImageClassifiers
import numpy as np


def main():
    emb = np.load("sunflower/sunflower1.npy")
    mr_job = ImageClassifiers(emb, args=['-r', 'local', "images/file_names.txt"])

    mr_job.run_mapper(step_num=1)


    # with mr_job.make_runner() as runner:
    #     runner.run()


def save_names():
    path = "images"
    file_names = glob.glob(os.path.join(path, "*.hdf5"))
    # remove the path
    file_names = [os.path.basename(file_name).replace(".hdf5", "") for file_name in file_names]

    # save the file names to a text file
    with open(path+"/file_names.txt", "w") as f:
        for file_name in file_names:
            f.write(file_name + "\n")


if __name__ == '__main__':
    main()