import glob
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from image_classifictio
import numpy as np

def main():
    path = "images"
    file_names = glob.glob(os.path.join(path, "*.hdf5"))
    emb = np.load("sunflower/sunflower1.npy")
    mr_job = ImageClassifiers(emb, args=['-r', 'local', file_names])
    with mr_job.make_runner() as runner:
        runner.run()

if __name__ == '__main__':
    main()