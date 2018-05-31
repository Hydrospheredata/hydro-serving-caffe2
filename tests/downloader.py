import shutil
from caffe2.python.models.download import downloadFromURLToFile, getURLFromName
import os


def download_modelzoo_model(model, model_folder, force=False):
    # Check if that folder is already there
    if os.path.exists(model_folder) and not os.path.isdir(model_folder):
        if not force:
            raise Exception("Cannot create folder for storing the model,\
                            there exists a file of the same name.")
        else:
            print("Overwriting existing file! ({filename})"
                  .format(filename=model_folder))
            os.remove(model_folder)
    if os.path.isdir(model_folder):
        if not force:
            return
        print("Overwriting existing folder! ({filename})".format(filename=model_folder))
        shutil.rmtree(model_folder)

    # Now we can safely create the folder and download the model
    os.makedirs(model_folder)
    for f in ['predict_net.pb', 'init_net.pb']:
        try:
            downloadFromURLToFile(getURLFromName(model, f),
                                  '{folder}/{f}'.format(folder=model_folder,
                                                        f=f))
        except Exception as e:
            print("Abort: {reason}".format(reason=str(e)))
            print("Cleaning up...")
            shutil.rmtree(model_folder)
            exit(0)


def download_googlenet(model_path, force=False):
    download_modelzoo_model("bvlc_googlenet", model_path)
