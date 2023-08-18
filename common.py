from os.path import join as ospj

from os.path import expanduser
ROOT_DIR = ospj(expanduser("~"), "scratch")  # TODO change this to your root path // root dir
# DATA_PATH = ospj(ROOT_DIR, "datasets")       # TODO change this to your data path // dataset path
# EXPS_PATH = ospj(ROOT_DIR, "logs_tsm")       # TODO change this to your logs path // saving logs
DATA_PATH = '/stage/algo-datasets/DB/CBI'
EXPS_PATH = '/algo/CBI_artifacts/logs_tsm'

def inner_set_manual_data_path(data_path, exps_path):
    if data_path is not None:
        global DATA_PATH
        DATA_PATH = data_path

    if exps_path is not None:
        global EXPS_PATH
        EXPS_PATH = exps_path


def set_manual_data_path(data_path, exps_path):
    inner_set_manual_data_path(data_path, exps_path)

    global STHV2_FRAMES
    STHV2_FRAMES = ospj(DATA_PATH, "something2something-v2", "frames")

    global JESTER_FRAMES
    JESTER_FRAMES = ospj(DATA_PATH, "jester", "20bn-jester-v1")

    global ACTIVITYNET_1_3_AUG23_FRAMES
    ACTIVITYNET_1_3_AUG23_FRAMES = ospj(DATA_PATH, "ActivityNet1_3", "Frames_aug23", "train_val")

    global MINIK_AUG23_FRAMES
    MINIK_AUG23_FRAMES = ospj(DATA_PATH, "Kinetics_400", "mini_kinetics_frames_aug23")
