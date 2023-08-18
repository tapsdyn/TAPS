import common


def return_somethingv2():
    root_data = common.STHV2_FRAMES
    filename_categories ='data/somethingv2/classInd.txt'
    filename_imglist_train = "data/somethingv2/train_videofolder.txt"
    filename_imglist_val = "data/somethingv2/val_videofolder.txt"
    train_folder_suffix = ""
    val_folder_suffix = ""
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, train_folder_suffix, val_folder_suffix


def return_somethingv2_half1():
    root_data = common.STHV2_FRAMES
    filename_categories ='data/somethingv2_half1/classInd.txt'
    filename_imglist_train = "data/somethingv2_half1/train_videofolder.txt"
    filename_imglist_val = "data/somethingv2_half1/val_videofolder.txt"
    train_folder_suffix = ""
    val_folder_suffix = ""
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, train_folder_suffix, val_folder_suffix


def return_jester():
    root_data = common.JESTER_FRAMES
    filename_categories = 'data/jester/classInd.txt'
    filename_imglist_train = 'data/jester/train_split.txt'
    filename_imglist_val = 'data/jester/validation_split.txt'
    train_folder_suffix = ""
    val_folder_suffix = ""
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, train_folder_suffix, val_folder_suffix


def return_minik_aug23():
    root_data = common.MINIK_AUG23_FRAMES
    filename_categories = 'data/kinetics/minik_classInd.txt'
    filename_imglist_train = 'data/kinetics/mini_train_videofolder_aug23.txt'
    filename_imglist_val = 'data/kinetics/mini_val_videofolder_aug23.txt'
    train_folder_suffix = ""
    val_folder_suffix = ""
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, train_folder_suffix, val_folder_suffix


def return_activitynet_1_3_aug23():
    root_data = common.ACTIVITYNET_1_3_AUG23_FRAMES
    filename_categories = 'data/ActivityNet/classInd.txt'
    filename_imglist_train = 'data/ActivityNet/actnet_train_videofolder_aug23.txt'
    filename_imglist_val = 'data/ActivityNet/actnet_val_videofolder_aug23.txt'
    train_folder_suffix = ''
    val_folder_suffix = ''
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, train_folder_suffix, val_folder_suffix


def return_dataset(dataset, data_path):
    dict_single = {'somethingv2': return_somethingv2,
                   'somethingv2_half1': return_somethingv2_half1,
                   'jester': return_jester,
                   'minik_aug23': return_minik_aug23,
                   'activitynet_1_3_aug23': return_activitynet_1_3_aug23}

    common.set_manual_data_path(data_path, None)
    file_categories, file_imglist_train, file_imglist_val, root_data, train_folder_suffix, val_folder_suffix = \
        dict_single[dataset]()

    with open(file_categories) as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]
    n_class = len(categories)
    if dataset in ['minik_aug23', 'activitynet_1_3_aug23']:
        prefix = 'image_{:05d}.jpg'
    else:
        prefix = '{:05d}.jpg'
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix, train_folder_suffix, val_folder_suffix
