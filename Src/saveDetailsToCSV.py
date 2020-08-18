import os

def file_name_path(file_dir, dir=True, file=True):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return:
    """
    print(file_dir)

    for root, dirs, files in os.walk(file_dir):
        files.pop(0)
        print(root)
        print(dirs)
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", len(files))
            return files


def save_file2csv(file_dir, file_name):
    """
    save file path to csv
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    image = "image"
    mask = "mask"
    file_image_dir = file_dir + "/" + image
    file_mask_dir = file_dir + "/" + mask
    file_paths = file_name_path(file_image_dir, dir=False, file=True)
    out.writelines("Image,Mask" + "\n")
    print(file_paths)
    for index in range(len(file_paths)):
        out_file_image_path = file_image_dir + "/" + file_paths[index]
        out_file_mask_path = file_mask_dir + "/" + file_paths[index]
        out.writelines(out_file_image_path + "," + out_file_mask_path + "\n")



save_file2csv("/Volumes/Samsung_T5/gen_image3d", "vertebra3dSegmentation.csv")