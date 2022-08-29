import os

if __name__ == '__main__':
    datasets_dir = "datasets"

    types_name = os.listdir(datasets_dir)
    types_name = sorted(types_name)

    list_files = open('cls_train.txt', 'w')

    for cls_id, type_name in enumerate(types_name):
        photos_path = os.path.join(datasets_dir, type_name)
        if not os.path.isdir(photos_path):
            continue
        photos_name = os.listdir(photos_path)

        for photo_name in photos_name:
            list_files.write(
                str(cls_id) +
                ';' +
                '%s' %
                (os.path.join(
                    os.path.abspath(datasets_dir),
                    type_name,
                    photo_name)))
            list_files.write('\n')
list_files.close()
