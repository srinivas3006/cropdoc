import os, shutil, math

original_dataset_dir = './dataset_original/'
classes_list = os.listdir(original_dataset_dir)

base_dir = './dataset_splits'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir): os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir): os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir): os.mkdir(test_dir)

for cls in classes_list:
    if not os.path.exists(os.path.join(train_dir, cls)): os.mkdir(os.path.join(train_dir, cls))
    if not os.path.exists(os.path.join(validation_dir, cls)): os.mkdir(os.path.join(validation_dir, cls))
    if not os.path.exists(os.path.join(test_dir, cls)): os.mkdir(os.path.join(test_dir, cls))

for cls in classes_list:
    path = os.path.join(original_dataset_dir, cls)
    fnames = os.listdir(path)
    train_size = math.floor(len(fnames) * 0.8)
    validation_size = math.floor(len(fnames) * 0.15)
    test_size = math.floor(len(fnames) * 0.05)
    
    train_fnames = fnames[:train_size]
    print(f"Train size({cls}): {len(train_fnames)}")
    for fname in train_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(train_dir, cls, fname)
        shutil.copyfile(src, dst)
        
    validation_fnames = fnames[train_size:(validation_size + train_size)]
    print(f"Validation size({cls}): {len(validation_fnames)}")
    for fname in validation_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(validation_dir, cls, fname)
        shutil.copyfile(src, dst)
        
    test_fnames = fnames[(train_size+validation_size):(validation_size + train_size + test_size)]
    print(f"Test size({cls}): {len(test_fnames)}")
    for fname in test_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(test_dir, cls, fname)
        shutil.copyfile(src, dst)
