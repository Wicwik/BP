import os
path = './to_label2'
for filename in os.listdir(path):
    prefix, num = filename[:-4].split('-')
    num = num.zfill(6)
    new_filename = prefix + "-" + num + ".png"
    os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
    #print(new_filename)
