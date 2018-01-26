import os, shutil
import channels

def clear_dir(folder):
    if os.path.exists(folder):
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def args_setup(args):
    # setup parametrization
    if args.chan == "uv":
        get_channels = channels.get_uv
    elif args.chan == "yuv":
        get_channels = channels.get_yuv 
    elif args.chan == "no_luma":
        get_channels = channels.center_luma
    else:
        get_channels = channels.get_bgr

    print("Training Movie:", args.training_mov)
    print("Input Movie:", args.movie_file)
    print("Segmentations:", args.segmentation_dir) 

    # reset segmentation directory
    clear_dir(args.segmentation_dir)
    create_dir(args.segmentation_dir)

    return get_channels
    