import argparse
import os
import sys
from moviepy.editor import VideoFileClip

# Models supported for Vehicle Detection -
# svm / yolo3 / ssd / mrcnn
def process_image(img):
    exec('from util_'+args.vdmodel+' import *')
    image = eval('car_'+args.vdmodel)(img)
    
    return image


"""
Converts a string to y/n boolean
"""
def s2b(s):
	s = s.lower()
	return s == 'true' or s == 'yes' or s == 'y' or s == '1'

"""
parameters and defaults setting
"""
def set_args():
    parser = argparse.ArgumentParser(description='Vehicle Detection')
    parser.add_argument('-m', help='model: svm/yolo3/ssd/mrcnn', dest='vdmodel', type=str, default='svm')
    parser.add_argument('-v', help='video file', dest='vfile', type=str, default='project_video.mp4')
    
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    return args

# process video or image file
def main():
    fname, fext = os.path.splitext(args.vfile)
    proj_output = fname + '_' + args.vdmodel + fext
    clip = VideoFileClip(args.vfile)
    proj_clip = clip.fl_image(process_image)
    proj_clip.write_videofile(proj_output, audio=False)
    

if __name__ == '__main__':
    heat_threshold = 1 # threshold of heatmap
    args = set_args()
    main()


    
    
       
