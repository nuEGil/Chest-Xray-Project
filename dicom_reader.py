'''Goal of this file is to get some experience reading the dicom file byte stream'''
import argparse
import glob 
import os


def sum_bytes(x):
    sum_ = 0
    for i, x_ in enumerate(x):
        sum_ += (x_*(256**i)) 
    return sum_

def set_parser():
    # Read args
    parser = argparse.ArgumentParser()
    # diorectory that has dcm files 
    parser.add_argument('--dcm_dir', default = None, type = str)
    args = parser.parse_args()
    return args

class dcm_reader():
    def __init__(self, args):
        self.files = sorted(glob.glob(os.path.join(args.dcm_dir,'*.dcm')))
        self.nf = len(self.files) # count how many files we have
        print('number of files in directory:{}'.format(self.nf))
        self.read()

    def read(self):
        print('--'*8, 'Reading', '--'*8)
        f = open(self.files[0], 'rb')

        # remember f.seek(x) to jump to byte x
        preamble = list(f.read(128))
        print('preamble', preamble)

        f.seek(128)
        prefix = f.read(4)
        # for DICOM should be 4 chars = DICM
        # if not DICM not dicom. 
        print('prefix', prefix) 
        
        
        f.seek(132)
        file_meta_info_group_len = sum_bytes(f.read(2))
        print('file meta info group length ', file_meta_info_group_len)
        # close the file when done 
        f.close()


if __name__ == '__main__':
    print()
    args = set_parser() # get the main arguments.
    dr = dcm_reader(args) #