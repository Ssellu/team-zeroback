import sys
import os
from collections import defaultdict


def count_numnber_of_each_label():
    """ Returns an integer list of the number of labelled boxes of each class.
    The length of the list is 7 in order to the number of class is 7.
    """

    # annotations path
    dir_path = './dataset/annotations'

    # list to store files
    res = [0,0,0,0,0,0,0]

    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        annotation_file_path = os.path.join(dir_path, path)
        if os.path.isfile(annotation_file_path):
            with open(annotation_file_path, 'r') as f:
                lst = [int(x.split(" ")[0]) for x in f.readlines()]
                for n in lst:
                    res[n] += 1
    print('0(left):{} \n1(right):{} \n2(stop):{} \n3(crosswalk):{} \n4(uturn):{} \n5(traffic_light):{} \n6(ignore):{}'.
          format(res[0],res[1],res[2],res[3],res[4],res[5], res[6]))
    return res

if __name__ == '__main__':
    count_numnber_of_each_label()