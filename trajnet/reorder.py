import argparse
import os
import pickle
from os import listdir
from os.path import isfile, join

'''
Input file must be in the format:
frame1, object1, x1_1, y1_1
frame2, object1, x1_2, y1_2
frame1, object2, x2_1, y2_1

Creates output as a pickle in the form:
[[object1, x1_1, y1_1], [object2, x2_1, y2_1]]
[[object2, x2_2, y2_2]]
'''

parser = argparse.ArgumentParser(description='Reorder trajnet data to time->pedestrian')
parser.add_argument('--directory', metavar='G',
                    help='directory of samples to reorder')
args = parser.parse_args()

dir = os.path.join(os.path.dirname(__file__), args.directory)
files = [f for f in listdir(dir) if isfile(join(dir, f))]

output = []

for file in files:
    #Opening location is independant from where the script is called
    f = open(os.path.join(dir, file), 'r')

    # Reads and splits the 
    array = []
    lines = f.read()
    lines = lines.split('\n')

    for line in lines:
        array.append(line.split(' '))

    #Finds the max timestep in the file
    max = 0
    for i in array:
        if int(i[0]) > max:
            max = int(i[0])

    reordered = [[] for i in range(int(max/12)+1)]

    for i in range(len(array)):
        index = int(int(array[i][0])/12)
        reordered[index].append([array[i][1], array[i][2], array[i][3]])

    output.extend(reordered)

#Output file is reordered_{original}.p
pickle.dump((reordered), open(os.path.join(os.path.dirname(__file__), f'reordered_{args.directory}.p'), 'wb'))
