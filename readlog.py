import sys

filename = str(sys.argv[1])
f = open(filename, 'r')
for i in range(22): trashbox = f.readline()
line  = f.readline()
while line != '':
    for notnumber in range(8):
        trashbox = f.readline()
    print float(line[18:-1])
    line = f.readline()

