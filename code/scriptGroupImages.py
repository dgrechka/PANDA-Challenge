import os
import sys

# this script is intended to group flattened images (e.g. hunded thousands)  into a
# directory tree structure (only single level deep) by using first 2 latters of the filename
# as a directory to move the file to
# filenames strarting with 00 - ff directories. 256 in total

targetDir = sys.argv[1]
print("Dir to work with: {0}".format(targetDir))

filenames = os.listdir(targetDir)
filenames = [x for x in filenames if x.endswith(".tiff")]
print("{0} files to group into dirs".format(len(filenames)))

# TODO: finish. 