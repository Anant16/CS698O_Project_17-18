
import os
import shutil
import sys
dstA = "testA"
dstB = "testB"
ind = 0
num = ""

path_to_data = sys.argv[1]

# os.makedirs(dstA)
# os.makedirs(dstB)
# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk(path_to_data):
    path = root.split(os.sep)
    # print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
        # print(len(path) * '---', file)
        # print(path)
        # print(root)
        # print(file)
        # if ind == 10:
        # 	sys.exit()
        if ".BMP" in file:
        	# ind+=1
            # print(path[9])
		    if "visible" in path and "without_glasses" in path:
				if path[2] == num:
					ind+=1
				else:
					num = path[2]
					ind = 1
				if ind >2:
					continue
				try:
					shutil.copy(root+"/"+file,dstA)
				except:
					print(shutil.Error)

        	elif "infrared" in path :#and "without_glasses" in path:
        		# try:
        		# shutil.copy(root+"/"+file,dstB)
        		# print("\t\t\t\t\t" + path)
        		# except:
        			# print(shutil.Error)