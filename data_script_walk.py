import os
import sys
path = sys.argv[1]
print(path)
for root, dirs, files in os.walk(path, topdown=False):
	for name in files:
		print(os.path.join(root, name))
   # for name in dirs:
   #    print(os.path.join(root, name))
	for file in files:
		location = os.path.join(root, file)
		if "infrared" in location:
			

