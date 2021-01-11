import os

count = 0
for filename in os.listdir(os.getcwd()):
   with open(os.path.join(os.getcwd(), filename), 'r') as f: # open in readonly mode
      l = f.readlines()
      count += len(l)

print(count)