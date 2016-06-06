'''
Created on 2016. 6. 6.

@author: root
'''

src= [1, 2, 3, 4]
index = [1, 0, 1, 0];
dst= [1, 2, 3, 4]
for j in range(0,4):
    if index[j]==0:
        dst[j] = 0;

print src

print index
print dst