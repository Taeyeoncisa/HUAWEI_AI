import os
file = open('2019302110106.txt', mode='w')
for dir in os.listdir('expression'):
    file.write('label=0,shot=\''+dir+'\'\n')