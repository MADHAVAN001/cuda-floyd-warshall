import os

for j in range(100,4800,100):
    for i in range(1,13):
        os.environ["OMP_NUM_THREADS"]=str(i)
        os.system("./a.out "+str(j))
    print "Done for matrix size: " + str(j)
    f = open('dynamic.txt','a')
    f.write('Matrix SIZE completed:'+str(j))
    f.write('\n\n')
    f.close()
