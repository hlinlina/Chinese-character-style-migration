import matplotlib.pyplot as plt
import csv

csvfile = open('run_.-tag-l1_loss.csv', 'r')
plots = csv.reader(csvfile,delimiter=',')
x = []
y = []
for row in plots:
    y.append((row[2]))
    x.append((row[1]))

plt.plot(x,y)
plt.xlabel('Steps')
plt.ylabel('score')
plt.legend()
plt.show()
