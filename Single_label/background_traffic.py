import matplotlib.pyplot as plt

packetrate_up = [0,20,50,100,150,200,300,400]
accuracy_up = [96.40,98.88,98.75,95,93.75,90,95,85]

packetrate_ud = [0,40,100,200,300,400,600,800]
accuracy_ud = [96.40,100,98.75,98.75,87.5,96.25,86.25,81.25]

plt.ylim([75,101])
plt.plot(packetrate_up, accuracy_up, label='Only upstream packets')
plt.plot(packetrate_ud, accuracy_ud, label='Upstream and downstream packets')
plt.title('The effect of background traffic on the classification accuracy')
plt.ylabel('Accuracy on the testset')
plt.xlabel('# packets/s of added background traffic')
plt.legend(loc=4)
plt.show()

