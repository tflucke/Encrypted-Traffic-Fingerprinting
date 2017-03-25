import matplotlib.pyplot as plt

packetrate_up = [0,20,50,100,150,200,300,400]
accuracy_up = [100,98.72,100,93.75,96.30,97.44,96.30,92.5]

packetrate_ud = [0,40,100,200,300,400,600,800]
accuracy_ud = [100,98.75,96.25,97.5,96.25,95,95,92.5]

plt.ylim([80,101])
plt.plot(packetrate_up, accuracy_up, label='Only upstream packets')
plt.plot(packetrate_ud, accuracy_ud, label='Upstream and downstream packets')
plt.title('The effect of background traffic on the classification accuracy')
plt.ylabel('Accuracy on the testset')
plt.xlabel('# packets of added background traffic')
plt.legend(loc=4)
plt.show()

