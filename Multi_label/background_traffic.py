import matplotlib.pyplot as plt

packetrate_ud = [0,40,100,200,400,600,800]
accuracy_ud = [83.87, 81.65, 70.25, 68.04, 63.26, 68.99, 56.65]
loss_ud = [0.04619, 0.04984, 0.09019, 0.09019, 0.09968, 0.08940, 0.13449]

#plt.ylim([80,101])
#plt.plot(packetrate_ud, accuracy_ud, label='Upstream and downstream packets')
#plt.plot(packetrate_ud, accuracy_ud, label='Upstream and downstream packets')
#plt.title('The effect of background traffic on the classification accuracy')
#plt.ylabel('Accuracy on the testset')
#plt.xlabel('# packets/sec of added background traffic')
#plt.legend(loc=4)
#plt.show()


fig, ax1 = plt.subplots()
acc = ax1.plot(packetrate_ud, accuracy_ud, 'b-', label='Subset accuracy')
ax1.set_xlabel('# packets/sec of added background traffic')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Subset Accuracy on the testset')
#ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
loss = ax2.plot(packetrate_ud, loss_ud, 'r-', label='Hamming loss')
ax2.set_ylabel('Hamming loss on the testset')
#ax2.tick_params('y', colors='r')

lns = acc+loss
labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc=9)

fig.tight_layout()

plt.show()