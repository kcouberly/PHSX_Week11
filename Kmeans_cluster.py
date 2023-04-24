import numpy as np
import matplotlib.pyplot as plt
#K means algorithm from scikit learn
from sklearn.cluster import KMeans

#constructing data mixture

#3 clusters - centered around [1,1], [2,1], [2,2]
N = 100
data = []
for i in range(N):
    #using categorical dist to pick cluseter
    a = np.random.rand()
    if a < 0.3333:
        #centers around [1,1] (range 0.5 to 1.5)
        data.append(np.random.rand(2)+0.5)
    elif a < 0.6667:
        #centers around [2,2]
        data.append(np.random.rand(2)+1.5)
    else:
        #centers around [2,1]
        temp = np.random.rand(2)+0.5
        temp[0] += 1
        data.append(temp)

#Using scikit learn's KMeans clustering        
X = np.asarray(data)
kmeans = KMeans(n_clusters=3).fit(X)
#assigning data to clusters
clu1 = []
clu2 = []
clu3 = []
for i in range(len(data)):
    if kmeans.labels_[i] == 0:
        clu1.append(data[i])
    if kmeans.labels_[i] == 1:
        clu2.append(data[i])
    if kmeans.labels_[i] == 2:
        clu3.append(data[i])        


kmeans.predict([[0, 0], [3, 3],[3,1]])

centers = kmeans.cluster_centers_
print('predicted centers:',centers)
print('true centers: [2,1], [1,1], [2,2]')

#separating clusters into x,y coordinates
x_values = []
y_values = []
clusters = [clu1,clu2,clu3]
for y in range(3):
    tempx = []
    tempy = []    
    for i in range(len(clusters[y])):
        tempx.append(clusters[y][i][0])
        tempy.append(clusters[y][i][1])
    x_values.append(tempx)
    y_values.append(tempy)
    
plt.plot(x_values[0],y_values[0],'.',label='center [2,1]')
plt.plot(x_values[1],y_values[1],'.',label='center [1,1]')    
plt.plot(x_values[2],y_values[2],'.',label='center [2,2]')
plt.xlabel('x value')
plt.ylabel('y value')
plt.legend()
plt.title('KMeans cluster of mixture data')
plt.show()