import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to get the euclidean distance for any n dimensional vectors of the same length
def euclid_nd(a,b):
    
    # Get the sum of squares for all n dimensions of a and b, where a is an object of x
    # and b is a centroid
    sum_of_squares = 0
    for n in range(len(a)):
        sum_of_squares += (b[n] - a[n])**2
    
    # return square root of the sum of squares
    return math.sqrt(sum_of_squares)


# Function for determining the closest of centroids for each object in x based on euclidean distance
'''
Note that no code change is needed from the example "kmeans.py"
Assignments for objects in x are returned in vectors regardless of the feature dimensions of x

'''

def closest_nd(x, centroids):
    
    # List for holding cluster assignments for every object in x
    assignments = []
    
    # Iterate over every object in x
    for i in x:
        
        # distance between one data point and centroids
        distance=[]
        
        # Iterate over every centroid
        for j in centroids:
            
            distance.append(euclid_nd(i, j))
            
        # assign each data point to the cluster with closest centroid   
        assignments.append(np.argmin(distance))

    return np.array(assignments)


# Updates the centroids based on assigned clusters from function closest_nd
def update_nd(x, clusters, K):
    
    # Build np array of centroids for number of axes * clusters
    new_centroids = np.zeros(shape=(K,x.shape[1]))

    # Get cluster mean(s) for each cluster K
    for c in range(K):

        # For each feature dimension (axis) of x
        for n in range(x.shape[1]):

            # Get mean for cluster along the nth axis and assign to centroid
            new_centroids[c,n] = x[clusters == c,n].mean()
    
    return new_centroids


# Function for assigning clusters as labels for a new 
def label_clusters(x, clusters):
    
    # Add cluster label to each onject
    # reshapes clusters to len of x
    # uses hstack to add each cluster label as a new column of x
    labeled_x = np.hstack((x,clusters.reshape(len(x),1)))
    return labeled_x
    
    
# k-means main function
def kmeans_nd(x, K):
    
    # initialize the centroids of K clusters with a range of max of x plus 10%
    centroids = round(1.1 * max(x.max(axis = 0))) * np.random.rand(K, x.shape[1])
    print('Initialized centroids: {}'.format(centroids))
    
    # Assign clusters for intial centroids
    clusters = closest_nd(x, centroids)
    print(clusters)
    
    # Iterate to find minimums, 10 should be enough
    for i in range(10):
        clusters = closest_nd(x, centroids)
        centroids = update_nd(x, clusters, K)
        print('Iteration: {}, Centroids: {}'.format(i, centroids))
    
    print('\nFinal Centroids: {}\nFinal Clusters: {}'.format(centroids, clusters))
    
    # Return cluster labeled dataset and final centroids
    return label_clusters(x, clusters), centroids



### Driver code ###

x = np.array([[2,4],[1.7,2.8],[7,8],[8.6,8],[3.4,1.5],[9,11]])
K = 2

# Get labeled set and centoids for plotting
labeled, centroids = kmeans_nd(x, K)



# Convert x and centroids to dataframes
labeled_df = pd.DataFrame(labeled, columns = ['Feature_1','Feature_2','Cluster'])
centroid_df = pd.DataFrame(centroids, columns = ['Feature_1', 'Feature_2'])

print("\nLabeled dataset:")
print(labeled_df)
print("\nCentroids:")
print(centroid_df)

# Print plot for clusters, centroids
sns.scatterplot(data=labeled_df, x="Feature_1", y="Feature_2", hue="Cluster")
sns.scatterplot(data=centroid_df, x="Feature_1", y="Feature_2", color = "r", label = "Centroid")
plt.show()