import math
import statistics as st
import numpy as np
import scipy.stats
from sklearn.cluster import DBSCAN
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_samples, silhouette_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler

from utils import *
from scipy.interpolate import interp1d


def median_neighbor_distance(values):
    # Calculate all pairwise distances
    consec_distances = []

    for i in range(1, len(values)):
        d = values[i] - values[i - 1]
        consec_distances.append(d)

    median_distance = st.median(consec_distances)

    return median_distance


def median_minpts(values, dist):
    neighbor_pts = []

    for i in range(len(values)):
        neighbor_pts.append(np.sum(abs(values - values[i]) <= dist))

    return st.median(neighbor_pts)


######## DBSCAN - Density Based Clustering method

def anomaly_detection_DBSCAN(values):
    # compute the median of the consecutive distances
    v = values.reshape(-1, 1)
    m = median_neighbor_distance(values)
    min_pts= median_minpts(values, 40 * m)

    # apply DBSCAN
    db = DBSCAN(eps=40 * m, min_samples=int(min_pts - 0.1 * min_pts)).fit(v)
    labels = db.labels_

    return abs(labels), None


######## Inter-Quantile method


def interquantile_outlier(values):
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)

    IQR = Q3 - Q1

    upperbound = Q3 + 1.5 * IQR

    return abs((values < upperbound).astype(int) - 1), upperbound


######## Statistical methods


### MAD


def Median_Absolute_Deviation(values, alpha=2.9):
    upper = stats.median_abs_deviation(values, scale='normal')

    upper = np.median(values) + alpha * upper

    return abs((values < upper).astype(int) - 1), upper


### Variance based

def Variance_based(values, alpha=1.5):
    std = np.std(values)

    upper = np.mean(values) + alpha * std  # 2 very good

    return abs((values < upper).astype(int) - 1), upper


### "Naive" method: tries to compute the first significant "jump" and from that point on, all values are classified outliers

def First_Jump(values):
    m = median_neighbor_distance(values)

    limit = False

    labels = np.zeros(len(values))
    for i in range(len(values) - 1):
        if values[i + 1] - values[i] > 20 * m:
            limit = True

        if limit:
            labels[i + 1] += 1

    return labels, None


### SN estimation method

def rousseeuwcroux_SN(values, alpha=3):
    SN = []

    for i in range(len(values)):
        tem = []
        for j in range(len(values)):
            if i != j:
                tem.append(abs(values[i] - values[j]))

        SN.append(np.median(tem))

    std_est = c(len(values)) * 1.1926 * np.median(SN)

    upper = np.median(values) + alpha * std_est

    return abs((values < upper).astype(int) - 1), upper


### QN estimation method

def rousseeuwcroux_QN(values , alpha=3):
    n = len(values)
    A = np.abs(np.subtract.outer(values, values))
    y = A[np.triu_indices(n, k=1)]
    d = _d(n)
    std_est = d * 2.2219 * np.percentile(y, 25)

    upper = np.median(values) + alpha * std_est

    return abs((values < upper).astype(int) - 1), upper


################################################ auxilary functions
def _d(n):
    table = [1, 0.399, 0.994, 0.512, 0.844, 0.611, 0.857, 0.669, 0.872]
    if n % 2:
        return n / (n + 1.4)
    else:
        return n / (n + 3.8) if n > 0 else 1


def c(n):
    # finite sample correction for Sn
    table = [1, 0.743, 1.851, 0.954, 1.351, 0.993, 1.198, 1.005, 1.131]

    if n % 2 == 1:
        c = n / (n - 0.9)  # odd
    else:
        c = 1  # even

    if n < 9:
        c = table[n - 1]

    return c


####### just the plot of the silhouette
def silhouette_analysis(values, labels):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    X = values.reshape(-1, 1)
    x_axis = np.arange(len(X))

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]c
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (2 + 1) * 10])

    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    y_lower = 10
    for i in range(2):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / 2)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(labels.astype(float) / 2)
    ax2.scatter(x_axis, -X, marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    # Draw white circles at cluster centers

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis - Silhouette score: ", silhouette_avg),
                 fontsize=14, fontweight='bold')

    return silhouette_avg


def silhouette_score_and_average(values, labels):
    """
    This function calculates the silhouette score for each data point and returns the average score.

    Args:
        values: The data points used for clustering.
        labels: The cluster labels for each data point.

    Returns:
        A tuple containing:
            - The silhouette score for each data point as a NumPy array.
            - The average silhouette score.
    """
    X = values.reshape(-1, 1)  # Reshape for silhouette_score
    silhouette_values = silhouette_samples(X, labels)
    silhouette_avg = silhouette_score(X, labels)
    return silhouette_values, silhouette_avg, X


def plot_silhouette(silhouette_values, labels, silhouette_avg, values, threhsold):
    """
    This function creates the silhouette plot and cluster visualization.

    Args:
        silhouette_values: Silhouette score for each data point as a NumPy array.
        labels: The cluster labels for each data point.
        silhouette_avg: The average silhouette score.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # Configure and plot the silhouette scores
    x_axis = np.arange(len(silhouette_values))
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(silhouette_values) + (2 + 1) * 10])

    y_lower = 10
    for i in range(2):
        ith_cluster_silhouette_values = silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / 2)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # Plot the data points with cluster colors
    colors = cm.nipy_spectral(labels.astype(float) / 2)
    ax2.scatter(x_axis, -values, marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
    ax2.axhline(y=threhsold, color="purple" , linestyle="--")

    # Labeling (cluster centers not implemented here for brevity)
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis - Silhouette score: ", silhouette_avg),
                 fontsize=14, fontweight='bold')
    plt.show()


def compute_wss_bss(data, labels):
    """
    Compute the within sum of squares (WSS) and between sum of squares (BSS).

    Parameters:
    - data: 1D numpy array containing the data points (each row represents a data point).
    - labels: 1D numpy array containing the cluster labels for each data point.

    Returns:
    - wss: Within sum of squares.
    - bss: Between sum of squares.
    """
    labels = (labels).astype(int)
    # Compute the centroids of each cluster
    centroids = [np.mean(data[labels == label]) for label in np.unique(labels)]

    cardinality = [np.sum(labels == label) for label in np.unique(labels)]

    # Compute the total centroid (mean) of all data points
    total_centroid = np.mean(data)

    # Compute the between sum of squares (BSS)
    bss = np.sum(cardinality * (centroids - total_centroid) ** 2)

    # Compute the within sum of squares (WSS)
    wss = sum(np.sum((data[labels == label] - centroids[label]) ** 2) for label in np.unique(labels))

    return wss, bss

########################################## Average within-cluster distance

def AWCD(residual_matrix , partition_matrix,fuzzifier=2):  # the lower the better
    """
    Compute the Avarage within-cluster distance

    Parameters:
    - residual_matrix
    - Partition_matrix
    - fuzzifier ----- The fuzzifier is a parameter controlling the "softness" of the clustering, as it approaches to 1, the clustering gets hard (element i,j either 0 or 1), experimentally a value of fuzzifier=2 gives good results. The same fuzzifier of the partition matrix muste be used
    
    Returns:
    - AWCD 
    """
    N,c=residual_matrix.shape[0],residual_matrix.shape[1]
    
    AWCD=0
    
    for i in range(c):
        numerator=0
        denominator=0
        
        for j in range(N):
            denominator+=partition_matrix[j,i]**fuzzifier
            numerator+=(residual_matrix[j,i]**2)*partition_matrix[j,i]**fuzzifier
            
        
        AWCD+=(1/c)* numerator/denominator
    return AWCD
    
    
    
#########################################  Fuzzy silhouette
def Fuzzy_silhouette(residual_matrix , partition_matrix ,alpha=1, fuzzifier=2): # the larger the better
    """
    Compute the Avarage within-cluster distance

    Parameters:
    - residual_matrix
    - Partition_matrix
    - fuzzifier ----- The fuzzifier is a parameter controlling the "softness" of the clustering, as it approaches to 1, the clustering gets hard (element i,j either 0 or 1), experimentally a value of fuzzifier=2 gives good results. The same fuzzifier of the partition matrix muste be used
    - alpha ----- alpha is a user defined parameter, by default alpha=1 is used
    
    Returns:
    - AWCD 
    """
    
    
    N=partition_matrix.shape[0]
    c=partition_matrix.shape[1]
    if c==1: return 1  # single model
    
    labels=np.zeros(N)

    for i in range(N):
        labels[i]=np.argmax(partition_matrix[i,:])
        
    # compute silhouette values using the hard clustering
        
    avarage_intracluster_distance=np.zeros(N)
    for i in range(N):
        l=labels[i].astype(int)
        avarage_intracluster_distance[i]=np.mean(abs(residual_matrix[i,l]-residual_matrix[np.where(labels==l),l]))
        
    
        
    avarage_intercluster_distance=np.zeros(partition_matrix.shape)+np.inf
    
    for i in range(N):
        l=labels[i]
        for j in range(c):
            if j!=l: avarage_intercluster_distance[i,j]=np.mean(abs(residual_matrix[i,j]-residual_matrix[np.where(labels==j),j]))
                
                
    si=np.zeros(N)
    for i in range(N):
        a=avarage_intracluster_distance[i]
        b=np.min(avarage_intercluster_distance[i,:])
  
        si[i]=(b-a)/np.max(np.array([a,b]))
        

        
    
    P_mat=partition_matrix.copy()
    numerator=0
    denominator=0
    for i in range(N):
        mu_p_index=np.argmax(P_mat[i,:])
        mu_p=P_mat[i,mu_p_index]
        P_mat[i,mu_p_index]=-np.inf
        mu_q=np.max(P_mat[i,:])
        
        numerator+=((mu_p-mu_q)**alpha) *si[i]
        denominator+=((mu_p-mu_q)**alpha)
        

        
    FS=numerator/denominator
        
                
    return FS


def t_test(group1, group2):
    """
    Perform a two-sample t-test.

    Parameters:
    group1 (array-like): First group of data.
    group2 (array-like): Second group of data.

    Returns:
    t_statistic (float): The calculated t-statistic.
    p_value (float): The two-tailed p-value.
    """

    t_statistic, p_value = stats.ttest_ind(group1, group2)
    return t_statistic, p_value


def f_test(sample1, sample2):
    """
        Perform F-test for variance.

        Parameters:
        - sample1, sample2: NumPy arrays representing the samples to be compared.

        Returns:
        - result: A tuple containing the F-statistic and the p-value.
        """

    # Compute variances
    var1 = np.var(sample1, ddof=1)  # Use sample variance with Bessel's correction
    var2 = np.var(sample2, ddof=1)

    # Compute F-statistic
    F = var1 / var2 if var1 >= var2 else var2 / var1

    # Compute degrees of freedom
    df1 = len(sample1) - 1
    df2 = len(sample2) - 1

    # Compute p-value
    p_value = 2 * min(stats.f.cdf(F, df1, df2), 1 - stats.f.cdf(F, df1, df2))

    return F, p_value


def forward_search(residuals, initial_m0=2, initial_percentile=95, alpha=0.05, smoothing_factor=0.9):
    """
    Perform Forward Search with dynamic adjustment of m0.

    Args:
    residuals: array of residuals from the model
    initial_m0: initial number of expected outliers
    percentile: the percentage of inliers expected
    alpha: significance level for the chi-square test

    Returns:
    outlier_mask: boolean mask indicating outliers (True) and inliers (False)
    threshold: final threshold used for outlier detection
    """
    n_samples = len(residuals)
    max_iter = n_samples - 1

    # Use Median Absolute Deviation as a robust estimator
    mad = stats.median_abs_deviation(residuals)

    # Initial core set with smallest squared residuals
    core_set = residuals[np.argsort(residuals ** 2)[:len(residuals)//7]]

    m0 = initial_m0
    inlier_indices = set()
    outlier_indices = set()
    current_percentile = initial_percentile

    for iteration in range(max_iter):
        # Standardized residuals based on core set
        std_residuals = (residuals - np.mean(core_set)) / mad

        # Sort standardized residuals
        sorted_indices = np.argsort(np.abs(std_residuals))

        # Identify potential outliers
        potential_outliers = sorted_indices[-m0:]

        # Update sets
        new_inliers = set(sorted_indices[:-m0]) - inlier_indices
        inlier_indices.update(new_inliers)
        outlier_indices.update(potential_outliers)

        # Update core set
        core_set = residuals[list(inlier_indices)]

        # Update current percentile
        new_percentile = np.floor((len(inlier_indices) / n_samples) * 100)
        current_percentile = smoothing_factor * current_percentile + (1 - smoothing_factor) * new_percentile
        current_percentile = np.floor(max(min(current_percentile, 99), 80))  # Constrain between 80 and 99

        # Dynamic adjustment of m0
        if len(core_set) > 2:  # Ensure we have enough points for chi-square test
            chi2_statistic = np.sum((std_residuals[list(inlier_indices)] ** 2))
            p_value = 1 - stats.chi2.cdf(chi2_statistic, len(core_set) - 1)

            if p_value < alpha:
                m0 = min(m0 + 1, n_samples - len(core_set))
            else:
                m0 = max(initial_m0, m0 - 1)

        # Stopping criterion
        if len(potential_outliers) == 0 or len(inlier_indices) + m0 == n_samples:
            break

    # Final threshold computation
    threshold = np.percentile(np.abs(residuals), current_percentile)

    # Create outlier mask
    outlier_mask = np.abs(residuals) > threshold

    return outlier_mask, float(threshold)

