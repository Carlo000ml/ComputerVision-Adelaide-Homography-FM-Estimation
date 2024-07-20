import numpy as np

from stats import *

class Inlier_Thresholder:

    ########### initialize the object with the 1D array of values
    def __init__(self, values, n_inliers=None, n_outliers=None, type="FM"):
        self.values = values
        self.threshold = None
        if type=="FM":
            self.methods = ["Median AD", "Rosseeuw SN", "Rosseeuw QN", "Forward Search"]#, "IQR"]#, "Variance based"]#, "First Jump","DBSCAN"]
            
        if type=="H":
            self.methods = ["Median AD", "Rosseeuw SN", "Rosseeuw QN", "IQR" , "Variance based"]#, "First Jump","DBSCAN"]
           
         
        self.internal_validation_measures = ["Silhouette", "BSS", "WSS"]
        self.n_inliers = n_inliers
        self.n_outliers = n_outliers

    ########### specify the method among the available ones and return the labels
    def compute_inlier_threshold(self, method):

        assert method in self.methods

        if method == "IQR":
            return interquantile_outlier(self.values)
        if method == "DBSCAN":
            return anomaly_detection_DBSCAN(self.values)
        if method == "Median AD":
            return Median_Absolute_Deviation(self.values)
        if method == "Variance based":
            return Variance_based(self.values)

        if method == "First Jump":
            return First_Jump(self.values)

        if method == "Rosseeuw SN":
            return rousseeuwcroux_SN(self.values)
        if method == "Rosseeuw QN":
            return rousseeuwcroux_QN(self.values)
        if method == "Forward Search":
            tot = self.n_inliers + self.n_outliers
            if self.n_inliers is not None and self.n_outliers is not None:
                #return forward_search(self.values, m0=self.n_outliers, percentile=(self.n_inliers / tot) * 100)
                return forward_search(self.values, initial_m0=self.n_outliers)
            elif self.n_inliers is None and self.n_outliers is None:
                return forward_search(self.values)

        return

    ########### Majority voting ensemble
    def ensemble_inlier_thresholder(self):

        L = []

        for met in self.methods:
            tmp, _ = self.compute_inlier_threshold(met)
            threshold = np.array(tmp)
            threshold[threshold == None] = 0.90
            L.append(threshold)

        sums = np.array([sum(values) for values in zip(*L)])

        #### sums>=5 if more than 4 models agree
        return (sums >= 5).astype(int)

    ############## Select the internal validation measure as an objective function
    ############## Return the method that optimize the internal validation measure
    ############## Default is Silhouette: empirically the best one

    def use_best_method(self, verbose=False, internal_validation_measure="Silhouette"):

        assert internal_validation_measure in self.internal_validation_measures

        score_sil = []  # score for the silhouette

        score_sep = []  # score for the separation BSS

        score_coh = []  # score for the cohesion WSS
        
        thresh=[]

        for met in self.methods:
            lab, _ = self.compute_inlier_threshold(met)
            thresh.append(_)

            if len(np.unique(lab)) == 1:
                score_sil.append(0), score_sep.append(0)
            else:

                silhouette_avg = silhouette_score(self.values.reshape(-1, 1), lab)
                wss, bss = compute_wss_bss(self.values, lab)

                score_sil.append(silhouette_avg)
                score_sep.append(bss)
                score_coh.append(wss)
                

        lab = self.ensemble_inlier_thresholder()

        if len(np.unique(lab)) == 1:
            score_sil.append(0), score_sep.append(0)
        else:
            silhouette_avg = silhouette_score(self.values.reshape(-1, 1), lab)
            wss, bss = compute_wss_bss(self.values, lab)

            score_sil.append(silhouette_avg)
            score_sep.append(bss)
            score_coh.append(wss)
 
        

        if internal_validation_measure == "Silhouette": to_use = self.fair_silhouette(score_sil[:-1],thresh)
        if internal_validation_measure == "BSS": to_use = np.argmax(score_sep)
        if internal_validation_measure == "WSS": to_use = np.argmin(score_coh)

        if to_use < len(self.methods):
            if verbose:
                print(self.methods[to_use])

            return self.compute_inlier_threshold(self.methods[to_use])

        if to_use == len(self.methods):
            if verbose:
                print("Ensemble")
            return self.ensemble_inlier_thresholder()
        
    def fair_silhouette(self,score_sil, thresh):
        thresh=np.array(thresh)

        max_sil=np.max(score_sil)

        model_indexes=np.where(score_sil==max_sil)

        if len(model_indexes[0])==1:
            return np.argmax(score_sil)

        else:
            best_threshold=np.max(thresh[model_indexes])
            return np.where(thresh==best_threshold)[0][0]
                            
