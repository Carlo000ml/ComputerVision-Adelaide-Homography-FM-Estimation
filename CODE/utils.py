import warnings
from copy import deepcopy
import cv2
import numpy as np
import visual as vi
from matplotlib import pyplot as plt


def compute_residual(src_, dst_, homography):
    """Given src points and dst points and the homograpgy, it returns the residuals associated to each pair of points
    
    
    Args:
      src_points: Source points as a NumPy array (shape: Nx2 or Nx3, where N is the number of points).
      dst_points: Destination points as a NumPy array (same shape as src_points).
      homography: The homography matrix as a NumPy array (shape: 3x3).

  Returns:
      A NumPy array containing the residual of each pair of points.
  """

    
    
    src_ = np.array(src_)
    dst_ = np.array(dst_)
    # Apply the homography to the source points
    projected = projectiveTransform(src_, homography)
    projected = projected.reshape(src_.shape[0], 2)

    # Compute the Euclidean distance between the projected points and the actual destination points
    residuals = (np.sum((projected - dst_) ** 2, axis=1)) ** 0.5

    return residuals


def verify_cv2_H(src_points, dst_points, ransacReprojThreshold=1):
    """Given src points and dst points and the ransac reprojection error threshold, it estimates the homography using RANSAC and returns labels for inliers and outliers
    
    
    Args:
      src_points: Source points as a NumPy array (shape: Nx2 or Nx3, where N is the number of points).
      dst_points: Destination points as a NumPy array (same shape as src_points).
      ransacReprojThreshold: The projection error threshold.

  Returns:
      H : homography, (3x3 numpy matrix)
      mask : labels for inlier and outliers    1-inlier ;  0-outlier (numpy array of shape (No_points , )
  """

    
    kps1, kps2, matches = build_keypts_matches(src_points, dst_points)
    src_pts = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold)
    inliers = deepcopy(mask).astype(np.float32).sum()
    print(inliers, 'inliers found')
    return H, mask.ravel()


def verify_LMEDS_H(src_points, dst_points, confidence=0.975, verbose=True):
    
        """Given src points and dst points and the confidence, it estimates the homography using LMEDS and returns labels for inliers and outliers
    
    
    Args:
      src_points: Source points as a NumPy array (shape: Nx2 or Nx3, where N is the number of points).
      dst_points: Destination points as a NumPy array (same shape as src_points).
      confidence: 

  Returns:
      H : homography, (3x3 numpy matrix)
      mask : labels for inlier and outliers    1-inlier ;  0-outlier (numpy array of shape (No_points , )
  """

    kps1, kps2, matches = build_keypts_matches(src_points, dst_points)
    src_pts = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS, confidence=confidence)
    if verbose: print(deepcopy(mask).astype(np.float32).sum(), 'inliers found')
    return H, mask.ravel()


def verify_cv2_FM(src_points, dst_points, ransacReprojThreshold=1):
    kps1, kps2, matches = build_keypts_matches(src_points, dst_points)
    src_pts = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, ransacReprojThreshold)
    inliers = deepcopy(mask).astype(np.float32).sum()
    print(inliers, 'inliers found')
    return H, mask.ravel()


def verify_LMEDS_FM(src_points, dst_points, confidence=0.975, verbose=True):
    kps1, kps2, matches = build_keypts_matches(src_points, dst_points)
    src_pts = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_LMEDS, confidence=confidence)
    if verbose: print(deepcopy(mask).astype(np.float32).sum(), 'inliers found')
    return H, mask.ravel()


def projectiveTransform(points, H):
        """Given a set of points in 2D coordinates it applies the homography H on them.
        Consider d=dimension of points, if 2D points d=2, if 3D points d=3. In our scenario d=2
    
    Args:
      points: Source points as a NumPy array (shape: Nx2 or Nx3, where N is the number of points). shape(N,2) for our instance
      H: Homography.

  Returns:
      (N,1,d) numpy array of points.
  """

    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), H)


def extract_points(models, data):
    
    """Given the models and the data folder it
    
    Args:
      models : list of coordinate points of len(models)= Number of models, organized as: assume we want to consider the model 0:
               models[0] is a list of len(models[0])=4 ;
               
               - models[0][0]= list of coordinates x of points of the model 0, in the first image
               - models[0][1]= list of coordinates y of points of the model 0, in the first image
               - models[0][2]= list of coordinates x of points of the model 0, in the second image
               - models[0][3]= list of coordinates y of points of the model 0, in the second image
               
      data : single data file from the folder of files

  Returns:
      points :  dictionary of 2D points coordinates for each model. keys= "src_points" and "dst_points"
              - src_points = list of arrays. One array for each model. src_points[0] is an array of points of model 0, shape (N , 2) each                                  point as an array of dim 2 [x_coord , y_coord]
              - dst_points = analogous for dst_points
              
      inliers1 : numpy array of dim (N,2) with N total number of points. It is a collection of all the points that are inliers in img1.                        Usefull for residual matrix computation
      inliers2: numpy array of dim (N,2) with N total number of points. It is a collection of all the points that are inliers in img2.                        Usefull for residual matrix computation
      
      labels of inliers : labels only for the inlier points. So for each point in inliers1 or inliers2 we know specifically its model.
  """
    points = {"src_points": [], "dst_points": []}
    for i in range(len(models)):
        src_points = np.array(list(zip(models[i][0], models[i][1])))

        dst_points = np.array(list(zip(models[i][2], models[i][3])))

        points["src_points"].append(src_points)
        points["dst_points"].append(dst_points)

    l = data["label"]
    p = data["data"]
    inl = np.where(l[0] != 0)
    inlp1_x = p[0][inl]
    inlp1_y = p[1][inl]
    inlp2_x = p[3][inl]
    inlp2_y = p[4][inl]

    inliers1 = np.array([inlp1_x, inlp1_y])
    inliers2 = np.array([inlp2_x, inlp2_y])

    return points, np.transpose(inliers1), np.transpose(inliers2), l[0][inl]


def build_keypts_matches(src_points, dst_points):
    
    
    src_kpts = [cv2.KeyPoint(x, y, 1) for x, y in src_points]
    dst_kpts = [cv2.KeyPoint(x, y, 1) for x, y in dst_points]
    assert len(src_kpts) == len(dst_kpts)
    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(dst_kpts))]

    return src_kpts, dst_kpts, matches


def draw_matches(img1, img2, src_points, dst_points, matchColor=(255, 255, 0), singlePointColor=None, flags=2,
                 mask=None):
    src_kpts, dst_kpts, matches = build_keypts_matches(src_points, dst_points)
    if mask is not None: mask = mask.ravel().tolist()

    draw_params = dict(matchColor=matchColor, singlePointColor=singlePointColor, flags=flags)
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.drawMatches(img1, src_kpts, img2, dst_kpts, matches, None, matchesMask=mask, **draw_params))
    return


def build_residual_matrix(data, plot=False, verbose=True, type='H'):
    
    """Given the data it automatically fit the homography or the fundamental matrix for each model and returns the residual matrix.
        It uses LMEDS.
    
    Args:
               
      data : single data file from the folder of files
      
      plot :  if to plot the points that are considered outliers for the model
      
      verbose : if to plot the total number of points of the model
      
      type :  H for Homography, FM for Fundamental matrix

  Returns:
      residual_matrix  : numpy array of shape (Number of points , Number of models). At position i,j there is the residual of point i for                              model j
  """
    
    img1, img2 = data["img1"], data["img2"]

    outliers, models = vi.group_models(data)["outliers"], vi.group_models(data)["models"]

    points = extract_points(models, data)

    tot_src = points[1]

    tot_dst = points[2]

    points = points[0]

    num_of_inliers = np.sum(data["label"] != 0)

    residual_matrix = np.zeros((num_of_inliers, len(models)))

    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    for i in range(len(models)):
        src = points["src_points"][i]
        dst = points["dst_points"][i]

        if type == 'H':
            cv2_M, cv2_mask = verify_LMEDS_H(src, dst, verbose=verbose)
        elif type == 'FM':
            cv2_M, cv2_mask = verify_LMEDS_FM(src, dst, verbose=verbose)
        else:
            warnings.warn("The given type is wrong. Types:\n'H'\n'FM'")
            return

        if verbose: print("the total number of point is: ", len(src))

        if plot:
            # random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            plt.figure()

            draw_matches(img1, img2, src, dst, matchColor=color[i % len(color)], mask=1 - cv2_mask)
            plt.show()

        if type == 'H':
            residual_matrix[:, i] = compute_residual(tot_src, tot_dst, cv2_M)
        elif type == 'FM':
            residual_matrix[:, i] = compute_residuals_FM(tot_src, tot_dst, cv2_M, 'sampson')

    return residual_matrix


def point_belongs_to_model(models, type='H', method='sampson', threshold=1):
    Ms = []
    masks = []

    for i in range(len(models)):
        src_points = np.array(list(zip(models[i][0], models[i][1])))
        dst_points = np.array(list(zip(models[i][2], models[i][3])))
        if type == 'H':
            M, mask = verify_cv2_H(src_points, dst_points)
        elif type == 'FM':
            M, mask = verify_cv2_FM(src_points, dst_points)
        else:
            warnings.warn("Attenzione: nessun tipo valido rilevato. Types: 'H' o 'FM'")
            return
        masks.append(mask)
        Ms.append(M)

    res = compute_residual_different_model(Ms, models, type, method)
    return extract_significant_point_idx(res, threshold)


def extract_significant_point_idx(res, threshold):
    result = {}
    for i in range(len(res)):
        for j in range(len(res[i])):
            if i != j:
                result["model" + str(i + 1) + "_point" + str(j + 1)] = [k for k in range(len(res[i][j])) if
                                                                        res[i][j][k] < threshold]

    return result


def compute_residual_different_model(M, models, type, method):
    residuals = np.zeros((len(models), len(M)), dtype=object)

    for i in range(len(models)):
        for j in range(len(M)):
            if i == j:
                residuals[i][j] = [0]
            else:
                if type == 'H':
                    residuals[i][j] = compute_residual(list(zip(models[i][0], models[i][1])),
                                                   list(zip(models[i][2], models[i][3])),
                                                   M[j])
                elif type == 'FM':
                    residuals[i][j] = compute_residuals_FM(list(zip(models[i][0], models[i][1])),
                                                           list(zip(models[i][2], models[i][3])),
                                                           M[j], method)

    return residuals


def compute_inliers_residual_curve(data, res=None, type='H'):
    """Given the data it automatically estimates the homography or the fundamental matrix, compute the residuals and plot the residual curves.        returns residuals of inliers in incremental order.
    
    Args:
               
      data : single data file from the folder of files
      
      type : H for Homography, FM for Fundamental Matrix

  Returns:
     inlier_residuals : list of numpy arrays. One array for model. Each array contains residuals in incremental order.
  """
    outliers, models = vi.group_models(data)["outliers"], vi.group_models(data)["models"]

    points = extract_points(models, data)

    labels = points[3]

    inlier_residuals = []

    if res is None:
        res = build_residual_matrix(data, plot=False, verbose=False, type=type)

    for i in range(res.shape[1]):  # for i in range(num of models)

        mod_inliers = np.where(labels == i + 1)[0]  # ground truth of model points

        residuals = np.sort(res[mod_inliers, i])

        inlier_residuals.append(residuals)

    return inlier_residuals


def plot_inliers_residual_curves(data, res=None):
    """ Plot the residuals in incremental order, for each model in data. Coordinate x is an arange (1,2,3,...etc), coordinate y is -residual value. The negative sign is just because we liked it to be like this ;) """ 
    residuals = compute_inliers_residual_curve(data, res=res)

    for i in range(len(residuals)):
        mod = residuals[i]

        x_axis = np.arange(len(mod))

        plt.figure()
        plt.scatter(x_axis, -mod, color="b", marker='o')
        plt.xlabel('Points')
        plt.ylabel('Scores')
        plt.title('Scatter Plot of Points')

    return


def plot_res_curve(res, mask, title='Scatter Plot of Points'):
    x_axis = np.arange(len(res))
    blues = np.where(mask == 0)[0]
    reds = np.where(mask == 1)[0]

    plt.figure()
    plt.scatter(x_axis[blues], -res[blues], color="b", marker='o')
    plt.scatter(x_axis[reds], -res[reds], color="r", marker='o')
    plt.xlabel('Points')
    plt.ylabel('Scores')
    plt.title(title)

    return


def calculate_reprojection_error(src_points, dst_points, M, inlier_mask, type):
    """
  This function calculates the reprojection error for points transformed using a homography.

  Args:
      src_points: Source points as a NumPy array (shape: Nx2 or Nx3, where N is the number of points).
      dst_points: Destination points as a NumPy array (same shape as src_points).
      M: The homography matrix as a NumPy array (shape: 3x3).
      inlier_mask: A mask indicating inlier points (1 for inlier, 0 for outlier) as a NumPy array (same length as src_points).

  Returns:
      A NumPy array containing the reprojection errors for each point (only for inliers based on the mask).
  """

    if type == 'H':
        # Project source points using the homography
        #projected_points = cv2.perspectiveTransform(src_points[inlier_mask].reshape(-1, 1, 2), M)

        # Calculate the reprojection error (distance between actual and projected destination points)
        #reprojection_errors = np.linalg.norm(projected_points[:, 0, :] - dst_points[inlier_mask], axis=1)
        reprojection_errors = compute_residual(src_points[inlier_mask], dst_points[inlier_mask], M)
    elif type == 'FM':
        reprojection_errors = compute_residuals_FM(src_points[inlier_mask], dst_points[inlier_mask], M)
    else:
        return
    return reprojection_errors


def analyze_reprojection_error(data, thresholds=[1], method="LMEDS", type='H'):
    """
  This function analyzes the average reprojection error for inliers after fitting with LMEDS at different thresholds.

  Args:
      data: A tuple containing source and destination points (src_points, dst_points).
      thresholds: A list of inlier thresholds to evaluate.

  Returns:
      A dictionary where keys are thresholds and values are the average reprojection errors for inliers at that threshold.
  """
    outliers, models = vi.group_models(data)["outliers"], vi.group_models(data)["models"]

    points = extract_points(models, data)
    src_points, dst_points = points[1], points[2]
    average_errors = {}
    for threshold in thresholds:
        assert method.upper() == "LMEDS" or method.upper() == "RANSAC"
        if type == 'H':
            # Fit the model (homography) using LMEDS
            if method.upper() == "LMEDS":
                M, mask = verify_LMEDS_H(src_points, dst_points)
            elif method.upper() == "RANSAC":
                M, mask = verify_cv2_H(src_points, dst_points, ransacReprojThreshold=threshold)
        elif type == 'FM':
            if method.upper() == "LMEDS":
                M, mask = verify_LMEDS_FM(src_points, dst_points)
            elif method.upper() == "RANSAC":
                M, mask = verify_cv2_FM(src_points, dst_points, ransacReprojThreshold=threshold)
        inlier_mask = mask.ravel() > 0  # Convert mask to 1D array with True for inliers

        # Calculate reprojection errors and compute the average for inliers only
        reprojection_errors = calculate_reprojection_error(src_points, dst_points, M, inlier_mask, type)
        average_errors[threshold] = np.mean(reprojection_errors)
    return average_errors


def compute_sampson_distance(src_points, dst_points, F):
    src_ = np.array(src_points)
    dst_ = np.array(dst_points)
    F = np.array(F, dtype=np.float64)

    # Convert source point to homogeneous coordinates
    src_points_hom = np.array([np.append(src_point, 1) for src_point in src_], dtype=np.float64)

    # Convert destination points to homogeneous coordinates
    dst_points_hom = np.array([np.append(dst_point, 1) for dst_point in dst_], dtype=np.float64)

    return np.array([cv2.sampsonDistance(src_point_hom, dst_point_hom, F) for src_point_hom, dst_point_hom in zip(src_points_hom, dst_points_hom)])


def distance_point_line(ps, ls):

    return np.array([np.dot(np.array(l),np.array(p).T)/np.sqrt(l[0]**2+l[1]**2) for l, p in zip(ls, ps)])


def compute_SED(src_points, dst_points, F):
    src_ = np.array(src_points)
    dst_ = np.array(dst_points)

    # Convert source point to homogeneous coordinates
    src_points_hom = np.array([np.append(src_point, 1) for src_point in src_])

    # Convert destination points to homogeneous coordinates
    dst_points_hom = np.array([np.append(dst_point, 1) for dst_point in dst_])

    # Compute the epilines of the source points
    ep_ls = np.array([np.dot(F.T, dst_point_hom.T) for dst_point_hom in dst_points_hom])

    # Compute the epilines of the destination points
    ep_ls_prime = np.array([np.dot(F, src_point_hom.T) for src_point_hom in src_points_hom])

    return np.sqrt(distance_point_line(src_points_hom, ep_ls)**2 + distance_point_line(dst_points_hom, ep_ls_prime)**2)


def compute_residuals_FM(src_points, dst_points, F, method='sampson'):
    if method == 'sampson':
        return compute_sampson_distance(src_points, dst_points, F)
    if method == 'sed':
        return compute_SED(src_points, dst_points, F)
