import matplotlib.pyplot as plt
import numpy as np


def group_models(data):

    keypts=data["data"]
    
    lbl=data["label"]

    max_mod=np.max(lbl)

    groups=[]

    for i in range(max_mod+1):
        temp_group=np.where(lbl[0]==i)
        groups.append(temp_group)

    #outliers
    outliers_img1_x=keypts[0][groups[0]]
    outliers_img1_y=keypts[1][groups[0]]

    outliers_img2_x=keypts[3][groups[0]]
    outliers_img2_y=keypts[4][groups[0]]

    outliers=np.array([outliers_img1_x , outliers_img1_y , outliers_img2_x , outliers_img2_y])



    models=[]

    for i in range(1,max_mod+1):
        model_img1_x=keypts[0][groups[i]]
        model_img1_y=keypts[1][groups[i]]

        model_img2_x=keypts[3][groups[i]]
        model_img2_y=keypts[4][groups[i]]

        model=np.array([model_img1_x,model_img1_y,model_img2_x, model_img2_y])

        models.append(model)
        
    to_ret={'outliers': outliers , "models" : models}
    
    return to_ret

def plot_images(data):
    img1=data['img1']
    img2=data['img2']

    
    objects=group_models(data)
    models=objects["models"]
    outliers=objects["outliers"]
    
    fig,axes= plt.subplots(1,2,figsize=(10,5))
    

    axes[0].imshow(img1)
    axes[0].scatter(outliers[0] , outliers[1] , s=10 , c="white" , marker='o' , label="Outliers")
    
    for i in range(len(models)):
        axes[0].scatter(models[i][0] , models[i][1] , s=10 ,  marker='o' , label=f"Model {i+1}")

        


    axes[0].set_title("Image 1")
    axes[0].legend(loc='upper left')
    axes[0].axis('off')

    axes[1].imshow(img2)
    axes[1].scatter(outliers[2] , outliers[3] , s=10 , c="white" , marker='o' , label="Outliers")
    
    for i in range(len(models)):
        axes[1].scatter(models[i][2] , models[i][3] , s=10 , marker='o' , label=f"Model {i+1}")

    axes[1].set_title("Image 2")
    axes[1].legend(loc='upper left')
    axes[1].axis('off')
    

    plt.show()

    