# Recommendation-System
* Developing a recommendation system using ```Deep Neural Network``` and ```Convolutional Neural Network``` for an online retail company.
## Reference
* https://github.com/ml4a/ml4a-guides/blob/master/notebooks/image-search.ipynb
## Requirements
* Python 3.5 (or later)
* scikit-learn
* scipy
* keras
* h5py
* matplotlib
## Methods for Training and Prediction
* `SVD`
* `Matrix Factorization`
* `Deep Neural Network`
* `vgg16`
* `Nearest Neighbors`
* `PCA`
## Performance Metrics
* Recall : number of recommended products purchased / number of entire purchase
* AUROC : measuring whether the model classified items that were purchased and not purchased accurately
## How to use
### MF_RFM_Basic
* `Matrix Factorization`
* Basic protoype model using RFM (Recency, Frequency, Monetary) and Matrix Factorization
* The following model provides a data frame that contains recommended items for individual users as a result
#### Overview
* Made a for loop that splits train and test data in a monthly order
```python
i = 0 
s = 7
top_n = 10
user_factor = 10
item_factor = 10

while i<=4 :
    raw02 = raw[((raw.month==(s+i))|(raw.month==(s+1+i)))]
    raw02 = raw02.drop_duplicates()
    train = raw02[raw02.month==(s+i)].drop("month",axis=1)
    test = raw02[raw02.month==(s+1+i)].drop("month",axis=1)
```
* Extracted data that has the same or above 10 order counts
* Made a rating matrix using SVD (Singual Vector Decomposition)
* Optimal parameter
  * svds(R_demeaned, k=18)
  * ord_count = 10
* Performance
  * Recall : .26
* Example output


![image](https://user-images.githubusercontent.com/42960718/53979279-b4526a80-4150-11e9-9ed3-01a2d2704744.png)

### MF_RFM_DNN
* `Deep Neural Network`
* Applied RFM (Recency, Frequency, Monetary) metrics as the weight for the model to reflect merchantability
* The following model provides a data frame that contains recommended items for individual users as a result
* Model summary




![image](https://user-images.githubusercontent.com/42960718/53905608-50666e00-408c-11e9-90bb-ec7cbd1166e0.png)

#### Overview
* Made a for loop that splits train and test data in a monthly order
* Added densed for each user based RFM and item based RFM
* Concatenated the dense by using embedding and flatten
* Optimal parameter (in progress)
  * top_n = 10 
  * user_factor = 27
  * item_factor = 27
  * factors = 20
  * batch_size = 256
  * learning_rate = .0001
  * epoch = 30
  * rfm_dense1 = 32
  * rfm_dense2 = 60
  * dot_dense1 = 16
  * dot_dense2 = 8
  * optimizer = Adam(lr = learning_rate)
  * loss = binary_crossentropy 
* Performance (not stablized yet)
  * Recall = .31
  * AUC Score = .659
  
  
  ![image](https://user-images.githubusercontent.com/42960718/53905266-77707000-408b-11e9-8ff5-82809e747f8d.png)
  
  
  ![image](https://user-images.githubusercontent.com/42960718/53905287-83f4c880-408b-11e9-8075-9ac1921b1819.png)
  

  ![image](https://user-images.githubusercontent.com/42960718/53942026-11bccc00-40fd-11e9-8c06-0a39eb8327fc.png)


  ![image](https://user-images.githubusercontent.com/42960718/53942066-31ec8b00-40fd-11e9-9577-9c4672ef76a5.png)


* Example output


![image](https://user-images.githubusercontent.com/42960718/53905459-fcf42000-408b-11e9-9298-6dff01e0452c.png)

### vgg16_RFM_CNN
* `Convolutional Neural Network`
* Applied RFM (Recency, Frequency, Monetary) metrics as the weight for the model to reflect merchantability
* Used vgg16 as base model
* Model summary




![image](https://user-images.githubusercontent.com/42960718/53956482-11cdc380-411f-11e9-9bd2-6461a6a8fdf1.png)
#### Overview
* Input image size = 224, 224, 3
* Applied item based RFM and item categories (categories are classified by numbers) to vgg 16 model
* Pre-process images for better model performance
```python
def preprocess_input(x):
    x /= 255
    x -= 0.5
    x *= 2
    return x
```
* Find similar items using euclidean nearest neighbors
```python
nn_num = 6
X = list(aa["features"])
nbrs = NearestNeighbors(n_neighbors=nn_num, algorithm= "ball_tree", 
                        metric="euclidean", n_jobs = -1).fit(X)
```
* User defined function that presents top 5 recommendations and use pca
```python
def show5recommendations(name, table, NearestN,  idnr, directory, columnfeature):
    key = table[(table.item == idnr)].iloc[0][columnfeature]
    distances, indices = NearestN.kneighbors(key)
    listindices = pd.DataFrame(indices).values.tolist()
    listindices2 = listindices[0]
    ids = udfsimular(listindices2, table)
    paths2 = udfidfpathh(ids,directory)
    fig, ((ax1, ax2, ax3, ax4, ax5, ax6)) = plt.subplots(nrows=1, ncols=6, sharex=True, sharey=True, figsize=(14,3))

    ax1.imshow(mpimg.imread(paths2[0]))
    ax1.set_title(r"$\bf{" + str(name) + "}$"+"\n Targer:\n"+ ids[0])
    ax1.set_yticklabels([])
    ax2.imshow(mpimg.imread(paths2[1]))
    ax2.set_title("Rec 1:\n"+ ids[1])
    ax3.imshow(mpimg.imread(paths2[2]))
    ax3.set_title("Rec 2:\n"+ ids[2])
    ax4.imshow(mpimg.imread(paths2[3]))
    ax4.set_title("Rec 3:\n"+ ids[3])
    ax5.imshow(mpimg.imread(paths2[4]))
    ax5.set_title("Rec 4:\n"+ ids[4])
    ax6.imshow(mpimg.imread(paths2[5]))
    ax6.set_title("Rec 5:\n"+ ids[5])
 ```
```python
show5recommendations(df_img["item"][7949] , df_img, nbrs, df_img["item"][7949], "../../linkshops.full.5.img/", "features")
```
```python
pca = PCA(n_components=1000)
```
 * Get images that have the closest cosine similarity based on pca features
 ```python
similar_idx = [distance.cosine(pca_features[query_image_idx], feat) for feat in pca_features]
idx_closest = sorted(range(len(similar_idx)), key=lambda k: similar_idx[k])[1:11]
# load all the similarity results as thumbnails of height 100
thumbs = []
for idx in idx_closest:
    img = image.load_img(images[idx])
    img = img.resize((int(img.width * 100 / img.height), 100))
    thumbs.append(img)

# concatenate the images into a single image
concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)

# show the image
plt.figure(figsize = (16,12))
plt.imshow(concat_image)

def get_closest_images(query_image_idx, num_results=10) :
    distances = [ distance.cosine(pca_features[query_image_idx], feat) for feat in pca_features ]
    idx_closest = sorted(range(len(distances)), key=lambda k : distances[k])[1 :num_results+1]
    return idx_closest

def get_concatenated_images(indexes, thumb_height) :
    thumbs = []
    for idx in indexes :
        img = image.load_img(images[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image
```
* Example Output


![image](https://user-images.githubusercontent.com/42960718/53979141-5160d380-4150-11e9-8dca-4b0f84ecc6c5.png)
![image](https://user-images.githubusercontent.com/42960718/53979164-60478600-4150-11e9-8841-9275fb6ce9f7.png)


## Note
* Following repository will only contain sample codes due to contract issues with the following company
* Full code for recommendation system nor PowerPoint presentation file will not be uploaded until the project ends
