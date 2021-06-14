# image2vec
Building applications on top of Image Embeddings. Recommendation Engine, Image similarity.

This project converts product images into embeddings using the Inception V3 model pretrained with ImageNet.
Any type of images can be used, not only from products.

Once we have good embeddings, then we have accurate similarities between images. 
That's why created 2 demos: A recommendation engine, and a Image similarity search.
Both demos are created using streamlit.

#### Requirements
The requirement packages are: numpy, scikit-learn, matplotlib, tensorflow, pillow, and streamlit.

    $ pip install numpy
    $ pip install scikit-learn
    $ pip install matplotlib
    $ pip install tensorflow
    $ pip install pillow
    $ pip install streamlit

---
### Use

#### Change 'config.py'
Download any dataset of images and change the 'DATABASE_PATH' variable from 'config.py' to your dataset path.
Also, change the 'N_IMG' variable to the number of images you want to embed from your dataset.
And set 'DO_KPCA' to True if you want to perform Kernel PCA dimensionality reduction on the last hidden layer from Inception V3 model.

#### Run '0_image2vec.py'
Run this file

#### Run '1_image_similarity.py'
```bash
$ streamlit run 1_image_similarity.py
```

#### Run '2_user_recommendation.py'
```bash
$ streamlit run 2_user_recommendation.py
```
