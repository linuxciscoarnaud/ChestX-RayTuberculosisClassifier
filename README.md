# Tuberculosis diagnosis implemented with CNN and trained on frontal Chest X-Rays

This is an implementation of a CNN-based tuberculosis diagnosis described in the paper: [Efficient Deep Network Architectures for Fast 
Chest X-Ray Tuberculosis Screening and Visualization](https://www.nature.com/articles/s41598-019-42557-4). It makes use of the 
deeplearning4j library. I trained the network using the Early Stopping technique to make sure i end up with the best model and avoid overfitting. The network is successively trained on two datasets: the Montgomery County chest X-ray set (MC) and the 
Shenzhen chest X-ray set, which can both be found [here](https://ceb.nlm.nih.gov/repositories/tuberculosis-chest-x-ray-image-data-sets/).

## Get it to work
- Clone the repository.
- Download the training data at the following link: (to be specify later). These data have already been pre-processed as described in the previously cited paper. They have already been orginized so as to be ready for the 6-fold cross-validation study. Unzip the downloaded file and place the resulting directory within the following location: "application directory"/src/main/resources/.

## Results
I conducted a 6-fold cross-validation study on both the two datasets. The network was trained on my old laptop which has the following characteristics:
- Processor: Intel(R) Core(TM) i3-2310M CPU @ 2.10GHz 2.10GHz
- Installed memory (RAM): 4.00 GB (3.85 GB usable)
- System type: 64-bit Operating System, x64-based processor

The training time can be significantly reduced if application is runned on more advanced CPU/GPU architectures.

### Montgomery County chest X-ray set (MC)

I report here the validation accuracy, AUC value and AUC for the 6-fold cross-validation study on the Montgomery County chest X-ray set.

![Validation_Results](https://user-images.githubusercontent.com/1300982/63007064-dd9dc200-be77-11e9-9aba-4005982dd2f0.png)

![Validation_AUC](https://user-images.githubusercontent.com/1300982/63007040-d1b20000-be77-11e9-8d63-fbed8f644157.png)

### Shenzhen chest X-ray set
(Comming soon)

