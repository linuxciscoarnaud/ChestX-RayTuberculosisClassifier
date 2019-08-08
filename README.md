# Tuberculosis diagnosis implemented with CNN and trained on frontal Chest X-Rays

This is an implementation of a CNN-based tuberculosis diagnosis described in the paper: [Efficient Deep Network Architectures for Fast 
Chest X-Ray Tuberculosis Screening and Visualization](https://www.nature.com/articles/s41598-019-42557-4). It makes use of the 
deeplearning4j library. The network is successively trained on two datasets: the Montgomery County chest X-ray set (MC) and the 
Shenzhen chest X-ray set, which can both be found [here](https://ceb.nlm.nih.gov/repositories/tuberculosis-chest-x-ray-image-data-sets/).

## Results
I conducted a 6-fold cross-validation study on both the two datasets. The network was trained on my old laptop which has the following characteristics:
- Processor: Intel(R) Core(TM) i3-2310M CPU @ 2.10GHz 2.10GHz
- Installed memory (RAM): 4.00 GB (3.85 GB usable)
- System type: 64-bit Operating System, x64-based processor

### Montgomery County chest X-ray set (MC)
