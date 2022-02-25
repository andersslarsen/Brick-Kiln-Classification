# Brick-Kiln-Classification
Implementation of a Convolutional Neural Network as a method for identifying Brick Kilns in Bangladesh

Due to storage limitations, the online notebook environment Kaggle was used, for running and
training the model, utilizing the CPU and GPU capacities available. However, due to spacial
and computational limitations within this environment, we were only able to use a subset of the
data-set, - 40% of its original size. We found that this was not a limitation with regards to the
computed results.
The public data-set provided for this task consists of low-resolution images from the Sentinel 2
satellite constellation from the Google Earth Engine, taken between October 2018 and May 2019.
Roughly 73 thousand images are provided in the set, with 6,329 containing a brick kiln, and 67,284
not containing any kiln. The 64x64 (10m resolution) images are stored as HDF5 files, alongside
various metadata relating to the image. Originally, the data contains information regarding the
geographical location of each image, which can be used to locate potentially identified brick kilns.
This task is is considered out of scope of this paper, and will thus not be implemented, although
it is worth mentioning that this is a possible extension for future work.

Sustainbench has published a GitHub repository [SustainBench GitHub n.d.], which provides dataloaders
and baseline models for most of the benchmark tasks, as well as processed versions of the
necessary datasets. We used the supplied functionality for several aspects of the data processing,
which was helpful in getting the data to the desired format.
