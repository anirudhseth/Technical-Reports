# Technical-Reports
A collection of technical reports written at KTH. Summary of each document.
### Accoustic Word Embeddings.pdf 
**Abstract** : Acoustic Word Embedding (AWE) is a fixed-dimensional representation of a variable-length audio signal in an embedding space. The motivation behind the
approach is to capture the rich information from speech and to eliminate the errors
caused by the process of transcribing. AWE have started gaining popularity because
they are easy to train, scale well to large amounts of training data, and do not
require a lexicon. In this project, we investigate recent publications, literature and
implement a LSTM Encoder-Decoder framework that borrows the methodology
of skipgram and continuous bag-of-words to learn contextual word embeddings
directly from speech data. The word embeddings are then evaluated on widely
used word similarity benchmarks and compared with the embeddings learned by
Word2vec and fastText. Experimental results from our model show that it is in
fact possible that the embeddings learned from speech outperform those produced
by its textual counterparts.

### Deep Learning - COVID19 detection.pdf
**Abstract** :The global pandemic is affecting human lives in various ways and introducing
social and economic problems. To fight against the spread of the virus, it is
significant to have a practical yet precise diagnostic tool. Chest X-ray images are
widely known as a low cost, highly available solution, however they require the
assessment of a medical professional and may introduce personal or professional
biases. In this project, we investigate and develop a multi class deep convolutional
network to assist in the classification of chest X-ray images. To account for the lack
of publicly available COVID19 data we explore and perform comparative analysis
of various data augmentation techniques and their applicability on X-ray images.
Finally we apply Grad-CAM to highlight the regions of high importance and
improve the explainability of our deep neural network. We find that standard data
augmentation techniques as well as advanced methods, such as image segmentation,
do not improve the results. Whether this is a result of removing biases or a lacking
implementation is uncertain.
### Deep Network Explanation Methods.pdf
**Abstract** : Deep Neural Networks have achieved state of the art performance in several domains like natural language
processing [15], computer vision [8] , weather forecasting [4] etc. The high accuracy is often achieved
by models which are quite complex and have millions of trainable parameters.This often comes at the
cost of interpretability of the model.With the increased use of such networks in critical domains like law
enforcement [2],medical diagnosis [12] ,autonomous vehicles [9] etc, there is a inherent need to understand
how the various aspects of training data drive the decisions of the NN.The explainability of a model can
instill trust in its predictions and also provide useful insights for further improvement.
This essay summaries three different approaches to understand the predictions made by a neural
network. Koh et al.[5] rely on influence functions , a classic technique from robust statistics , Lundberg
et al.[6] propose a unified framework called SHAP based on Shapley values, a concept from cooperative
game theory and finally Selvaraju et al [11] introduce Grad-CAM that relies on gradients to generate a
heat map indicating regions of importance of an image.

### Reproduction - GPLVM.pdf
**Abstract** :This project builds on two research papers published in 2004 and 2005 by Neil D. Lawrence that present
a novel dimensionality reduction (DR) technique called Gaussian Process Latent Variable Model (GPLVM)
and compare it to other DR techniques known at the time. Here we reproduce the original results, confirming
the claims made by Lawrence by using our own implementation of the methods being used and compare the
results to more recent approaches to dimensionality reduction. Furthermore, we critically discuss how our
results compare to those from the original papers and suggest improvements to the experimental setup.
### Bitonic Merge Sort - MPI.pdf
**Abstract** : Sorting is often used as a basic building block when designing algorithms. It is an important component
for many applications like searching, closest pair, element uniqueness, frequency distribution, selection,
convex hulls etc. Batchers Bitonic sort is one such algorithm which is represented by a sorting network
consisting of multiple butter
y stages.
In this project we study and implement a sequential and a parallel version of the Bitonic sorting algorithm,
evaluate its performance using multiple metrics and compare it to some other state of the art sorting
algorithms.

### Uncertainty Estimation with Deep Networks.pdf
**Abstract** : Neural Networks have proven to be immensely powerful and achieved state-of-the-art performance on
wide range of domains like natural language processing [21], computer vision [15] , weather forecasting [2]
and medical diagnosis [19]. Despite achieving impressive scores on benchmark tasks ,NN's are prone to
produce overconfident predictions [8],[6]. These overconfident predictions are a result of poor assessment
and quantification of predictive uncertainty especially when dealing with data scarcity.
The uncertainty in predictions can arise due to the lack of knowledge (or understanding) of the model
a.k.a distributional uncertainty [13] or epistemic uncertainty [3]. Aleatoric uncertainty on the other hand
is irreducible and is representative of the unknowns that differ on each run of the experiment. Majority
of the real world problems also suffer from the problem of dataset shift [17] ,a mismatch between the
distributions of your training and test dataset.With the increased application of NN's in safety critical
tasks like perception systems for autonomous vehicles [16], medical diagnosis [19] etc. estimating and
evaluating the quality of predictive uncertainty of the model has become a crucial task.
This essay summaries the research on Bayesian approaches - Bayes by Backprop [1] as well as Non-
Bayesian approaches like Ensembles with random-initialization [11]. and Ensemble Distribution Distilla-
tion (EnD2) [14] to tackle the aforementioned task.
### Visual Explanations for Deep Convolutional Networks.pdf
**Abstract** : We replicate selected experiments from [1] which evaluates the localization ability
of Grad-CAM and compare it to other methods. Additionally we show its applicability
to detecting adversarial noise, Image Captioning and mode collapse detection
during the training of a GAN. Some discrepancies compared to the original paper
are reported. The localization errors are higher, but the overall trend is similar when
comparing to other methods. The code for our project can be found on GitHub 1.
### Generative Models.pdf
**Abstract** : Generative modeling has emerged as a major branch of unsupervised machine learning. Such models are
immensely powerful in learning useful data distributions which also allows them to generate synthetic
data. Recent advances in neural networks architectures combined with the progress in computational
power and optimization techniques have enabled such networks to model high-dimension data including
text [18] and speech [9] and videos and images [12],[24],[14] possible.In this essay , I review three different
neural network architectures of generative modeling and there application to images-:
1. Variational autoencoders (VAEs) [15] - NVAE [24]
2. Flow-based generative models [6],[5] - GLOW [14]
3. Generative adversarial networks [8] - StyleGANv2 [12]
The applications of above models range from text to image translation [17] reconstruction of 3d models
using images [19],drug discovery [1], music generation [7] etc.
### Rescuing Matt Damon - Artificial Intelligence.pdf
**Abstract** :In this project we create a simulated environment within which we test three planning algorithms: one
using a genetic algorithm (GA), one using ant colony optimization (ACO), and one using a greedy approach.
We attempt to measure the sensitivity of these methods to uncertainty as well as outline when each algorithm
may outperform the others.
Our simulated environment and the task we evaluate these planning algorithms on are both inspired by
the 2015 film, The Martian. The environment is a grid world representing the Martian surface and containing
objects which represent Matt Damon and valuable scientific equipment. The task is to plan a path for a rover
to traverse the grid world to collect Matt Damon and as many pieces of scientific equipment as possible before
returning to the starting location. The challenge of this task is the high degree of uncertainty present in the
problem as well as the constraint of limited battery usage. Instead of providing the rover with knowledge
of where Damon and the equipment are in the world, only a PDF showing the likelihood of finding each
object of interest is provided. Additionally, instead of providing the rover with knowledge of the terrain in
advance, the rover is provided with a noisy approximation of the terrain.
Our analysis ends with a discussion of the ways to improve our analysis and possible directions for future
work.