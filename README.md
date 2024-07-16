# Content
- Closest Resemblance classifier: CR classifier
- several classifiers from scikit-learn package: k Nearest Neighbors (K-NN), Support Vector Machine (SVM), Random Forest (RF), Neural Network (NN), Naive Bayesian (NB) and Nearest Centroid (NC)...
- Statistical analysis for  CR in comparing with other classifiers + details results of 5 cross validation (for other k -cross validation you need to change the parameter in the script).
- Different metrics and confusion matrix with graphs
- 
# CR Classifier
The Closest Resemblance Classifier: CR-General
* Abstract:
* Classifiers face a myriad of challenges in today's data-driven world, ranging from overfitting and high computational costs to low accuracy, imbalanced training datasets, and the notorious black box effect. Furthermore, many traditional classifiers struggle with the robust handling of noisy and missing feature values. In response to these hurdles, we present  classification methods that leverage the power of feature partitioning learning and outranking measures.
Our classification algorithms offer an innovative approach, eliminating the need for prior domain knowledge by automatically discerning feature intervals directly from the data. These intervals capture essential patterns and characteristics within the dataset, empowering our classifiers with newfound adaptability and insight.
In addition, we employ outranking measures to mitigate the influence of noise and uncertainty in the data. Through pairwise comparisons of alternatives on each feature, we enhance the robustness and reliability of our classification outcomes.
The developed classifiers are empirically evaluated on several data sets from UCI repository and are compared with well-known classifiers including k Nearest Neighbors (K-NN), Support Vector Machine (SVM), Random Forest (RF), Neural Network (NN), Naive Bayesian (NB) and Nearest Centroid (NC). The experiments result demonstrate that the classifiers based on feature interval learning and outranking approaches are robust to imbalanced data and to irrelevant features and achieve comparably and even better performances than the well-known classifiers in some cases. Moreover, our proposed classifiers produce more explainable models whilst preserving high predictive performance levels.

# Usage for K-CR.py

* K-CR.py [-h]  [-k CV_K] [-t T] [-cls CLASSIFIERS] [-file FILE_NAME]

* Example usage: python K-CR.py -k 5 -t 1 -cls 0 -file Geobia.csv

* Optional arguments:
  * -h, --help            show this help message and exit.

  * -k CV_K              Cross validation parameter. CV_K = 0 indicates leave
                         one out, while any other positive integer
                         bigger than 1 indicates the number of folds for k fold
                         validation.

  * -t T                Used when generating prototypes with
                        single interval algorithm using mean and standard
                        deviation.

 * -cls CLASSIFIERS      An integer as indicator. 0 means executing all the
                        other baseline classifiers on the current dataset. 1
                        only CR classifier.

 * -file FILE_NAME       A string indicates the name of the file. It takes both formats txt and csv. It could be
                        the single file including both training and testing
                        data, or the training file when a test file name is
                        provided. In either case, the first line of the file should be dataset information (the header names of each column or other
                        information with your description) and continue with the data. In addition, if the dataset contains both data and label, the label 
                        should be placed at the last column.
   * For example:
   * V1  V2  V3  V4
   * 0.2 0.3 0.4 1
   * . . . . . . .
   * 0.4 0.5 0.6 0
   * In txt file, the data should be separated by '\t', and in csv file, the data should be separated by ','.


# Statistical testing CR1 or CR2 and for testing several data sets using 5-fold cross validation.

* usage: StatisticalAnalysis.py [-h] [-t T] [-files FILE_NAMES]
          - files FILE_NAMES space seperated file names to perform statistical testing.

* Example:

StatisticalAnalysis.py -t 1.1 -files Dermatology.csv HeartDisease.csv thyroid.csv pageblocks.csv BreastCancer.csv shuttle.csv newthyroid.csv Geobia.csv Glass.csv leaf.csv Sonar.csv SPECTF.csv

# Environment:
This program is developed and tested under following dependencies and packages:
* Python		3.7
* Pandas 1.3.5
* pip 23.0.1
* scikit-learn	1.0.2
numpy		1.19.5

# Citation:
We would appreciate using the following citations:

```
@article{CR2024,
	  doi = {10.20944/preprints202407.1257.v1},
	  url = {https://doi.org/10.20944/preprints202407.1257.v1},
	  year = 2024,
	  month = {July},
	  publisher = {Preprints},
	  author = {Nabil Belacel},
	  title = {A Closest Resemblance Classifier with Feature Interval Learning and Outranking Measures for Improved Performance},
	  journal = {Preprints}
}

```
```
@INCOLLECTION{Belacel04KCR,
  AUTHOR ={Belacel, N.},
  TITLE ={The k-closest resemblance approach for multiple criteria classification problems},
  BOOKTITLE ={{Modelling, Computation and Optimization Information and Management Sciences}},
  isbn ={1-903398-21-5},
  PUBLISHER ={Hermes Sciences Publishing},
  YEAR ={2004},
  editor ={Hoai, {L.-T.} and Tao, P.},
  volume = {},
  series ={},
  type ={},
  chapter ={},
  pages ={525-534},
  address ={},
  edition ={},
  keywords ={},
  note ={} 
  }

```

```
@inproceedings{belacelREMOTESensingData2020k,
  title={The K-Closest Resemblance Classifier for Remote Sensing Data},
  author={Belacel, Nabil and Duan, Cheng and Inkpen, Diana},
  booktitle={Canadian Conference on Artificial Intelligence},
  pages={49--54},
  year={2020}
}

```

```
@PHDTHESIS{Belacel_PHD_1999,
    AUTHOR = "Belacel, N.",
    TITLE = "Méthodes de classification multicritère: Méthodologie et application à l’aide au diagnostic médical.",
    SCHOOL = "Univ. Libre de Bruxelles",
    number = "148",
    ADDRESS ="Belgium, Brussels",
    YEAR ="1999"
    }

```

# Acknowledgements

Thanks to our Co-op students Rani Adhaduk and Durga Prasad Rangavajjala for their invaluable assistance in python implementation.
