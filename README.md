# Network Security Systems Final Project
## Malicious URL Detection
## EECE 461 Network Security Systems
### Manhattan College, Electrical & Computer Engineering

**Faculty Mentor:** Kevin Muller
## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [How We Did It](#how-we-did-it)
4. [Models in the Spotlight](#models-in-the-spotlight)
5. [What We Learned](#what-we-learned)
6. [Making a Difference](#making-a-difference)
7. [Conclusion](#conclusion)
8. [References](#references)

## Project Overview
The purpose of this project is to investigate malicious URLs, attempting to differentiate them using machine learning techniques. The goal is to train a model that is able to consistently differentiate between various types of URLs, including spam, phishing, malware, and defacement. I have also set the additional goal to seek to understand how this insight can be used to overall improve various network security systems.

## Dataset
Like most machine learning projects, the dataset you use is the most integral part of creating the model. The dataset that was used consists of 138,047 URLs, with each URL labeled as legitimate or malicious based on its characteristics. The dataset was initially preprocessed by removing specific features like 'Name' and 'md5', which are not relevant to the classification task, resulting in 54 features used for model training and testing.


## How We Did It

The development of the machine learning models for detecting malicious URLs was performed in Python using Jupyter Notebook, a platform that simplifies the running of code for machine learning based projects. The initial step in the process involved following a structured approach, outlined in a YouTube tutorial on creating a malware detection machine learning model (https://www.youtube.com/watch?v=fn33V4iX5G4). This tutorial provided a foundational framework, which included steps for data preparation, feature selection, and basic model training.
Building on this foundation, I integrated optimizations using Optuna, a hyperparameter optimization framework. Optuna automates the process of finding the most effective parameters for machine learning models. It significantly enhances model performance by testing a range of values for each parameter and selecting those that produce the best results based on predefined evaluation metrics. This approach was particularly beneficial for fine-tuning the deep learning model built with TensorFlow, allowing the identification of optimal network configurations and learning rates.

The next step was to create a script to be able to test the various models on custom URLs that were not a part of the test script or training data. This proved to be rather difficult as since the model was trained on features of the URL and not the raw URLs themselves, the implementation would have to implement the feature extraction for every inputted URL. This was not practical to implement, but my own research shows that others have done this rather difficult step and have had high accuracies in the detections. I created a theoretical test script, with the only missing aspect being the feature extraction for the given URL.

At the conclusion of the project, all software developed, including the complete source code, detailed documentation, and references used, is made publicly available on my GitHub repository at Joel-Mendez0. This repository serves as a resource for anyone interested in the methods, models, and optimizations employed in this project.

## Models in the Spotlight
Random Forest Classifier: A robust performer with an impressive 98.38% accuracy on tests. It’s like having the wisdom of the forest on our side!
Logistic Regression: Quick and insightful, perfect for understanding the basics with a solid 92.85% accuracy.
Deep Neural Network (TensorFlow): This smart cookie got even smarter, boasting over 96% accuracy post-Optuna tuning!

## What We Learned
The primary learning objectives of this project revolved around gaining a deep understanding of machine learning techniques for detecting malicious URLs and learning the feature engineering required for such tasks. Additionally, comparing various machine learning algorithms highlighted on how each processes data differently and their suitability for different patterns. For instance, while logistic regression struggled to balance recall and precision optimally, random forests and deep learning models were superior in handling complex data structures. Moreover, employing Optuna for hyperparameter tuning in the TensorFlow model highlighted the influence of hyperparameters, providing hands-on experience in fine-tuning neural networks for peak performance.

## Making a Difference
The ability to accurately detect malicious URLs is paramount for enhancing cybersecurity measures. Implementations of this project can be integrated into various real-time systems for internet security, offering benefits in several domains. For instance, incorporating this model into web browsers could provide real-time alerts to users about potentially malicious URLs, helping prevent access to harmful sites. Similarly, the model could be employed in email filters to enhance email security by screening and filtering out messages that contain malicious links, thus protecting users from phishing and other email-based threats. Additionally, deploying this technology in network security systems, such as firewalls, would allow for the proactive blocking of traffic from known malicious sources, mitigating potential attacks before they can reach the user and significantly enhancing the overall security infrastructure of networks.


## Conclusion

This project not only achieved the goal of understanding and applying machine learning techniques to detect malicious URLs but also provided insights into the practical applications of such models in enhancing cybersecurity measures. The experience gained in feature engineering, model selection, and tuning has broader implications for future projects in machine learning and data science

## References

https://www.youtube.com/watch?v=fn33V4iX5G4
https://github.com/AsimGull/Malware-detection-with-ML-and-deep-learning
https://github.com/PacktPublishing/Mastering-Machine-Learning-for-Penetration-Testing/tree/master/Chapter03
https://medium.com/nerd-for-tech/url-feature-engineering-and-classification-66c0512fb34d
https://towardsdatascience.com/extracting-feature-vectors-from-url-strings-for-malicious-url-detection-cbafc24737a
https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9133026/
https://www.youtube.com/watch?v=I1refTZp-pg
https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques

