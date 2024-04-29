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
Welcome to our exploration into the world of cybersecurity through machine learning! The aim of this project is to use machine learning techniques to identify malicious URLs — from spam to malware. We're not just about predicting; we're about making the digital world safer.

## Dataset
We worked with a dataset of 138,047 URLs, each tagged as either safe or dangerous. We trimmed down to 54 essential features after some neat preprocessing. It's all about focusing on what truly matters!

## How We Did It
Our journey began in the familiar confines of Python and Jupyter Notebooks. We followed a structured approach inspired by this awesome YouTube tutorial, but didn’t stop there! We cranked up our models’ prowess using Optuna for hyperparameter optimization, making our TensorFlow model smarter and swifter.

## Models in the Spotlight
Random Forest Classifier: A robust performer with an impressive 98.38% accuracy on tests. It’s like having the wisdom of the forest on our side!
Logistic Regression: Quick and insightful, perfect for understanding the basics with a solid 92.85% accuracy.
Deep Neural Network (TensorFlow): This smart cookie got even smarter, boasting over 96% accuracy post-Optuna tuning!

## What We Learned
Every model told a story. This project was a fantastic playground for diving deep into the nuances of machine learning, from crafting features to fine-tuning networks. We learned, we applied, and we conquered!

## Making a Difference
Imagine browsing safer, emailing without fear of phishing, and fortifying networks with our findings. That’s the future this project contributes towards—enhancing cybersecurity measures everywhere!

## Conclusion

We set out to decode the murky waters of malicious URLs and emerged with potent insights and robust models ready to make a real-world impact. It’s been a rewarding adventure in machine learning and cybersecurity!

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

