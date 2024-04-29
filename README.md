EECE 461: Network Security Systems
Machine Learning for Malicious URL Detection
Prepared By: Joel Mendez
Submitted to: Kevin Muller
Date: 04/28/2024
Project Overview
The purpose of this project is to investigate malicious URLs, attempting to differentiate them using machine learning techniques. The goal is to train a model that is able to consistently differentiate between various types of URLs, including spam, phishing, malware, and defacement. Additionally, insights from this project aim to improve various network security systems.

Dataset
The dataset consists of 138,047 URLs, labeled as legitimate or malicious based on their characteristics. After preprocessing to remove irrelevant features, 54 features were retained for model training and testing.

Procedure
The development of machine learning models for detecting malicious URLs was performed in Python using Jupyter Notebook. Following a structured approach outlined in this YouTube tutorial, I built upon this foundation by integrating optimizations using Optuna to fine-tune a deep learning model built with TensorFlow.

Model Selection
Random Forest Classifier: Achieved 98.38% accuracy on the test set with an F1 score of 97.31%, demonstrating strong performance and reliability.
Logistic Regression: Achieved a testing accuracy of 92.85%, providing baseline measurements and interpretability.
Deep Neural Network (TensorFlow): After tuning with Optuna, it achieved over 96% accuracy in both training and testing phases, showcasing its capability to handle complex data structures.
Learning Goals and Outcomes
The primary objectives were to understand machine learning techniques for detecting malicious URLs and to learn the feature engineering required for such tasks. Comparing various algorithms highlighted how each processes data differently, suitable for different patterns and complexities.

Real-World Application
Implementations of this project can be integrated into real-time systems for internet security, such as web browsers for real-time alerts on malicious URLs, email filters to enhance security, and network security systems like firewalls to block traffic from known malicious sources.

Conclusion
This project not only achieved its goals of understanding and applying machine learning techniques to detect malicious URLs but also provided insights into their practical applications in enhancing cybersecurity measures.

References
YouTube Tutorial on Malware Detection
Packt Publishing - Mastering Machine Learning for Penetration Testing
URL Feature Engineering and Classification
Extracting Feature Vectors from URL Strings
My Source Code
GitHub Repository for the Project
