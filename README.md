# Assingment-6_Llerena

According to the American Cancer Society (ACS), breast cancer mortality rates have dropped nearly 44% over the past 30 years, largely driven by faster detection times. Machine learning plays a crucial role in expediting this process. The goal of this assignment is to train and evaluate three classification models (Logistic Regression, Decision Trees, and SVC) to automatically classify tumor biopsies as benign or malignant, helping accelerate screening and diagnosis.

This project runs using **Python 3** within an **Anaconda/Conda** environment.
Main project sklearn(datasets, model trainers, and evaluation metrics)

Data from this dataset was divided into training 80% and testing 20% sets using a fixed seed random_state=250 and stratification to maintain class proportions.

Perfomance Analysis
Among the three evaluated models, Logistic Regression achieved the strongest overall performance, yielding an accuracy of 0.95, precision of 0.98, recall of 0.94, and an F1-score of 0.96. This demonstrates that the model accurately classifies patient diagnostic status in 95% of test instances while maintaining an optimal balance between true detections and minimal false alarms.
The other two architectures, the Decision Tree Classifier and the Support Vector Classifier (SVC) performed similarly to each other, both remaining slightly inferior to Logistic Regression in aggregate balance.
Crucially, SVC attained the highest recall across all three models at 0.98. In a clinical diagnostic screening workflow, this indicates that the classifier successfully detected 98% of all true positive cases in the test cohort, missing only 2%.
