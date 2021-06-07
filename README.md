# Classification_of_COVID-19_CT_images [20210417 ~ 20210524]

## Poster Presentation : 허지혜, 이수빈, 양원혁, 임동훈

### Data
Data1 : covid-19 CT image 184, non covid-19 CT image 5000 unbalanced data <br>
![image](https://user-images.githubusercontent.com/64202709/121008560-15ed5d80-c7ce-11eb-93fa-e8f61c47879c.png)

Data2 : covid-19 CT image 1928, non covid-19 CT image 1072 balanced data <br>
![image](https://user-images.githubusercontent.com/64202709/121008672-34ebef80-c7ce-11eb-95a5-e8d6ea972a9a.png)

### Model Architecture
1st : VGG16 <br>
![image](https://user-images.githubusercontent.com/64202709/121008174-a8d9c800-c7cd-11eb-826b-559c909ac030.png)<br>

2nd : ResNet18 <br>
![image](https://user-images.githubusercontent.com/64202709/121008194-b000d600-c7cd-11eb-97e6-775734e805fe.png)

### Evaluate
ROC curve and AUC

### Results
Data1
![image](https://user-images.githubusercontent.com/64202709/121009092-b5aaeb80-c7ce-11eb-89ce-e0653cd7b304.png)
Data2
![image](https://user-images.githubusercontent.com/64202709/121009125-c2c7da80-c7ce-11eb-8f14-5bacbbe9d6c1.png)


- This paper compared the image classification performance of the transfer learning model ResNet18 and vgg16 with two real covid-19 CT image data.
- The two transfer learning models were considered separately. The first is when the weights of the existing model are used as initial values, and the second is when the weights of the existing model are used as they are.
- Overall, it indicates that resnet18 model outperforms vgg16 model in terms of roc curve and auc.

### Reference 
```
[1] https://github.com/shervinmin/DeeepCovid  
[2] https://data.mendeley.com/datasets/3y55vgckg6/1
```
