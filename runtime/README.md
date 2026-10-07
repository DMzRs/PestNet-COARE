## PestNet: Hierarchical Scaled Dot-Product Attention for Insect Pest Detection

#### Overall architecture of the proposed PestNet model for insect pest detection
<img width="968" height="505" alt="image" src="https://github.com/user-attachments/assets/7a5517c3-4f62-479e-abb5-415f0ea958ee" />


#### Dataset
| Dataset | Download | Class # | Image # |
|---------|----------|---------|---------|
| IP102   | [Drive](https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo?usp=sharing)   | 102 | 75,222 |
| R2000   | [GitHub](https://github.com/awsomespark/R2000)        | 16  | 2,046  |

#### Training
Hyperparameters:
* Epochs: 200
* Batch size: 16
* Pre-train: None
  
| Dataset | mAP@50 | mAP@50:95 |
|---------|--------|----------|
| IP102   | 69.7   | 44.9     |
| R2000   | 82.6   | 68.2     |

#### IP102 Training Plot
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/ca4054d1-165a-4443-aae0-b63ca94008f2" />

<img width="1811" height="1248" alt="image" src="https://github.com/user-attachments/assets/4e3f0d8b-4fb5-4fa0-b85c-054e5831bcb7" />



#### R2000 Training Plot

<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/50455831-9020-4f70-9d47-60b60ff36883" />

<img width="879" height="1068" alt="image" src="https://github.com/user-attachments/assets/11f7f782-dab2-4f81-a7a2-7628d883d4ce" />

#### Pest24 validation

<img width="926" height="439" alt="image" src="https://github.com/user-attachments/assets/e52244d2-c60d-480c-8c4c-2ec701b9a5a7" />



#### Visualization
<img width="1648" height="758" alt="image" src="https://github.com/user-attachments/assets/cab46bd6-b519-45ec-9376-a9f112ec830d" />

