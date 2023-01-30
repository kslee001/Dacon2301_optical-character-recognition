# DACON_2301 : 2023 교원그룹 AI 챌린지 
- Optical Character Recognition


## 1. 대회 결과
- 최종 성적
    - Public  :
        - **Accuracy : 0.86296  ( 31 / 430 )**
            - 1위 : 0.9642
    - Private :
        - **Accuracy : 0.86113  ( 31 / 430 , top 7% )**
            - 1위 : 0.96274

## 2. 대회 개요
- 주최 : (주)교원
- 주관 : 데이콘
- 대회기간 : 2022.12.26 ~ 2023.01.16 (예선)
- Task
    - **optical chracter recognition**
    - 손글씨 인식 AI 모델 개발
- Data : handwritten characters
    - train data
        - 76,888개
    - test data
        - 74,120개

- 상금 : 총 1000만원 (본선)
    - 1위 : 500
    - 2위 : 300만원
    - 3위 : 200만원

## 3. 진행 과정
### 데이터 전처리
- 불균형 데이터를 증강시키기 위해 cutmix기법을 이용하여 length가 2 이상인 데이터 생성  
- translation, rotation 등을 이용하여 train-time이 아니라 dataset definition 단계에서 증강된 데이터를 생성  
    - torch.utils.data.ConcatDataset( [train_dataset, *train_aug_datasets] )  
    - 이것이 확실히 성능 개선에 도움이 되었는지는 알 수 없음  
        - overfit의 위험이 없는지 check 필요  

### 학습
- backbone  
    - image encoder  
        - TPS_SpatialTransformerNetwork  
        - Regnety_120  (main cnn extractor, pretrained)  
    - sequence decoder  
        - Transformer Encoder + Linear head (vanilla torch module)  
- loss function : CTC loss  
- optimizer : Adam optimizer  
- scheduler : ReduceLROnPlateau  


## 4. Self-feedback?
- decoder를 좀 더 좋은걸 썼으면 좋았을텐데...  
- image encoder-text decoder에 익숙하지 않아 더 좋은 모델을 만들지 못한 것 같다. 
