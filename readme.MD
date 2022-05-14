#  Deep Knowledge Tracing
##  Deep Knowledge Tracing Baseline Code

### Installation

```
pip install -r requirements.txt
```

### How to run

1. training
   ```
   python train.py
   ```
2. Inference
   ```
   python inference.py
   ```
   
# recsys13-git-test

## 협업 규칙✔️

#### 👉 Experiment
- 개인 실험에서 성능 향상이 일어난 경우에만 Prodiction 코드로 작업됩니다.
  - 성능 향상은 Cross Vallidation과 Public 점수 간 상관관계로 판단합니다. 
- Merge(Dev -> Main): 매주 금요일 피어세션 시간에 의견을 종합하여 병합합니다. 
- Merge(Feature -> Dev): PR을 통해 Issue가 해결될 시 병합합니다. 

#### 👉 Issue
- Production 코드로 작업될 실험은 미리 Issue를 생성하여 모든 참여자에게 알립니다.
- 작업은 가설에 대한 실험 결과를 바탕으로 진행됩니다.
- Issue 제목은 해당 실험에 대한 키워드를 포함시킵니다. 
- Issue에는 작업 내용을 상세히 적어주세요.
  - 가설, 실험 내용, 결과(CV, 리더보드 등)

#### 👉 Pull Request
- 각 실험은 하나의 Issue로 생성되며, Issue를 해결하기 위해 실험별 기능(feature)들을 추가하는 방식으로 진행이 됩니다. 
- 작업은 기본적으로 별도의 브랜치를 생성하여 작업합니다.
  - 브랜치 이름은 "f#N"으로 통일합니다 (ex. f#11)
  - 실험 폴더명은 Issue에 작성한 키워드로 생성합니다. 
- 실험의 개별 feature가 완성되면, Pull Request로 합니다.
  - PR시, Issue를 멘션합니다.
  - 모든 팀원들의 리뷰(Review)가 있었다면, Merge합니다. 
    - 머지 방식은 Squash & Merge를 따릅니다.


#### 👉 커밋 메시지 
  - 커밋 메세지는 제목(첫 줄), 본문(제목으로부터 한 줄 띄어서 작성)으로 구성됩니다.
  - 커밋 메시지의 첫 문자는 소문자로 통일하며, `:`을 표시합니다. 이때, `:` 뒤에만 띄어쓰기를 붙여줍니다. (ex. feat: lightGCN)
  - 커밋 컨벤션은 다음 표를 따릅니다.   
     <img width="442" alt="image" src="https://user-images.githubusercontent.com/83912849/164678287-5454603e-c8bf-4aaf-9e38-d65822dbf034.png">
 
## Contributors😎
|김원섭(T3044)|김진수(T3058)|민태원(T3080)|이상목(T3146)|조민재(T3204)|
|:--:|:--:|:--:|:--:|:--:|
|[![](https://avatars.githubusercontent.com/u/83912849?v=4)](https://github.com/whattSUPkim)|[![](https://avatars.githubusercontent.com/u/70852156?v=4)](https://github.com/KimJinSuPKNU)|[![](https://avatars.githubusercontent.com/u/62104797?v=4)](https://github.com/mintaewon)|[![](https://avatars.githubusercontent.com/u/62589993?v=4)](https://github.com/SNMHZ)|[![](https://avatars.githubusercontent.com/u/77037041?v=4)](https://github.com/binyf)|
|[Github](https://github.com/whattSUPkim)|[Github](https://github.com/KimJinSuPKNU)|[Github](https://github.com/mintaewon)|[Github](https://github.com/SNMHZ)|[Github](https://github.com/binyf)|

