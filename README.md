# 2021-2-OSSP1-ChiliSause-5

## 주제
2021-2 공개SW프로젝트_01 칠리소스 프로젝트팀 - GAN을 이용한 자연스러운 얼굴 비식별화 구현

## 팀원
### ChiliSause팀

- 유성민
- 김병현
- 이서영
- 조혜근


---------------------

# 프로그램 파이프라인
![image](https://user-images.githubusercontent.com/48210134/146313223-9137af67-869f-49e0-aea1-cf124293c13e.png)

## 실행 환경

Ubuntu 20.04.3

pytorch 1.10.0

## 실행 방법

### 1. 실행환경 세팅
```bash
git clone https://github.com/CSID-DGU/2021-2-OSSP1-ChiliSause-5.git
cd 2021-2-OSSP1-ChiliSause-5
conda env create --file environment.yaml
conda activate psp_env
cd DDFA_V2
./build.sh
cd ..
```

### 2. weight 파일 다운로드


### 3. Input data 준비

사람이 들어간 영상을 sample.mp4로 저장한다.

### 4. 프로그램 실행
```bash
python main.py
```

## 데모
![original](https://user-images.githubusercontent.com/48210134/146315345-32b60d54-960a-468c-8e92-0e2c18730a0f.gif)
![after](https://user-images.githubusercontent.com/48210134/146315355-cca24e77-d4f3-4163-b159-2ad5b254f257.gif)
