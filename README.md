# Meta - sapiens - Tutorial 

[sapiens Github](https://github.com/facebookresearch/sapiens?tab=readme-ov-file)


sapiens 공식 깃헙을 보면, 어떻게 시작해야할지 막막합니다.

구글에서 서칭을 해도 tutorial이나 colab 기초 코드를 제공받기 어렵습니다.

따라서 저는 간단한 예제 코드를 만들어 보았습니다. 

좋아요 부탁드립니다. 


When I look at the official Sapiens GitHub repository, it feels overwhelming to know where to start.

Even after searching on Google, it’s challenging to find tutorials or basic Colab code examples.

So, I decided to create a simple example code myself.

Please give it a thumbs up!

## 시작하기 

### 🔧 환경설정 

Set up the minimal sapiens_lite conda environment (pytorch >= 2.2):

```bash
conda create -n sapiens_lite python=3.10
conda activate sapiens_lite
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python tqdm json-tricks
```

### ⚙️ 모델 다운로드 

```bash
chmod +x download2.sh

./download2.sh
```

