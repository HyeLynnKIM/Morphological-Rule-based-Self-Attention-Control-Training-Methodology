# Morphological-Rule-based-Self-Attention-Control-Training-Methodology


## File Direcotry
```bash
├── sts_data
│   └── model
├── grammer_data
│  └── 10_.prompt_문법성 판단.jsonl
├── Chunker.ini
├── HTML_Processor.toml
├── HTML_Utils.txt
├── KLUE_PN.txt
├── KLUE_STS.sh
├── Ranking_ids.sh
├── evaluate2.py
├── get_pn_datasets.py
├── get_pn_datasets2.py
├── optimization.py
├── pos_augmented_tokenizer.py
├── utils.py
├── modeling2.py
├── modeling_electra.py
├── tokenization.py
├── super_main.py
├── super_main2.py
├── super_main3.py
├── layers_.py
├── ber_config_mecab_base_rr.json
└── 규칙.txt
``` 

# Abstract

---
# Model

![image](https://github.com/HyeLynnKIM/Morphological-Rule-based-Self-Attention-Control-Training-Methodology/assets/64192139/ce6a6cb6-1536-4b7d-8c85-6a2a00fe36cb)

#### [그림 1] 모델 전체 학습 구조 및 규칙 기반 어텐션 제어 세부 구조
---
# File Description

---
# Morphological Rule
|어텐션 제어 규칙|내용|예시|
|:---:|:---:|:---:|
|의존 명사|의존 명사와 그 앞의 수식어구를 서로 강조한다.|네가 알려준 만큼만 알고 있어.|
|조사 (보격 조사 제외)|조사와 그 앞의 체언을 서로 강조한다.|나는 집에 왔다.|
|보격 조사|보격 조사와 그 앞의 보어를 서로 강조한다. 또한, 보어는 주어와 함께 강조한다|얼음은 물이 되었다.|
|관형사|관형사와 그 수식되는 체언을 서로 강조한다.|진짜 멋진 신발이다.|
|부사|부사와 그 수식되는 품사를 서로 강조한다.|밥을 빨리 먹자.|
|보조 용언| 보조 용언과 본용언을 서로 강조한다.|나는 그렇게 알고 있다.|
