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


셀프 어텐션은 입력된 시퀀스 내에서 현재 위치 단어를 포함한 모든 단어와의 관련성을 계산하여 문맥을 더욱 정확하게 파악할 수 있도록 제안된 방법으로, 입력되는 시퀀스의 문맥에 대한 이해가 중요하게 작용하는 자연어처리 분야에서 핵심 역할을 하고 있다. 이러한 셀프 어텐션을 문장을 더 잘 이해할 수 있는 방향으로 이끌 수 있다면 문맥 파악에 대한 능력 향상을 통해 언어모델의 성능이 더욱 개선될 것이다.
본 논문에서는 한국어 형태소 규칙을 제작하여, 규칙을 통해 어텐션 레이블을 생성해 사전학습 언어모델의 0번 계층에서 학습되는 어텐션을 제어하는 방식을 제안하고자 한다. 어텐션의 제어를 통해 모델이 형태소 규칙을 학습하도록 유도하면 문맥을 더 잘 이해할 수 있을 것이며, 실제 실험에서 모델의 성능 향상을 통해 방법의 유효성을 확인할 수 있었다. 제안하는 방법론은 이후 형태소 규칙의 추가, 미세조정 과정이 아닌 사전학습 과정에 적용, 혹은 다른 계층에서의 적용이 가능하므로 다양한 파생적인 발전을 기대할 수 있을 것이다.

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
