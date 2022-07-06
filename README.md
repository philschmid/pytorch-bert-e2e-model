# Pytorch BERT E2E Model

This repository contains examples on how to build a e2e model including pre & post processing in the model graph using `pytorch` and `torchtext`. 
[TorchText received](https://pytorch.org/blog/pytorch-1.12-new-library-releases/) extended support for scriptable BERT tokenizer. See the release notes [here](https://github.com/pytorch/text/releases).

For usage details, please refer to the corresponding [documentation](https://pytorch.org/text/main/transforms.html#torchtext.transforms.BERTTokenizer).

```python
>>> from torchtext.transforms import BERTTokenizer
>>> VOCAB_FILE = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
>>> tokenizer = BERTTokenizer(vocab_path=VOCAB_FILE, do_lower_case=True, return_tokens=True)
>>> tokenizer("Hello World, How are you!") # single sentence input
>>> tokenizer(["Hello World","How are you!"]) # batch input
```

## Text Classification example

```python
# !pip install "torch>=1.12" "torchtext>=0.13" transformers
from typing import List, Union, Dict, Any
import torch
from torch import nn
from torchtext import transforms as T
from transformers import AutoModelForSequenceClassification

class PTBertTokenizer(nn.Module):
  def __init__(self,vocab_file_path=None,do_lower_case=True,bos_idx=101,eos_idx=102,padding_value=0,max_seq_len=512):
    super().__init__()
    self.tokenizer=T.Sequential(
      T.BERTTokenizer(vocab_path=vocab_file_path, do_lower_case=do_lower_case, return_tokens=False),
      T.StrToIntTransform(),
      T.Truncate(max_seq_len - 2),
      T.AddToken(token=bos_idx, begin=True),
      T.AddToken(token=eos_idx, begin=False),
      T.ToTensor(padding_value=padding_value)
  )

  def forward(self,input:Union[str,List[str]])->Dict[str,torch.Tensor]:
    input_ids = self.tokenizer(input)
    # shape tensor to matching format for transformers model
    if input_ids.dim() == 1:
      input_ids = torch.reshape(input_ids, (1,input_ids.shape[-1]))
    return {'input_ids':input_ids,"attention_mask":input_ids.gt(0).to(torch.int64)}

  @classmethod
  def from_pretrained(cls, model_id: str):
    remote_file=f"https://huggingface.co/{model_id}/resolve/main/vocab.txt"
    return cls(vocab_file_path=remote_file)


class E2ETextClassification(nn.Module):
  def __init__(self,model_id=None):
    super().__init__()
    self.tokenizer=PTBertTokenizer.from_pretrained(model_id)
    self.model = AutoModelForSequenceClassification.from_pretrained(model_id)

  def forward(self,inputs:Union[str,List[str]]) -> List[Dict[str,Any]]:
      # preprocessing
      tokenized = self.tokenizer(inputs)
      with torch.no_grad():
        logits = self.model(**tokenized).logits
        scores=nn.Softmax(dim=-1)(logits)
      # post processing
      return [{"label": self.model.config.id2label[score.argmax().item()], "score": score.max().item()} for score in scores]

pipe = E2ETextClassification("distilbert-base-uncased-finetuned-sst-2-english")
scores = pipe(["I like you","i hate you very much"])
print(scores)
scores = pipe("I like you")
print(scores)      

# [{'label': 'POSITIVE', 'score': 0.9998695850372314}, {'label': 'NEGATIVE', 'score': 0.9991033673286438}]
# [{'label': 'POSITIVE', 'score': 0.9998695850372314}]
```

### Save and Load the model

```python
torch.save(pipe, "pipe.pt")
loaded_pipe=torch.load("pipe.pt")
loaded_pipe("it is so awesome that i can load and save the whole pipeline")
```


## Caveats

* Token_type_ids not supported yet.
* not traceable: https://github.com/pytorch/text/pull/1707#issuecomment-1176272282