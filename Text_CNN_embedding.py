root_dir = "root_dir"

import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
import torch.optim as optim
import os
import numpy as np
from sklearn.metrics import accuracy_score

class TextCNN(nn.Module):
  def __init__(self, config):
    super(TextCNN, self).__init__()
    
    self.word_vocab_size = config["word_vocab_size"]
    self.embedding_size = config["embedding_size"]
    self.max_length = config["max_length"]
    self.num_labels = config["num_labels"]
    self._activation = nn.ReLU()
    # CNN에 각 필터의 개수
    self.num_filters = config["num_filters"]
    self.embedding = nn.Embedding(num_embeddings=self.word_vocab_size, embedding_dim=self.embedding_size, padding_idx=0)
    
    # 단어의 pos정보도 임베딩으로 활용. 임베딩해야 concat가능
    self.embedding_pos = nn.Embedding(num_embeddings=44, embedding_dim=self.embedding_size, padding_idx=0)
    self.convolution_layers = nn.ModuleList([nn.Conv1d(in_channels=self.embedding_size*2, out_channels=self.num_filters,
                                            kernel_size=int(n_gram)) for n_gram in config["filter_size"]])
                                            
    self.hidden2num_tag = nn.Linear(in_features=self.num_filters*len(config["filter_size"]), out_features=self.num_labels)

  def forward(self, input_features, labels=None):
  
    # 50:100까지는 pos (concat) 
    pos_feature = input_features[0:, 50:100]
    input_features = input_features[0:, 0:50]
    
    # permute로 축 변경 (CNN활용)
    pos_input_features = self.embedding_pos(pos_feature).permute(0,2,1)
    lookup_input_features = self.embedding(input_features).permute(0,2,1)
    
    lookup_input_features = torch.cat([lookup_input_features, pos_input_features], dim=1)
    # Convolution 한 값을 저장
    filter_outputs = []
    for i in range(len(self.convolution_layers)):
      
      convolution_output = self.convolution_layers[i](lookup_input_features).permute(0, 2, 1).max(dim=1)[0]
      activated_output = self._activation(convolution_output)
      filter_outputs.append(activated_output)
    maxpool_output = torch.cat(tensors=filter_outputs, dim=-1)
    logits = self.hidden2num_tag(maxpool_output)
    
    # 라벨이 있으면 학습, 없으면 평가
    if labels is not None:
      loss_fnc = nn.CrossEntropyLoss()
      loss = loss_fnc(logits, labels)
      return loss
    else:
      output = torch.argmax(logits, -1)
      return output
      
      
from sklearn.preprocessing import OneHotEncoder

# 파라미터로 입력받은 파일에 저장된 단어 리스트를 딕셔너리 형태로 저장
def load_vocab(f_name):
  vocab_file = open(os.path.join(root_dir, f_name),'r',encoding='utf8')
  print("{} vocab file loading...".format(f_name))
  symbol2idx, idx2symbol = {"<PAD>":0, "<UNK>":1}, {0:"<PAD>", 1:"<UNK>"}

  index = len(symbol2idx)
   for line in tqdm(vocab_file.readlines()):
      symbol = line.strip()
      symbol2idx[symbol] = index
      idx2symbol[index]= symbol
      index+=1
    return symbol2idx, idx2symbol
    
# 입력 데이터를 고정 길이의 벡터로 변환
def convert_data2feature(data, symbol2idx, pos2idx = None, max_length=None):
  if max_length:
    feature = np.zeros(shape=(max_length), dtype=np.int)
    pos_lst = []
    words = data.split()
    for idx, word in enumerate(words[:max_length]):
      if word in symbol2idx.keys():
        feature[idx] = symbol2idx[word]
      else:
        feature[idx] = symbol2idx["<UNK>"]
        
      # 새로 작성한 코드
      temp_pos = ''
      for j in range(len(word)-1, -1, -1):
        temp_pos = word[j] + temp_pos
        if word[j] == '/':
          pos_lst.append(pos2idx[temp_pos[1:]])
          break
    return feature, pos_lst
    
  else:
    return symbol2idx[data]

# 파라미터로 입력받은 파일로부터 tensor 객체 생성
def load_data(config, f_name):
  
  file = open(os.path.join(root_dir, f_name),'r',encoding='utf8')
  word2idx, idx2word = load_vocab(config["word_vocab_file"])
  label2idx, idx2label = load_vocab(config["label_vocab_file"])
  pos2idx, idx2pos = load_vocab(config["pos_vacab_file"])
  indexing_questions, indexing_labels, indexing_poses = [], [], []
  print("{} file loading...".format(f_name))

  for line in tqdm(file.readlines()):
    question, label = line.strip().split('\t')
    indexing_question, indexing_pos = convert_data2feature(question, word2idx, pos2idx, config["max_length"])
    indexing_label = convert_data2feature(label, label2idx)
    indexing_questions.append(indexing_question)
    indexing_labels.append(indexing_label)
    indexing_poses.append(indexing_pos)
  for index in range(len(indexing_poses)):
    for j in range(50 - len(indexing_poses[index])):
      indexing_poses[index].append(0)
  indexing_questions = np.array(indexing_questions)
  indexing_poses = np.array(indexing_poses)
  
  indexing_questions = np.concatenate((indexing_questions, indexing_poses), axis = 1)
  indexing_questions = torch.tensor(indexing_questions, dtype=torch.long)
  
  indexing_labels = torch.tensor(indexing_labels, dtype=torch.long)
  
  return indexing_questions, indexing_labels

def tensor2list(input_tensor):
  return input_tensor.cpu().detach().numpy().tolist()


def train(config):
  model = TextCNN(config).cuda()
  train_input_features, train_labels = load_data(config, config["train_file"])
  test_input_features, test_labels = load_data(config, config["test_file"])

  train_features = TensorDataset(train_input_features, train_labels)
  train_dataloader = DataLoader(train_features, shuffle=True, batch_size=config["batch_size"])
  
  test_features = TensorDataset(test_input_features, test_labels)
  test_dataloader = DataLoader(test_features, shuffle=True, batch_size=config["batch_size"])
  
  optimizer = optim.Adam(model.parameters(), lr=0.0005)

  for epoch in range(config["epoch"]):
    losses = []
    for step, batch in enumerate(train_dataloader):
      batch = tuple(t.cuda() for t in batch)
      input_features, labels = batch
      loss = model(input_features, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if (step + 1) % 100 == 0:
        print("{} step processed.. current loss : {}".format(step + 1, loss.data.item()))
      losses.append(loss.data.item())
    print("Average Loss : {}".format(np.mean(losses)))
    torch.save(model.state_dict(), os.path.join(config["output_dir_path"], "epoch_{}.pt".format(epoch + 1)))

    do_test(model, test_dataloader)
    
def test(config):
  model = TextCNN(config).cuda()
  model.load_state_dict(torch.load(os.path.join(config["output_dir_path"], config["trained_model_name"])))
  test_input_features, test_labels = load_data(config, config["test_file"])
  test_features = TensorDataset(test_input_features, test_labels)
  test_dataloader = DataLoader(test_features, shuffle=True, batch_size=config["batch_size"])
  do_test(model, test_dataloader)
  
def do_test(model, test_dataloader):
  model.eval()
  predicts, answers = [], []
  for step, batch in enumerate(test_dataloader):
    batch = tuple(t.cuda() for t in batch)
    input_features, labels = batch
    output = model(input_features)
    predicts.extend(tensor2list(output))
    answers.extend(tensor2list(labels))
  print("Accuracy : {}".format(accuracy_score(answers, predicts)))
  
if(__name__=="__main__"):
  output_dir = os.path.join(root_dir, "output")
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
  config = {"mode": "test",
            "train_file":"train.txt",
            "test_file": "test.txt",
            "word_vocab_file":"vocab.txt",
            "label_vocab_file": "label_vocab.txt",
            "pos_vacab_file" : "pos_vocab.txt",
            "trained_model_name":"epoch_{}.pt".format(5),
            "output_dir_path":output_dir,
            "word_vocab_size":16385,
            "num_labels": 12,
            "filter_size":[2, 3, 4],
            "num_filters":100,
            "embedding_size":200,
            "max_length": 50,
            "batch_size":32,
            "epoch":5,
            }
  if(config["mode"] == "train"):
    train(config)
  else:
    test(config)
