import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cpu") # GPU 사용
target_n = 5 # 맞춰야하는 품목/품종의 수
learning_rate = 5e-4 # 학습률
BATCH_SIZE = 256 # 배치사이즈
EPOCHS = 50 # 총 eopochs
teacher_forcing = False # 교사강요 설정
n_layers = 3 # rnn레이어 층
dropout = 0.2 # 드롭아웃
window_size = 6 # 인코더 시퀀스 길이
future_size = 3 # 디코더 시퀀스 길이
hidden_dim = 256 # rnn 히든차원
save_path = f'./best_model.pt' # 모델 저장 경로

class CustomDataset(Dataset):
    def __init__(self, encoder_input, decoder_input):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        
    def __len__(self):
        return len(self.encoder_input)
    
    def __getitem__(self, i):
        return {
            'encoder_input' : torch.tensor(self.encoder_input[i], dtype=torch.float32),
            'decoder_input' : torch.tensor(self.decoder_input[i], dtype=torch.float32)
        }

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.n_layers = n_layers
        
        self.rnn = nn.GRU(input_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp_seq):
        inp_seq = inp_seq.permute(1,0,2)
        outputs, hidden = self.rnn(inp_seq)
        
        return outputs, hidden
    
class BahdanauAttention(nn.Module):
    def __init__(self, dec_output_dim, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(dec_output_dim, units)
        self.W2 = nn.Linear(dec_output_dim, units)
        self.V = nn.Linear(dec_output_dim, 1)

    def forward(self, hidden, enc_output):
        query_with_time_axis = hidden.unsqueeze(1)
        
        score = self.V(torch.tanh(self.W1(query_with_time_axis) + self.W2(enc_output)))
        
        attention_weights = torch.softmax(score, axis=1)
        
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, attention_weights
    
class Decoder(nn.Module):
    def __init__(self, dec_feature_size, encoder_hidden_dim, output_dim, decoder_hidden_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.n_layers = n_layers
        self.attention = attention
        
        self.layer = nn.Linear(dec_feature_size, encoder_hidden_dim)
        self.rnn = nn.GRU(encoder_hidden_dim*2, decoder_hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_output, dec_input, hidden):
        dec_input = self.layer(dec_input)
        context_vector, attention_weights = self.attention(hidden, enc_output)
        dec_input = torch.cat([torch.sum(context_vector, dim=0), dec_input], dim=1)
        dec_input = dec_input.unsqueeze(0)
        
        output, hidden = self.rnn(dec_input, hidden)

        prediction = self.fc_out(output.sum(0))
        
        return prediction, hidden
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, attention):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, encoder_input, decoder_input, teacher_forcing=False):
        batch_size = decoder_input.size(0)
        trg_len = decoder_input.size(1)
        
        outputs = torch.zeros(batch_size, trg_len-1, self.decoder.output_dim).to(device)
        enc_output, hidden = self.encoder(encoder_input)
        
        dec_input = decoder_input[:, 0]
        for t in range(1, trg_len):
            output, hidden = self.decoder(enc_output, dec_input, hidden)
            outputs[:, t-1] = output
            if teacher_forcing == True:
                dec_input = decoder_input[:, t]
            else:
                dec_input = output
        
        return outputs
            
def main():
    #Reading the file
    time_index = 'DateTime'
    data=pd.read_csv('data/AirQualityUCI.csv', parse_dates=[time_index])

    data.set_index(time_index, inplace=True)

    week_day_map = {}
    data['weekday'] = data.index.weekday
    for i, d in enumerate(data['weekday'].unique()):
        week_day_map[d] = i
    data['weekday'] = data['weekday'].map(week_day_map)

    norm = data.max(0)
    data = data / norm
    data = data.interpolate(method="ffill")

    x_data = []
    y_data = []
    for i in range(data.shape[0]-window_size-future_size):
        x = data.iloc[i:i+window_size].to_numpy()
        y = data[["CO(GT)","NMHC(GT)","C6H6(GT)","NOx(GT)","NO2(GT)"]].iloc[i+window_size:i+window_size+future_size].to_numpy()
        y_0 = np.zeros([1, y.shape[1]]) # 디코더 첫 입력값 추가
        x_data.append(x)
        y_data.append(np.concatenate([y_0, y], axis=0))
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    split_size = 0.9
    split_index = min(
        int(len(data) * split_size), len(data) - window_size - future_size
    )
    x_train = x_data[:-split_index]
    y_train = y_data[:-split_index]
    x_val = x_data[-split_index:]
    y_val = y_data[-split_index:]

    train_dataset = CustomDataset(x_train, y_train)
    val_dataset = CustomDataset(x_val, y_val)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,shuffle=False)

    encoder = Encoder(input_dim=x_data.shape[-1], hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
    attention = BahdanauAttention(dec_output_dim=hidden_dim, units=hidden_dim)
    decoder = Decoder(
        dec_feature_size=target_n, encoder_hidden_dim=hidden_dim, output_dim=target_n,
        decoder_hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout,
        attention = attention
    )

    model = Seq2Seq(encoder, decoder, attention)
    model = model.to(device)

    def my_custom_metric(pred, true):
        score = torch.mean(torch.abs((true-pred))/(true))
        return score

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss() # mae
    custom_metric = my_custom_metric
    
    print("Step 3")

    def train_step(batch_item, epoch, batch, training, teacher_forcing):
        encoder_input = batch_item['encoder_input'].to(device)
        decoder_input = batch_item['decoder_input'].to(device)
        if training is True:
            model.train()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(encoder_input, decoder_input, teacher_forcing)
                loss = criterion(output, decoder_input[:,1:])
                score = custom_metric(output, decoder_input[:,1:])
            loss.backward()
            optimizer.step()
            
            return loss, score
        else:
            model.eval()
            with torch.no_grad():
                output = model(encoder_input, decoder_input, False)
                loss = criterion(output, decoder_input[:,1:])
                score = custom_metric(output, decoder_input[:,1:])
            return loss, score
        
    loss_plot, val_loss_plot = [], []
    score_plot, val_score_plot = [], []

    for epoch in range(EPOCHS):
        total_loss, total_val_loss = 0, 0
        total_score, total_val_score = 0, 0
        
        tqdm_dataset = tqdm(enumerate(train_dataloader))
        training = True
        for batch, batch_item in tqdm_dataset:
            batch_loss, batch_score = train_step(batch_item, epoch, batch, training, teacher_forcing)
            total_loss += batch_loss
            total_score += batch_score
            
            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Loss': '{:06f}'.format(batch_loss.item()),
                'Total Loss' : '{:06f}'.format(total_loss/(batch+1)),
                'Score': '{:06f}'.format(batch_score.item()),
                'Total Score' : '{:06f}'.format(total_score/(batch+1)),
            })

        loss_plot.append((total_loss/(batch+1)).detach().numpy())
        score_plot.append((total_score/(batch+1)).detach().numpy())
        
        tqdm_dataset = tqdm(enumerate(val_dataloader))
        training = False
        for batch, batch_item in tqdm_dataset:
            batch_loss, batch_val_score = train_step(batch_item, epoch, batch, training, teacher_forcing)
            total_val_loss += batch_loss
            total_val_score += batch_val_score
            
            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Val Loss': '{:06f}'.format(batch_loss.item()),
                'Total Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
                'Val Score': '{:06f}'.format(batch_val_score.item()),
                'Total Val Score' : '{:06f}'.format(total_val_score/(batch+1)),
            })
        val_loss_plot.append(total_val_loss/(batch+1))
        val_score_plot.append(total_val_score/(batch+1))
        
        if np.min(val_loss_plot) == val_loss_plot[-1]:
            torch.save(model, save_path)
            
    plt.plot(loss_plot, label='train_loss')
    plt.plot(val_loss_plot, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss(mae)')
    plt.title('loss_plot')
    plt.legend()
    plt.show()

    plt.plot(score_plot, label='train_score')
    plt.plot(val_score_plot, label='val_score')
    plt.xlabel('epoch')
    plt.ylabel('score(nmae)')
    plt.title('score_plot')
    plt.legend()
    plt.show()

    model = torch.load(save_path)
    model = model.to(device)
    
    
    def predict(encoder_input):
        model.train()
        encoder_input = encoder_input.to(device)
        decoder_input = torch.zeros([1, future_size+1, target_n], dtype=torch.float32).to(device)
        with torch.no_grad():
            output = model(encoder_input, decoder_input, False)
        return output.cpu()

if __name__ == "__main__":
    main()