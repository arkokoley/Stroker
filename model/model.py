import torch.nn as nn
import torch
import torch.nn.functional as F
from base import BaseModel

class StrokeModel(BaseModel):
    def __init__(self, embedding_dim, hidden_dim, num_classes=10):
        super(StrokeModel, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, num_classes)

        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, num_classes)
        self.hidden1 = self.init_hidden1()
        self.hidden2 = self.init_hidden2()

        self.W_s1 = nn.Linear(hidden_dim, 30)
        self.W_s2 = nn.Linear(30, hidden_dim)

    def init_hidden1(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.randn(1, 1, self.hidden_dim),
                torch.randn(1, 1, self.hidden_dim))

    def init_hidden2(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))
    
    def attention_net(self, hidden, lstm_output):
        # attn_weight_matrix = torch.randn(self.hidden_dim, 1, 1)
        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(hidden.view(1, -1))))
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=1)
        attn_weight_matrix = torch.mm(attn_weight_matrix, lstm_output.view(-1, len(lstm_output)))
        return attn_weight_matrix

    def forward(self, seq):
        lstm_out1, self.hidden1 = self.lstm1(seq.view(len(seq),1, -1), self.hidden1)
        lstm_out2, self.hidden2 = self.lstm2(lstm_out1.view(len(lstm_out1), 1, -1), self.hidden2)
        (final_hidden_state, final_cell_state) = self.hidden2
        attn_output = self.attention_net(self.hidden2[0], lstm_out2)
        hidden_matrix = torch.mm(attn_output, lstm_out2.view(len(lstm_out2),-1))#.squeeze(0).view(-1, 1, 30, 20)
        # print(hidden_matrix.size())
        tag_space = self.hidden2tag(hidden_matrix)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores