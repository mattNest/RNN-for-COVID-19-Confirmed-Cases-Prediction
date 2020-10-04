#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.data as Data
import torchvision.datasets 
import torchvision.transforms

import pygal
from IPython.display import SVG, display

import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# In[2]:


df_covid_19 = pd.read_csv("covid_19.csv").fillna(0)
print(df_covid_19.shape)


# In[3]:


df_covid_19


# ### Correlation

# In[4]:


diff_seq = []
for row in range(2, df_covid_19.shape[0]):
    for col in range(4, df_covid_19.shape[1]):
        diff = float(df_covid_19.iloc[row,col]) - float(df_covid_19.iloc[row,col-1])
        diff_seq.append(diff)

diff_seq = np.array(diff_seq).reshape(185,81)
diff_seq


# In[5]:


# The difference sequence indexed by country
country_list = list(df_covid_19.iloc[2:,0])
df_seq = pd.DataFrame(diff_seq, index=country_list)
df_seq


# In[6]:


# heatmap for all countries
correlation_all = df_seq.T.corr()
#print(correlation)

sns.set(style="white")
mask = np.zeros_like(correlation_all, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
#cmap = sns.diverging_palette(220, 10, as_cmap=True)

heatmap_all = sns.heatmap(
            correlation_all, 
            mask = mask, 
            cmap = 'Reds', 
            vmin = -1.0,
            vmax = 1.0,
            linewidths = 0.5, 
            cbar_kws = {"shrink": 0.5},
            square = True)

heatmap_all = heatmap_all.get_figure()
heatmap_all.savefig("heatmap_all.png")


# In[7]:


# heatmap for first 10 countries
correlation_first_10 = df_seq.iloc[:10,:].T.corr()
#print(correlation_first_10)

sns.set(style="white")
mask = np.zeros_like(correlation_first_10, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
#cmap = sns.diverging_palette(220, 10, as_cmap=True)

heatmap_first_10 = sns.heatmap(
            correlation_first_10,
            cmap = "Reds",
            mask = mask,
            center = 0,
            vmin = -1,
            vmax = 1,
            xticklabels = df_seq.iloc[:10,:].index, 
            yticklabels = df_seq.iloc[:10,:].index,
            square = True)

heatmap_first_10 = heatmap_first_10.get_figure()
heatmap_first_10.savefig("heatmap_first_10.png")


# ### Data Preprocessing

# ![title](data_prepro_workflow.png)

# In[8]:


correlation_all


# In[9]:


correlation_all['China']


# #### First Step

# In[10]:


df_tmp = correlation_all > 0.7 # set threshold = 0.7
count_true = 0
count_false = 0
set_C = []

for row in range(0,df_tmp.shape[0]):
    for col in range(0,row):
        if df_tmp.iloc[row,col]==True:
            count_true += 1
            set_C.append([df_tmp.index[row], df_tmp.columns[col]]) # collect country pair into Set C
        if df_tmp.iloc[row,col]==False:
            count_false += 1

# find unique values in set C
set_C_unique = np.unique(np.array(set_C))
#print([count_true, count_false])
#print("Set C / all correlation data: {0} %".format(count_true/count_false*100))
print(set_C_unique.shape[0]/185) # choosen countries / all countries
print(set_C_unique)


# In[11]:


df_tmp = correlation_all > 0.6 # set threshold = 0.6
count_true = 0
count_false = 0
set_C = []

for row in range(0,df_tmp.shape[0]):
    for col in range(0,row):
        if df_tmp.iloc[row,col]==True:
            count_true += 1
            set_C.append([df_tmp.index[row], df_tmp.columns[col]]) # collect country pair into Set C
        if df_tmp.iloc[row,col]==False:
            count_false += 1

# find unique values in set C
set_C_unique = np.unique(np.array(set_C))
#print([count_true, count_false])
#print("Set C / all correlation data: {0} %".format(count_true/count_false*100))
print(set_C_unique.shape[0]/185) # choosen countries / all countries
print(set_C_unique)


# In[12]:


df_tmp = correlation_all > 0.7 # set threshold = 0.7
count_true = 0
count_false = 0
set_C = []

for row in range(0,df_tmp.shape[0]):
    for col in range(0,row):
        if df_tmp.iloc[row,col]==True:
            count_true += 1
            set_C.append([df_tmp.index[row], df_tmp.columns[col]]) # collect country pair into Set C
        if df_tmp.iloc[row,col]==False:
            count_false += 1

# find unique values in set C
set_C_unique = np.unique(np.array(set_C))
#print([count_true, count_false])
#print("Set C / all correlation data: {0} %".format(count_true/count_false*100))
print(set_C_unique.shape)
print(set_C_unique.shape[0]/185) # choosen countries / all countries
print(set_C_unique)


# #### Second / Third / Fourth Step

# In[13]:


df_seq


# In[14]:


# # one country one subsequence

# df_seq_selected = df_seq.loc[set_C_unique]

# interval_L = 7
# label = []

# for row in range(len(df_seq_selected)):
#         sub_sequence = df_seq_selected.iloc[row, 40:(40+interval_L)]
#         next_day = df_seq_selected.iloc[row, 40+interval_L]

#         #print(sub_sequence.iloc[-1], next_day)

#         if next_day > sub_sequence.iloc[-1]:
#             label.append(1)
#             df_seq_selected_multiple.append([sub_sequence.values, 1])

#         if next_day <= sub_sequence.iloc[-1]:
#             label.append(0)
#             df_seq_selected_multiple.append([sub_sequence.values, 0])


# print(len(label))
# print("Label 1: {0}, Label 0: {1}".format(label.count(1), label.count(0)))

# df_seq_selected.insert(0,"label",label, True)


# In[48]:


# one country many subsequences

df_seq_selected = df_seq.loc[set_C_unique]

interval_L = 3

label = []
subsequences = []
df_seq_selected_multiple = []

for row in range(len(df_seq_selected)):
    for col in range(0, len(df_seq_selected.iloc[0,:])-3):
        
        sub_sequence = df_seq_selected.iloc[row, col:(col+interval_L)]
        next_day = df_seq_selected.iloc[row, col+interval_L]

        #print(sub_sequence.iloc[-1], next_day)

        if next_day > sub_sequence.iloc[-1]:
            label.append(1)
            subsequences.append(sub_sequence)
            
            df_seq_selected_multiple.append([sub_sequence.values, 1])

        if next_day <= sub_sequence.iloc[-1]:
            label.append(0)
            subsequences.append(sub_sequence)
            
            df_seq_selected_multiple.append([sub_sequence.values, 0])


print(len(df_seq_selected_multiple), len(subsequences))
print(len(subsequences)*0.8, len(subsequences)*0.2)
print("Label 0: {0}, Label 1: {1}".format(label.count(0), label.count(1)))


# ### Model

# In[49]:


# # one country one subsequence
# # all data = 143 --> 80% training data, 20% testing data
# x_train = torch.from_numpy(df_seq_selected.iloc[:114, 40:(40+interval_L)].values)
# y_train = torch.from_numpy(df_seq_selected.iloc[:114, 0].values)
# x_train = x_train.view(-1, interval_L, 1).type(torch.FloatTensor) # (total data, time_step, input)
# y_train = y_train.view(-1,1)

# training_dataset = Data.TensorDataset(x_train, y_train)

# x_test = torch.from_numpy(df_seq_selected.iloc[114:, 40:(40+interval_L)].values)
# y_test = torch.from_numpy(df_seq_selected.iloc[114:, 0].values)
# x_test = x_test.view(-1, interval_L, 1).type(torch.FloatTensor) # (total data, time_step, input)
# y_test = y_test.view(-1,1)

# testing_dataset = Data.TensorDataset(x_test, y_test)


# In[50]:


# one country many subsequence

pos = int(len(subsequences)*0.8)
print(pos)

x_train = torch.from_numpy(np.array(subsequences[0:pos]))
y_train = torch.from_numpy(np.array(label[0:pos]))

x_train = x_train.view(-1, interval_L, 1).type(torch.FloatTensor) # (total data, time_step, input)
y_train = y_train.view(-1,1)

training_dataset = Data.TensorDataset(x_train, y_train)

x_test = torch.from_numpy(np.array(subsequences[pos:]))
y_test = torch.from_numpy(np.array(label[pos:]))

x_test = x_test.view(-1, interval_L, 1).type(torch.FloatTensor) # (total data, time_step, input)
y_test = y_test.view(-1,1)

testing_dataset = Data.TensorDataset(x_test, y_test)


# In[51]:


training_dataset[0]


# In[52]:


testing_dataset[0]


# In[54]:


train_loader = Data.DataLoader(
    dataset = training_dataset,
    shuffle = False,
    batch_size = 256,                    
    num_workers = 32,              
)

test_loader = Data.DataLoader(
    dataset = testing_dataset,
    shuffle = False,
    batch_size = 256,      
    num_workers = 32,            
)


# In[55]:


class history_package():
    def __init__(self, neural_net, train_loader, test_loader, EPOCH, LR, model_type):
        
        self.net = neural_net
        self.optimizer = torch.optim.Adam(neural_net.parameters(), lr = LR)
        self.outputs_prob_all = 0
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.EPOCH_ = EPOCH
        self.LR_ = LR
        self.net = self.net.to(device)
        self.model_type = model_type
        
        if device == 'cuda':
            torch.backends.cudnn.benchmark = True

    def start(self):

        history_loss = []
        history_train_acc = []
        history_test_acc = []
        
        for epoch in range(self.EPOCH_):
            print('Epoch:', epoch)
            print("============================")
            
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.test()

            history_loss.append(train_loss)
            history_train_acc.append(train_acc)
            history_test_acc.append(test_acc)
        
        return history_loss, history_train_acc, history_test_acc

    def train(self):
        
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for step, (batch_X, batch_y) in enumerate(self.train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            self.optimizer.zero_grad()
            outputs = self.net(batch_X)
            
            
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()        
        
        
        print('【Training】Loss: %.3f | Acc: %.3f%% (%d/%d)' % ( train_loss, 100.*(correct/total), correct, total))
        
        # save the model
        
        if self.model_type == "RNN":
            torch.save(self.net.state_dict(), 'RNN_model.pth')
        
        if self.model_type == "LSTM":
            torch.save(self.net.state_dict(), 'LSTM_model.pth')
        
        if self.model_type == "GRU":
            torch.save(self.net.state_dict(), 'GRU_model.pth')
            

        return train_loss, (correct/total)

    def test(self):
        self.net.eval()

        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad(): 
            for step, (batch_X, batch_y) in enumerate(self.test_loader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = self.net(batch_X)
                
                loss = self.criterion(outputs, batch_y)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                #print(predicted)
                
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()   
        
        print('【Testing】Loss: %.3f | Acc: %.3f%% (%d/%d)' % ( test_loss, 100.*(correct/total), correct, total ))
        
        return test_loss, (correct/total)


# #### RNN

# input_size: Corresponds to the number of features in the input. <br>
# --> the input_size of sequential data (ex: time seires) is equal to 1

# In[56]:


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        
        self.rnn = nn.RNN(input_size = 1,
                          hidden_size = 128,
                          num_layers = 1,
                          batch_first = True) 

        self.out = nn.Linear(in_features = 128, 
                             out_features = 1)
        
    def forward(self, x):
        out, _ = self.rnn(x, None)
        out = out.squeeze(0)
        out = self.out(out.squeeze(1))
        
        return out


# In[57]:


print(RNN())


# In[60]:


start_time = time.time()

rnn_module = history_package(RNN(), 
                             train_loader, 
                             test_loader, 
                             EPOCH=1000, 
                             LR=0.001,
                             model_type = "RNN")

history_loss_rnn, history_train_acc_rnn, history_test_acc_rnn = rnn_module.start()

end_time = time.time()
print('Training Time Cost: ',time.strftime("%H hr %M min %S sec", time.gmtime(end_time - start_time)))


# In[61]:


plt.figure(figsize=(12,6))
plt.plot(history_loss_rnn)
plt.xlabel("Number of Epochs")
plt.ylabel("Cross Entropy")
plt.title("RNN Learning Curve")
plt.savefig("RNN_learning_curve")


# In[62]:


plt.figure(figsize=(12,6))
plt.plot(history_train_acc_rnn, label="training acc")
plt.plot(history_test_acc_rnn, label="testing acc")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="upper right")
plt.title("RNN Training & Testing Accuracy (interval=3)")
plt.savefig("RNN_training_accuracy_interval_3")


# In[63]:


model_trained = RNN()
model_trained.load_state_dict(torch.load("RNN_model.pth"))
model_trained.eval()


# In[64]:


F.softmax(model_trained.forward(x_test)).shape


# #### LSTM

# In[65]:


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size = 1,
                          hidden_size = 128,
                          num_layers = 1,
                          batch_first = True) 

        self.out = nn.Linear(in_features = 128, 
                             out_features = 1)
        
    def forward(self, x):
        out, _ = self.lstm(x, None)
        out = out.squeeze(0)
        out = self.out(out.squeeze(1))
        #print(out)
        
        return out


# In[66]:


print(LSTM())


# In[67]:


start_time = time.time()

lstm_module = history_package(LSTM(), 
                             train_loader, 
                             test_loader, 
                             EPOCH=1000, 
                             LR=0.001,
                             model_type = "LSTM")

history_loss_lstm, history_train_acc_lstm, history_test_acc_lstm = lstm_module.start()

end_time = time.time()
print('Training Time Cost: ',time.strftime("%H hr %M min %S sec", time.gmtime(end_time - start_time)))


# In[68]:


plt.figure(figsize=(12,6))
plt.plot(history_loss_lstm)
plt.xlabel("Number of Epochs")
plt.ylabel("Cross Entropy")
plt.title("LSTM Learning Curve")
plt.savefig("LSTM_learning_curve")


# In[69]:


plt.figure(figsize=(12,6))
plt.plot(history_train_acc_lstm, label="training acc")
plt.plot(history_test_acc_lstm, label="testing acc")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="upper right")
plt.title("LSTM Training & Testing Accuracy (interval = 3)")
plt.savefig("LSTM_training_accuracy_interval_3")


# #### GRU

# In[273]:


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size = 1,
                          hidden_size = 128,
                          num_layers = 1,
                          batch_first = True) 

        self.out = nn.Linear(in_features = 128, 
                             out_features = 1)
        
    def forward(self, x):
        out, _ = self.gru(x, None)
        out = out.squeeze(0)
        out = self.out(out.squeeze(1))
        #print(out)
        
        return out


# In[274]:


print(GRU())


# In[275]:


start_time = time.time()

gru_module = history_package(GRU(),
                             train_loader, 
                             test_loader, 
                             EPOCH=1000, 
                             LR=0.001,
                             model_type = "GRU")

history_loss_gru, history_train_acc_gru, history_test_acc_gru = gru_module.start()


end_time = time.time()
print('Training Time Cost: ',time.strftime("%H hr %M min %S sec", time.gmtime(end_time - start_time)))


# In[276]:


plt.figure(figsize=(12,6))
plt.plot(history_loss_gru)
plt.xlabel("Number of Epochs")
plt.ylabel("Cross Entropy")
plt.title("GRU Learning Curve")
plt.savefig("GRU_learning_curve")


# In[277]:


plt.figure(figsize=(12,6))
plt.plot(history_train_acc_gru, label="training acc")
plt.plot(history_test_acc_gru, label="testing acc")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="upper right")
plt.title("GRU Training & Testing Accuracy (interval = 3)")
plt.savefig("GRU_training_accuracy_interval_3")


# #### World Map

# In[79]:


# The difference sequence indexed by country
df_seq


# In[80]:


# one country many subsequences

df_seq_selected = df_seq

interval_L = 3

label = []
subsequences = []
df_seq_selected_multiple = []

for row in range(len(df_seq_selected)):
    sub_sequence = df_seq_selected.iloc[row, 40:(40+interval_L)]
    next_day = df_seq_selected.iloc[row, 40+interval_L]

    #print(sub_sequence.iloc[-1], next_day)

    if next_day > sub_sequence.iloc[-1]:
        label.append(1)
        subsequences.append(sub_sequence)

        df_seq_selected_multiple.append([sub_sequence.values, 1])

    if next_day <= sub_sequence.iloc[-1]:
        label.append(0)
        subsequences.append(sub_sequence)

        df_seq_selected_multiple.append([sub_sequence.values, 0])


print(len(df_seq_selected_multiple), len(subsequences))
print(len(subsequences)*0.8, len(subsequences)*0.2)
print("Label 0: {0}, Label 1: {1}".format(label.count(0), label.count(1)))


# In[265]:


to_be_predicted = torch.from_numpy(np.array(subsequences))
to_be_predicted = to_be_predicted.view(-1, interval_L, 1).type(torch.FloatTensor) # (total data, time_step, input)


# In[266]:


model_trained = LSTM()
model_trained.load_state_dict(torch.load("LSTM_model.pth"))
model_trained.eval()


# In[267]:


predicted = []
_, predicted = model_trained(to_be_predicted).max(1)
predicted[0]


# In[268]:


ascending = []
decending = []

for i, (result,country) in enumerate(zip(predicted, country_list)):
    if predicted[i][0] == 0:
        decending.append(country_list[i])
    if predicted[i][0] == 1:
        ascending.append(country_list[i])


# In[269]:


from pygal_maps_world.i18n import COUNTRIES

def get_country_code(country_name):
    for code, name in COUNTRIES.items():  # 返回字典的所有鍵值對
        if name == country_name:  # 根據國家名返回兩個字母的國別碼
            return code
    return None  # 如果沒有找到則返回None


# In[270]:


ascending_country_code = []
decending_country_code = []

for i in range(len(ascending)):
    ascending_country_code.append(get_country_code(ascending[i]))

for j in range(len(decending)):
    decending_country_code.append(get_country_code(decending[j]))

ascending_country_code.append('south_america')
ascending_country_code = [country for country in ascending_country_code if str(country) != 'None']
decending_country_code = [country for country in decending_country_code if str(country) != 'None']


# In[271]:


ascending_country_code


# In[272]:


worldmap_chart = pygal.maps.world.World()
worldmap_chart.title = 'COVID 19 Trend'
worldmap_chart.add('Ascending', ascending_country_code)
worldmap_chart.add('Descending', decending_country_code)
display(SVG(worldmap_chart.render(disable_xml_declaration=True)))


# In[ ]:




