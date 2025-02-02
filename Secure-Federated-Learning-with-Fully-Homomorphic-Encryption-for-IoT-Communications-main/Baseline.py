
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tenseal as ts
import time
import torch
import math
import torch.nn.functional as F
import sys
import csv

from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix 

def initialize_models(number_of_samples, input_size=784, hidden_size=128, num_classes=2):
    model_dict = {}
    for i in range(number_of_samples):
        model = Net(input_size, hidden_size, num_classes)  # Net sınıfınızı kullanarak
        model = model.float()
        model_dict[f'model_{i}'] = model
    return model_dict

class Net(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
pd.options.display.float_format = "{:,.4f}".format

# Veri setini indirme ve yükleme
dataset_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Veri setini yükleme
dataset_1 = pd.read_csv(dataset_url, header=None, names=column_names)

num_of_features = (dataset_1.shape[1])-1


split_idx = math.floor(len(dataset_1) * 0.75)
train_DS = dataset_1[0:split_idx]
test_DS = dataset_1[split_idx:]

x_trainn = train_DS.iloc[:,0:-1]
y_trainn= train_DS.iloc[:,-1]
x_trainn = np.array(x_trainn)
y_trainn = np.array(y_trainn)


x_test =test_DS.iloc[:,0:-1]
y_test =test_DS.iloc[:,-1]
x_test = np.array(x_test)
y_test = np.array(y_test)

batch_size=32

split_idx = math.floor(len(x_trainn) * 0.8)
x_train, y_train = x_trainn[:split_idx], y_trainn[:split_idx]
x_valid, y_valid = x_trainn[split_idx:], y_trainn[split_idx:]

x_train_tensor = torch.tensor(x_train)
y_train_tensor = torch.tensor(y_train)

x_valid_tensor = torch.tensor(x_valid)
y_valid_tensor = torch.tensor(y_valid)

x_test_tensor = torch.tensor(x_test)
y_test_tensor = torch.tensor(y_test)

x_train.shape, y_train.shape , x_valid.shape, y_valid.shape, x_test.shape, y_test.shape

y_train_total=0
y_valid_total=0
y_test_total=0
total=0
for i in range(2):
    y_train_total=y_train_total + sum(y_train==i)
    y_valid_total=y_valid_total + sum(y_valid==i)
    y_test_total=y_test_total + sum(y_test==i)
    total=total+sum(y_train==i)+sum(y_valid==i)+sum(y_test==i)
    

def dividing_and_shuffling_labels(y_label, seed, amount):
    y_label=pd.DataFrame(y_label,columns=["labels"])
    y_label["i"]=np.arange(len(y_label))
    label_y_dict = dict()
    for i in range(2):
        var_name="label" + str(i)
        label_info=  y_label[y_label["labels"]==i]
        np.random.seed(seed)
        label_info=np.random.permutation(label_info)
        label_info=label_info[0:amount]
        label_info=pd.DataFrame(label_info, columns=["labels","i"])
        label_y_dict.update({var_name: label_info })
    return label_y_dict

def get_subsamples(label_dict, number_of_samples, amount):
    sample_dict= dict()
    batch_size=int(math.floor(amount/number_of_samples))
    for i in range(number_of_samples):
        sample_name="sample"+str(i)
        dumb=pd.DataFrame()
        for j in range(2):
            label_name=str("label")+str(j)
            a=label_dict[label_name][i*batch_size:(i+1)*batch_size]
            dumb=pd.concat([dumb,a], axis=0)
        dumb.reset_index(drop=True, inplace=True)    
        sample_dict.update({sample_name: dumb}) 
    return sample_dict

def create_subsamples(sample_dict, x_data, y_data, x_name, y_name):
    x_data_dict= dict()
    y_data_dict= dict()
    
    for i in range(len(sample_dict)):  
        xname= x_name+str(i)
        yname= y_name+str(i)
        sample_name="sample"+str(i)
        
        indices=np.sort(np.array(sample_dict[sample_name]["i"]))
        
        x_info= x_data[indices,:]
        x_data_dict.update({xname : x_info})
        
        y_info= y_data[indices]
        y_data_dict.update({yname : y_info})
        
    return x_data_dict, y_data_dict

class Net2nn(nn.Module):
    def __init__(self):
        super(Net2nn, self).__init__() 
        self.fc1=nn.Linear(num_of_features,10) 
        self.fc2=nn.Linear(10,10) 
        self.fc3=nn.Linear(10,2)     
        
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

class enc_model_weight:
  def __init__(self):
    self.fc1_enc = torch.zeros(size= [10,num_of_features])
    self.fc2_enc = torch.zeros(size= [10,10])
    self.fc3_enc = torch.zeros(size= [2,10])

    self.fc1_enc_B1 = torch.zeros(size= [10])
    self.fc2_enc_B2 = torch.zeros(size= [10])
    self.fc3_enc_B3 = torch.zeros(size= [10])

class dec_model_weight:
  def __init__(self):
    self.fc1_dec = torch.zeros(size= [10,num_of_features])
    self.fc2_dec = torch.zeros(size= [10,10])
    self.fc3_dec = torch.zeros(size= [2,10])

    self.fc1_dec_B1 = torch.zeros(size= [10])
    self.fc2_dec_B2 = torch.zeros(size= [10])
    self.fc3_dec_B3 = torch.zeros(size= [10])
    
def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    correct = 0
    latency_list = []

    for data, target in train_loader:
        output = model(data)
        loss = criterion(output, target.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

        start_time = time.time()
        prediction = output.argmax(dim=1, keepdim=True)
        end_time = time.time()
        latency = end_time - start_time
        latency_list.append(latency)

        correct += prediction.eq(target.view_as(prediction)).sum().item()

    train_loss /= len(train_loader)
    correct /= len(train_loader.dataset)
    latency_avg = sum(latency_list) / len(latency_list)
            
    return train_loss, correct, latency_avg

def validation(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    recall_all = 0
    precision_all = 0
    f1_score_all = 0

    TN_all = 0
    FP_all = 0
    FN_all = 0
    TP_all = 0

    with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target.long()).item()

                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(target.view_as(prediction)).sum().item()
                
                # Convert tensors to numpy arrays
                target_numpy = target.cpu().numpy()
                prediction_numpy = prediction.cpu().numpy().reshape(-1)  # Flatten predictions

                recall = recall_score(target_numpy, prediction_numpy)
                recall_all += recall

                precision = precision_score(target_numpy, prediction_numpy)
                precision_all += precision
                
                f1 = f1_score(target_numpy, prediction_numpy)
                f1_score_all += f1

                cm = confusion_matrix(target_numpy, prediction_numpy, labels=[0,1])
                TN, FP, FN, TP = cm.ravel()
                
                TN_all += TN
                FP_all += FP
                FN_all += FN
                TP_all += TP
             
             
    
    test_loss /= len(test_loader)
    correct /= len(test_loader.dataset)
    recall_all /= len(test_loader)
    precision_all /= len(test_loader)
    f1_score_all /= len(test_loader)



    return test_loss, correct, recall_all, precision_all, f1_score_all, TN_all, FP_all, FN_all, TP_all, len(test_loader)



def create_model_optimizer_criterion_dict(number_of_samples):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()
    encrypted_model_dict = dict()
    decrypted_model_dict = dict()

    for i in range(number_of_samples):
        model_name="model"+str(i)
        model_info=Net2nn()
        model_dict.update({model_name : model_info })

        enc_m_name="enc_model"+str(i)
        enc_m_info=enc_model_weight()
        encrypted_model_dict.update({enc_m_name : enc_m_info })

        dec_m_name="dec_model"+str(i)
        dec_m_info=dec_model_weight()
        decrypted_model_dict.update({dec_m_name : dec_m_info })
        
        optimizer_name="optimizer"+str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum)
        optimizer_dict.update({optimizer_name : optimizer_info })
        
        criterion_name = "criterion"+str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name : criterion_info})
        
    return model_dict, optimizer_dict, criterion_dict  , encrypted_model_dict , decrypted_model_dict

def decrypt(enc):
    return enc.decrypt().tolist()

def context():
    context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = pow(2, 40)
    context.generate_galois_keys()
    return context

context = context()

def apply_FHE(v1):

    v1_D1 = v1.size(dim=0)
    v1_D2 = v1.size(dim=1)
    plain1 = ts.plain_tensor(v1, [v1_D1, v1_D2])
    encrypted_tensor = ts.ckks_tensor(context, plain1)

    return encrypted_tensor

def enc_model(model_dict, encrypted_model_dict, number_of_samples):
  with torch.no_grad():

      for i in range(number_of_samples):
        model_c1 = model_dict[name_of_models[i]].fc1.weight.data.clone()
        E_W1 = apply_FHE(model_c1) # , Enc_size1
        encrypted_model_dict[name_of_enc_models[i]].fc1_enc = E_W1
        encrypted_model_dict[name_of_enc_models[i]].fc1_enc_B1 = model_dict[name_of_models[i]].fc1.bias.data.clone()
        
        model_c2 = model_dict[name_of_models[i]].fc2.weight.data.clone()
        E_W2 = apply_FHE(model_c2) # , Enc_size2
        encrypted_model_dict[name_of_enc_models[i]].fc2_enc = E_W2
        encrypted_model_dict[name_of_enc_models[i]].fc2_enc_B2 = model_dict[name_of_models[i]].fc2.bias.data.clone()

        model_c3 = model_dict[name_of_models[i]].fc3.weight.data.clone()
        E_W3 = apply_FHE(model_c3) # , Enc_size3
        encrypted_model_dict[name_of_enc_models[i]].fc3_enc = E_W3
        encrypted_model_dict[name_of_enc_models[i]].fc3_enc_B3 = model_dict[name_of_models[i]].fc3.bias.data.clone()
        
  return encrypted_model_dict

def dec_model(encrypted_model_dict, decrypted_model_dict, number_of_samples ):

  with torch.no_grad():
      for i in range(number_of_samples):

        fc1_dec_w = decrypt(encrypted_model_dict[name_of_enc_models[i]].fc1_enc)
        fc2_dec_w = decrypt(encrypted_model_dict[name_of_enc_models[i]].fc2_enc)
        fc3_dec_w = decrypt(encrypted_model_dict[name_of_enc_models[i]].fc3_enc)

        decrypted_model_dict[name_of_dec_models[i]].fc1_dec = fc1_dec_w
        decrypted_model_dict[name_of_dec_models[i]].fc2_dec = fc2_dec_w
        decrypted_model_dict[name_of_dec_models[i]].fc3_dec = fc3_dec_w

        decrypted_model_dict[name_of_dec_models[i]].fc1_dec_B1 = encrypted_model_dict[name_of_enc_models[i]].fc1_enc_B1
        decrypted_model_dict[name_of_dec_models[i]].fc2_dec_B2 = encrypted_model_dict[name_of_enc_models[i]].fc2_enc_B2
        decrypted_model_dict[name_of_dec_models[i]].fc3_dec_B3 = encrypted_model_dict[name_of_enc_models[i]].fc3_enc_B3

  return decrypted_model_dict

def Server_get_averaged_weights(matrix_dict, number_of_samples):
   
    fc1_mean_weight = torch.zeros(size=encrypted_model_dict[name_of_enc_models[0]].fc1_enc.shape)
    fc1_mean_bias = torch.zeros(size=encrypted_model_dict[name_of_enc_models[0]].fc1_enc_B1.shape)

    fc2_mean_weight = torch.zeros(size=encrypted_model_dict[name_of_enc_models[0]].fc2_enc.shape)
    fc2_mean_bias = torch.zeros(size=encrypted_model_dict[name_of_enc_models[0]].fc2_enc_B2.shape)

    fc3_mean_weight = torch.zeros(size=encrypted_model_dict[name_of_enc_models[0]].fc3_enc.shape)
    fc3_mean_bias = torch.zeros(size=encrypted_model_dict[name_of_enc_models[0]].fc3_enc_B3.shape)

    total_size_Enc_W_B = 0

    with torch.no_grad():
        for i in range(number_of_samples):

            fc1_mean_weight += matrix_dict['enc_model'+str(i)].fc1_enc
            total_size_Enc_W_B += sys.getsizeof(matrix_dict['enc_model'+str(i)].fc1_enc)

            fc2_mean_weight += matrix_dict['enc_model'+str(i)].fc2_enc
            total_size_Enc_W_B += sys.getsizeof(matrix_dict['enc_model'+str(i)].fc2_enc)

            fc3_mean_weight += matrix_dict['enc_model'+str(i)].fc3_enc 
            total_size_Enc_W_B += sys.getsizeof(matrix_dict['enc_model'+str(i)].fc3_enc)  

            fc1_mean_bias += matrix_dict['enc_model'+str(i)].fc1_enc_B1 
            total_size_Enc_W_B += sys.getsizeof(matrix_dict['enc_model'+str(i)].fc1_enc_B1)

            fc2_mean_bias += matrix_dict['enc_model'+str(i)].fc2_enc_B2 
            total_size_Enc_W_B += sys.getsizeof(matrix_dict['enc_model'+str(i)].fc2_enc_B2)

            fc3_mean_bias += matrix_dict['enc_model'+str(i)].fc3_enc_B3
            total_size_Enc_W_B += sys.getsizeof(matrix_dict['enc_model'+str(i)].fc3_enc_B3)
 
        
        plain_number_of_samples = ts.plain_tensor(1/number_of_samples)
        fc1_mean_weight1 =fc1_mean_weight * plain_number_of_samples
        fc1_mean_bias = fc1_mean_bias/ number_of_samples

        fc2_mean_weight2 =fc2_mean_weight * plain_number_of_samples
        fc2_mean_bias = fc2_mean_bias/ number_of_samples

        fc3_mean_weight3 =fc3_mean_weight * plain_number_of_samples    
        fc3_mean_bias = fc3_mean_bias/ number_of_samples


    return fc1_mean_weight1, fc1_mean_bias, fc2_mean_weight2, fc2_mean_bias, fc3_mean_weight3, fc3_mean_bias, total_size_Enc_W_B # , total_size

def update_dec_main_model(dec_main_model, decrypted_model_dict, number_of_samples):
    with torch.no_grad():
       
        dec_main_model.fc1_dec = decrypted_model_dict['dec_model0'].fc1_dec
        dec_main_model.fc2_dec = decrypted_model_dict['dec_model0'].fc2_dec
        dec_main_model.fc3_dec = decrypted_model_dict['dec_model0'].fc3_dec

        dec_main_model.fc1_dec_B1 = decrypted_model_dict[name_of_dec_models[0]].fc1_dec_B1
        dec_main_model.fc2_dec_B2 = decrypted_model_dict[name_of_dec_models[0]].fc2_dec_B2
        dec_main_model.fc3_dec_B3 = decrypted_model_dict[name_of_dec_models[0]].fc3_dec_B3

    return dec_main_model

def copy_dec_main_model_to_main_model(main_model, dec_main_model):
    with torch.no_grad():
       
        main_model.fc1.weight.data = torch.tensor(dec_main_model.fc1_dec).clone().detach() 
        main_model.fc2.weight.data = torch.tensor(dec_main_model.fc2_dec).clone().detach() 
        main_model.fc3.weight.data = torch.tensor(dec_main_model.fc3_dec).clone().detach() 

        main_model.fc1.bias.data = torch.tensor(dec_main_model.fc1_dec_B1).clone().detach() 
        main_model.fc2.bias.data = torch.tensor(dec_main_model.fc2_dec_B2).clone().detach() 
        main_model.fc3.bias.data = torch.tensor(dec_main_model.fc3_dec_B3).clone().detach() 

    return main_model

def set_averaged_Enc_weights_as_main_Enc_model_weights_and_update_main_Enc_model(enc_main_model, fc1_mean_weight, fc2_mean_weight, fc3_mean_weight, fc1_mean_bias, fc2_mean_bias, fc3_mean_bias):#, number_of_samples):
    with torch.no_grad():
       
        enc_main_model.fc1_enc = fc1_mean_weight
        enc_main_model.fc2_enc = fc2_mean_weight
        enc_main_model.fc3_enc = fc3_mean_weight

        enc_main_model.fc1_enc_B1 = fc1_mean_bias
        enc_main_model.fc2_enc_B2 = fc2_mean_bias
        enc_main_model.fc3_enc_B3 = fc3_mean_bias

    return enc_main_model

def send_main_model_to_nodes_and_update_model_dict_before_encryption(main_model, model_dict, number_of_samples):
    with torch.no_grad():
        for i in range(number_of_samples):

            model_dict[name_of_models[i]].fc1.weight.data = main_model.fc1.weight.data.clone()
            model_dict[name_of_models[i]].fc2.weight.data = main_model.fc2.weight.data.clone()
            model_dict[name_of_models[i]].fc3.weight.data = main_model.fc3.weight.data.clone() 
            
            model_dict[name_of_models[i]].fc1.bias.data = main_model.fc1.bias.data.clone()
            model_dict[name_of_models[i]].fc2.bias.data = main_model.fc2.bias.data.clone()
            model_dict[name_of_models[i]].fc3.bias.data = main_model.fc3.bias.data.clone() 
    
    return model_dict

def send_Enc_model_to_nodes_and_update_Enc_model_dict(enc_main_model, encrypted_model_dict, number_of_samples):
    with torch.no_grad():
        for i in range(number_of_samples):

            encrypted_model_dict[name_of_enc_models[i]].fc1_enc = enc_main_model.fc1_enc
            encrypted_model_dict[name_of_enc_models[i]].fc2_enc = enc_main_model.fc2_enc
            encrypted_model_dict[name_of_enc_models[i]].fc3_enc = enc_main_model.fc3_enc
            
            encrypted_model_dict[name_of_enc_models[i]].fc1_enc_B1 = enc_main_model.fc1_enc_B1
            encrypted_model_dict[name_of_enc_models[i]].fc2_enc_B2 = enc_main_model.fc2_enc_B2
            encrypted_model_dict[name_of_enc_models[i]].fc3_enc_B3 = enc_main_model.fc3_enc_B3
    
    return encrypted_model_dict

def prepare_data_dictionaries(x_train, y_train, number_of_samples):
    # Toplam veri sayısı
    total_samples = len(x_train)
    # Her node için veri sayısı
    samples_per_node = total_samples // number_of_samples
    
    x_train_dict = {}
    y_train_dict = {}
    
    for i in range(number_of_samples):
        start_idx = i * samples_per_node
        end_idx = (i + 1) * samples_per_node if i < number_of_samples - 1 else total_samples
        
        x_train_dict[f'x_train_{i}'] = x_train[start_idx:end_idx]
        y_train_dict[f'y_train_{i}'] = y_train[start_idx:end_idx]
    
    return x_train_dict, y_train_dict

def prepare_test_data_dictionaries(x_test, y_test, number_of_samples):
    total_samples = len(x_test)
    samples_per_node = total_samples // number_of_samples
    
    x_test_dict = {}
    y_test_dict = {}
    
    for i in range(number_of_samples):
        start_idx = i * samples_per_node
        end_idx = (i + 1) * samples_per_node if i < number_of_samples - 1 else total_samples
        
        x_test_dict[f'x_test_{i}'] = x_test[start_idx:end_idx]
        y_test_dict[f'y_test_{i}'] = y_test[start_idx:end_idx]
    
    return x_test_dict, y_test_dict


def start_train_end_node_process_without_print(number_of_samples, model_dict):
    global x_train, y_train, x_test, y_test
    global x_train_dict, y_train_dict, x_test_dict, y_test_dict
   
    if not model_dict:
        model_dict = initialize_models(number_of_samples)

    # Veri tiplerini dönüştür
    x_train = x_train.clone().detach().float()
    x_test = x_test.clone().detach().float()

    # Dictionary'leri hazırla
    if not hasattr(globals(), 'x_train_dict') or len(x_train_dict) == 0:
        x_train_dict, y_train_dict = prepare_data_dictionaries(x_train, y_train, number_of_samples)
    
    if not hasattr(globals(), 'x_test_dict') or len(x_test_dict) == 0:
        x_test_dict, y_test_dict = prepare_test_data_dictionaries(x_test, y_test, number_of_samples)
    
    for i in range(number_of_samples):
        # Model ve veri tiplerini kontrol et
        model = model_dict[f'model_{i}'].float()
        
        train_key = f'x_train_{i}'
        test_key = f'x_test_{i}'
        
        if train_key not in x_train_dict or test_key not in x_test_dict:
            print(f"Node {i} için veri bulunamadı!")
            continue
            
        # Dataset ve DataLoader oluştur
        train_ds = TensorDataset(x_train_dict[train_key], y_train_dict[f'y_train_{i}'])
        test_ds = TensorDataset(x_test_dict[test_key], y_test_dict[f'y_test_{i}'])
        
        batch_size = min(32, len(train_ds))
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        criterion = nn.CrossEntropyLoss()
        
        train_loss, train_accuracy, train_latency = train(model, train_dl, criterion, optimizer)
        
        model_dict[f'model_{i}'] = model

    return model_dict



x_train, y_train, x_valid, y_valid,x_test, y_test = map(torch.tensor, (x_train, y_train, x_valid, y_valid, x_test, y_test))
number_of_samples= 100
number_of_clusters = 5
learning_rate = 0.01
numEpoch =  30
batch_size = 32
momentum = 0.9

input_size = 784  # Giriş boyutunuzu ayarlayın
hidden_size = 128  # Gizli katman boyutunuzu ayarlayın
num_classes = 2   # Sınıf sayınızı ayarlayın

train_amount=6000
valid_amount=1000
test_amount=2000


train_ds = TensorDataset(x_train.float(), y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

valid_ds = TensorDataset(x_valid.float(), y_valid)
valid_dl = DataLoader(valid_ds, batch_size=batch_size * 2)

test_ds = TensorDataset(x_test.float(), y_test)
test_dl = DataLoader(test_ds, batch_size=batch_size * 2)


label_dict_train=dividing_and_shuffling_labels(y_label=y_train, seed=1, amount=train_amount) 
sample_dict_train=get_subsamples(label_dict=label_dict_train, number_of_samples=number_of_samples, amount=train_amount)
x_train_dict, y_train_dict = create_subsamples(sample_dict=sample_dict_train, x_data=x_train.float(), y_data=y_train, x_name="x_train", y_name="y_train")

label_dict_valid = dividing_and_shuffling_labels(y_label=y_valid, seed=1, amount=train_amount) 
sample_dict_valid = get_subsamples(label_dict=label_dict_valid, number_of_samples=number_of_samples, amount=valid_amount)
x_valid_dict, y_valid_dict = create_subsamples(sample_dict=sample_dict_valid, x_data=x_valid.float(), y_data=y_valid, x_name="x_valid", y_name="y_valid")

label_dict_test = dividing_and_shuffling_labels(y_label=y_test, seed=1, amount=test_amount) 
sample_dict_test = get_subsamples(label_dict=label_dict_test, number_of_samples=number_of_samples, amount=test_amount)
x_test_dict, y_test_dict = create_subsamples(sample_dict=sample_dict_test, x_data=x_test.float(), y_data=y_test, x_name="x_test", y_name="y_test")

main_model = Net2nn()
main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=0.9)
main_criterion = nn.CrossEntropyLoss()
enc_main_model = Net2nn()
dec_main_model = Net2nn()


model_dict, optimizer_dict, criterion_dict, encrypted_model_dict, decrypted_model_dict = create_model_optimizer_criterion_dict(number_of_samples)

name_of_x_train_sets=list(x_train_dict.keys())
name_of_y_train_sets=list(y_train_dict.keys())
name_of_x_valid_sets=list(x_valid_dict.keys())
name_of_y_valid_sets=list(y_valid_dict.keys())
name_of_x_test_sets=list(x_test_dict.keys())
name_of_y_test_sets=list(y_test_dict.keys())
name_of_models=list(model_dict.keys())
name_of_optimizers=list(optimizer_dict.keys())
name_of_criterions=list(criterion_dict.keys())
name_of_enc_models=list(encrypted_model_dict.keys())
name_of_dec_models=list(decrypted_model_dict.keys())

#################################### Original Approach ####################################

ExportToFile= 'Original_Output'+ "deneme"


Export=True
Flag=False

model_dict = initialize_models(number_of_samples)

name_of_models = [f"model{i}" for i in range(number_of_samples)]

for i in range(numEpoch):
    start_time = time.time()

    model_dict = send_main_model_to_nodes_and_update_model_dict_before_encryption(main_model, model_dict, number_of_samples)

    model_dict = start_train_end_node_process_without_print(number_of_samples, model_dict)

    encrypted_model_dict = enc_model(model_dict, encrypted_model_dict, number_of_samples) 
   
    fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias, total_size_Enc_W_B = Server_get_averaged_weights(encrypted_model_dict, number_of_samples)

    fc1_mean_weight_size = sys.getsizeof(fc1_mean_weight)
    fc2_mean_weight_size = sys.getsizeof(fc2_mean_weight)
    fc3_mean_weight_size = sys.getsizeof(fc3_mean_weight)
    fc1_mean_bias_size = sys.getsizeof(fc1_mean_bias)
    fc2_mean_bias_size = sys.getsizeof(fc2_mean_bias)
    fc3_mean_bias_size = sys.getsizeof(fc3_mean_bias)

    total_size_global_model = fc1_mean_weight_size + fc2_mean_weight_size + fc3_mean_weight_size + fc1_mean_bias_size + fc2_mean_bias_size + fc3_mean_bias_size

    enc_main_model = set_averaged_Enc_weights_as_main_Enc_model_weights_and_update_main_Enc_model(enc_main_model, fc1_mean_weight, fc2_mean_weight, fc3_mean_weight, fc1_mean_bias, fc2_mean_bias, fc3_mean_bias) #, number_of_samples) 

    encrypted_model_dict = send_Enc_model_to_nodes_and_update_Enc_model_dict(enc_main_model, encrypted_model_dict, number_of_samples)

    decrypted_model_dict = dec_model(encrypted_model_dict, decrypted_model_dict, number_of_samples)

    dec_main_model= update_dec_main_model(dec_main_model, decrypted_model_dict, number_of_samples) 

    main_model = copy_dec_main_model_to_main_model(main_model, dec_main_model)

    test_loss, test_accuracy, recall_all, precision_all, f1_score_all, TN_all, FP_all, FN_all, TP_all, len_test_loader = validation(main_model, test_dl, main_criterion)
    
    end_time = time.time()
    test_latency = end_time - start_time

    comm_overhead_B = total_size_Enc_W_B*number_of_samples*numEpoch
    comm_overhead_MB = comm_overhead_B/(1024*1024)
    
    comm_overhead_All_B = comm_overhead_B + total_size_global_model*number_of_samples*numEpoch

    comm_overhead = comm_overhead_All_B/(1024*1024)

    if(Export==True):
            with open(ExportToFile, 'a',newline='\n') as out:
                writer = csv.writer(out,delimiter=',')
                if (Flag==False):
                    header= np.concatenate([['Iteration', 'test accuracy' ,'test Recall' ,'test precision',
                            'test f1_score_all', 'test_loss', 'test latency', 'Communication_overhead', 'TN_all' , 'FP_all', 'FN_all', 'TP_all', 'len_test_loader']])
                    writer.writerow(header)
                a=np.concatenate([[i+1, test_accuracy, recall_all, precision_all, f1_score_all, test_loss, test_latency, comm_overhead , TN_all, FP_all, FN_all, TP_all, len_test_loader]])
                writer.writerow(a)
            out.close()
    Flag=True