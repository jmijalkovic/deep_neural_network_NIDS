import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn import preprocessing

import preprocessing_train_dataset_NSLKDD

warnings.filterwarnings("ignore")

# LOADING THE DATASET
col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
             "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"]

path2 = 'C:\\Users\\Jovana\\PycharmProjects\\neural_network_DNN\\include\\KDDTest+.txt'

test_dataset = pd.read_csv(path2, header=None, names=col_names)

# =============================DATA PREPROCESSING==============================================

# checking for nan values in each column
for i in test_dataset.columns:
    print(test_dataset[[i]].isnull().sum())
# we can see we do not have any missing values

# shape of the dataset
print(test_dataset.shape)

# descriptive statistics of dataset
print(test_dataset.describe())

# number of attack labels
print(test_dataset['label'].value_counts())


# changing attack labels to their respective attack class
def change_label(df):
    df.label.replace(
        ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm',
         'worm'], 'Dos', inplace=True)
    df.label.replace(['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail',
                      'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xlock', 'xsnoop'], 'R2L',
                     inplace=True)
    df.label.replace(['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan'], 'Probe', inplace=True)
    df.label.replace(['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm'], 'U2R',
                     inplace=True)


# calling change_label() function
change_label(test_dataset)

# distribution of attack classes
print(test_dataset.label.value_counts())

### DATA NORMALIZATION ###

# selecting numeric attributes columns from data
numeric_col = test_dataset.select_dtypes(include='number').columns

# using MinMax scaler for normalizing
minmax_scaler = preprocessing.MinMaxScaler()


def normalization(df, col):
    for i in col:
        arr = df[i]
        arr = np.array(arr)
        df[i] = minmax_scaler.fit_transform(arr.reshape(len(arr), 1))
    return df


# data before normalization
print(test_dataset.head())

# calling the normalization() function
test_dataset_norm = normalization(test_dataset.copy(), numeric_col)

# data after normalization
print(test_dataset_norm.head())

### ONE-HOT-ENCODING ###

# selecting categorical data attributes
cat_col = ['protocol_type', 'service', 'flag']

# creating a dataframe with only categorical attributes
categorical = test_dataset_norm[cat_col]
print(categorical.head())

# one-hot-encoding categorical attributes using pandas.get_dummies() function
categorical = pd.get_dummies(categorical, columns=cat_col)
print(categorical.head())

# some columns are missing, we have to add them with zeroes
trainservice = preprocessing_train_dataset_NSLKDD.train_dataset['service'].tolist()
testservice = test_dataset['service'].tolist()
difference = list(set(trainservice) - set(testservice))
string = 'service_'
difference = [string + x for x in difference]
print(difference)

for col in difference:
    categorical[col] = 0

print(categorical.head())

### FOR BINARY CLASSIFICATION ###

# changing attack labels into two categories 'normal' and 'attack'
bin_label = pd.DataFrame(test_dataset_norm.label.map(lambda x: 'normal' if x == 'normal' else 'attack'))

# creating a dataframe with binary labels (normal,attack)
bin_data = test_dataset_norm.copy()
bin_data['label'] = bin_label

# label encoding (0,1) binary labels (attack,normal)
le1 = preprocessing.LabelEncoder()
enc_label = bin_label.apply(le1.fit_transform)
bin_data['intrusion'] = enc_label

print(le1.classes_)

# dataset with binary labels and label encoded column
print(bin_data.head())

# one-hot-encoding attack label
bin_data = pd.get_dummies(bin_data, columns=['label'], prefix="", prefix_sep="")
bin_data['label'] = bin_label
print(bin_data)

# # pie chart distribution of normal and abnormal labels
# plt.figure(figsize=(8, 8))
# plt.pie(bin_data.label.value_counts(), labels=bin_data.label.unique(), autopct='%0.2f%%')
# plt.title("Pie chart distribution of normal and abnormal labels")
# plt.legend()
# plt.savefig('C:\\Users\\Jovana\\PycharmProjects\\neural_network_DNN\\plots\\Pie_chart_binary_test.png')
# plt.show()

### MULTICLASS CLASSIFICATION ###

# creating a dataframe with multi-class labels (Dos,Probe,R2L,U2R,normal)
multi_data = test_dataset_norm.copy()
multi_label = pd.DataFrame(multi_data.label)

# label encoding (0,1,2,3,4) multi-class labels (Dos,normal,Probe,R2L,U2R)
le2 = preprocessing.LabelEncoder()
enc_label = multi_label.apply(le2.fit_transform)
multi_data['intrusion'] = enc_label

print(le2.classes_)

# one-hot-encoding attack label
multi_data = pd.get_dummies(multi_data, columns=['label'], prefix="", prefix_sep="")
multi_data['label'] = multi_label
print(multi_data.head())

# # pie chart distribution of multi-class labels
# plt.figure(figsize=(8, 8))
# plt.pie(multi_data.label.value_counts(), labels=multi_data.label.unique(), autopct='%0.2f%%')
# plt.title('Pie chart distribution of multi-class labels')
# plt.legend()
# plt.savefig('C:\\Users\\Jovana\\PycharmProjects\\neural_network_DNN\\plots\\Pie_chart_multi_test.png')
# plt.show()

# MAKING THE DATASETS WITH ONE HOT ENCODED ATTRS AND NORMALIZED #

# BINARY #
numeric_bin = bin_data[numeric_col]
numeric_bin['intrusion'] = bin_data['intrusion']
numeric_bin = numeric_bin.join(categorical)
# then joining encoded, one-hot-encoded, and original attack label attribute
bin_data = numeric_bin.join(bin_data[['attack','normal','label']])
print(bin_data)