
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd


# A  mix of code from the notebooks of the KDD CUP '99 Network Intrusion Kaggle competition and own implementation:

cols = """
    duration,
protocol_type,
service,
flag,
src_bytes,
dst_bytes,
land,
wrong_fragment,
urgent,
hot,
num_failed_logins,
logged_in,
num_compromised,
root_shell,
su_attempted,
num_root,
num_file_creations,
num_shells,
num_access_files,
num_outbound_cmds,
is_host_login,
is_guest_login,
count,
srv_count,
serror_rate,
srv_serror_rate,
rerror_rate,
srv_rerror_rate,
same_srv_rate,
diff_srv_rate,
srv_diff_host_rate,
dst_host_count,
dst_host_srv_count,
dst_host_same_srv_rate,
dst_host_diff_srv_rate,
dst_host_same_src_port_rate,
dst_host_srv_diff_host_rate,
dst_host_serror_rate,
dst_host_srv_serror_rate,
dst_host_rerror_rate,
dst_host_srv_rerror_rate"""
cols = [c.strip() for c in cols.split(",") if c.strip()]
cols.append('target')

attacks_type = {
'normal': 'normal',
'back': 'dos',
'buffer_overflow': 'u2r',
'ftp_write': 'r2l',
'guess_passwd': 'r2l',
'imap': 'r2l',
'ipsweep': 'probe',
'land': 'dos',
'loadmodule': 'u2r',
'multihop': 'r2l',
'neptune': 'dos',
'nmap': 'probe',
'perl': 'u2r',
'phf': 'r2l',
'pod': 'dos',
'portsweep': 'probe',
'rootkit': 'u2r',
'satan': 'probe',
'smurf': 'dos',
'spy': 'r2l',
'teardrop': 'dos',
'warezclient': 'r2l',
'warezmaster': 'r2l',
    }

df = pd.read_csv("kddcup.data_10_percent", names=cols)
df['Attack'] = df.target.apply(lambda r: attacks_type[r[:-1]])
print("The data shape is (lines, columns):",df.shape)

df.Attack.value_counts()

# Resampling:
from sklearn.utils import resample

df_dos = df[df.Attack == 'dos']
df_normal = df[df.Attack == 'normal']
df_probe = df[df.Attack == 'probe']
df_r2l = df[df.Attack == 'r2l']
df_u2r = df[df.Attack == 'u2r']

df_dos_upsampled = resample(df_dos,
                            replace=True,  # sample with replacement
                            n_samples=1000,  # to match majority class
                            random_state=123)  # reproducible results

df_normal_upsampled = resample(df_normal,
                               replace=True,  # sample with replacement
                               n_samples=1000,  # to match majority class
                               random_state=123)  # reproducible results

df_probe_upsampled = resample(df_probe,
                              replace=True,  # sample with replacement
                              n_samples=1000,  # to match majority class
                              random_state=123)  # reproducible results

df_r2l_upsampled = resample(df_r2l,
                            replace=True,  # sample with replacement
                            n_samples=1000,  # to match majority class
                            random_state=123)  # reproducible results

df_u2r_upsampled = resample(df_u2r,
                            replace=True,  # sample with replacement
                            n_samples=1000,  # to match majority class
                            random_state=123)  # reproducible results


df_upsampled = pd.concat(
    [df_dos_upsampled, df_normal_upsampled, df_probe_upsampled, df_r2l_upsampled, df_u2r_upsampled])

# Display new class counts
df_upsampled.Attack.value_counts()

# Standarize columns
df_std = df_upsampled.std()
df_std = df_std.sort_values(ascending=True)

hajar_to_cup = {
    'is_hot_login' : 'is_host_login',
'urg' : 'urgent',
'protocol' : 'protocol_type',
'count_sec' : 'count',
'srv_count_sec' : 'srv_count',
'serror_rate_sec' : 'serror_rate',
'srv_serror_rate_sec' : 'srv_serror_rate',
'rerror_rate_sec' : 'rerror_rate',
'srv_error_rate_sec' : 'srv_rerror_rate',
'same_srv_rate_sec' : 'same_srv_rate',
'diff_srv_rate_sec' : 'diff_srv_rate',
'srv_diff_host_rate_sec' : 'srv_diff_host_rate',
'count_100' : 'dst_host_count',
'srv_count_100' : 'dst_host_srv_count',
'same_srv_rate_100' : 'dst_host_same_srv_rate',
'diff_srv_rate_100' : 'dst_host_diff_srv_rate',
'same_src_port_rate_100' : 'dst_host_same_src_port_rate',
'srv_diff_host_rate_100' : 'dst_host_srv_diff_host_rate',
'serror_rate_100' : 'dst_host_serror_rate',
'srv_serror_rate_100' : 'dst_host_srv_serror_rate',
'rerror_rate_100' : 'dst_host_rerror_rate',
'srv_rerror_rate_100' : 'dst_host_srv_rerror_rate',
}
def standardize_columns(df, cols_map=hajar_to_cup):
    if 'service' in df.columns:
        df = df.drop(['service'], axis = 1)
    df.rename(columns = cols_map)
    return df

df = standardize_columns(df_upsampled, cols_map=hajar_to_cup)
df = df.drop(['target',], axis=1)

# Encode string columns
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
df[['dst_bytes','src_bytes']] = scaler.fit_transform(df[['dst_bytes','src_bytes']])


labelencoder = LabelEncoder()
df['protocol_type'] = labelencoder.fit_transform(df['protocol_type'])
df['flag'] = labelencoder.fit_transform(df['flag'])
df['Attack'] = labelencoder.fit_transform(df['Attack'])

df.to_csv('./csv_files/KDD.csv')

