
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
load = pd.read_csv("C:/Users/rishabh.malhotra/Desktop/Anomaly Detection - IoT/DataCEP_A.csv")


# In[3]:


load.isnull().sum()


# In[4]:


load=load.dropna()
load.set_index('DATETIME',drop=True,inplace=True)
load=load.reset_index()


# ## Load and Load_final and Load_final1 copies

# In[5]:


load_final=load
load_final1=load_final


# ## Load_final1-Date_Time as a dataframe for datetime column

# In[6]:


Date_Time=pd.DataFrame(load_final1['DATETIME'])
Date_Time.shape


# ## Importing packages necessary for K-means

# In[7]:


import matplotlib
import seaborn
import matplotlib.dates as md
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ## Standardised Data as load_final2

# In[8]:


load_final1.columns


# In[9]:


load_final2 = load_final1[ ['COND_PMP_BRG_HORI_VIB', 'COND_PMP_BRG_VERT_VIB',
       'COND_PMP_DE_JRNL_BRG_TEMP', 'COND_PMP_MOT_STR__PH_TEMP',
       'COND_PMP_MOT_STR__PH_B_TEMP', 'COND_PMP_MOT_STR__PH_C_TEMP',
       'COND_PMP_MOTOR_CURRENT', 'COND_PMP_NDE_JRNL_BRG_TEMP',
       'COND_PMP_THR_BRG_MTL_TEMP', 'COND_PMP_OUTLET_PR',
       'COND_PMP_OUTLET_TEMP', 'COND_WATER_FLOW_1', 'COND_WATER_FLOW_2',
       'COND_WATER_PR', 'COND_WATER_RECIRC_FLOW', 'CONDENSATE_PMP_INLET_PR',
       'CONDENSATE_PMP_INLET_TEMP', 'CONDENSER_VACUUM',
       'CONDENSR_HOTWELL_WATER_LVL_1', 'CONDENSR_HOTWELL_WATER_LVL_2',
       'CONDENSR_HOTWELL_WATER_LVL_3']]
load_final3=load_final2
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(load_final2)
load_final2 = pd.DataFrame(np_scaled)


# ## Standardised Data load_final2 renaming columns

# In[10]:


load_final2.columns=[ ['COND_PMP_BRG_HORI_VIB', 'COND_PMP_BRG_VERT_VIB',
       'COND_PMP_DE_JRNL_BRG_TEMP', 'COND_PMP_MOT_STR__PH_TEMP',
       'COND_PMP_MOT_STR__PH_B_TEMP', 'COND_PMP_MOT_STR__PH_C_TEMP',
       'COND_PMP_MOTOR_CURRENT', 'COND_PMP_NDE_JRNL_BRG_TEMP',
       'COND_PMP_THR_BRG_MTL_TEMP', 'COND_PMP_OUTLET_PR',
       'COND_PMP_OUTLET_TEMP', 'COND_WATER_FLOW_1', 'COND_WATER_FLOW_2',
       'COND_WATER_PR', 'COND_WATER_RECIRC_FLOW', 'CONDENSATE_PMP_INLET_PR',
       'CONDENSATE_PMP_INLET_TEMP', 'CONDENSER_VACUUM',
       'CONDENSR_HOTWELL_WATER_LVL_1', 'CONDENSR_HOTWELL_WATER_LVL_2',
       'CONDENSR_HOTWELL_WATER_LVL_3']]
load_final2.head()


# ## Checking load_final2 and load_final3 shape

# In[11]:


if load_final2.shape==load_final3.shape:
    print('Oh, Yeah')
else:
    pass


# ## Resetting index to Datetime

# In[12]:


Date_Time=Date_Time.reset_index(drop=True)
load_final3=pd.concat([Date_Time,load_final3],axis=1)
load_final2=pd.concat([Date_Time,load_final2],axis=1)


# In[13]:


load_final3.head()


# ## Now we have Load_final2 and Load_final3 with datetime columns and standardised

# ## LOF- Local Outlier Fraction1

# ## PCA

# In[14]:


from sklearn.decomposition import PCA
load_final2=load_final2.set_index('DATETIME')
pca = PCA(n_components=2,random_state=5)
pca.fit(load_final2)
x_pca = pca.transform(load_final2)
load_final_pca=pd.DataFrame(x_pca,columns=['PCA1','PCA2'])


# In[15]:


load_final_pca.head()


# ## LOF

# #### AS MATRIX

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor


# In[17]:


# fit the model
clf = LocalOutlierFactor(n_neighbors=50,contamination=0.05)
y_pred = clf.fit_predict(load_final_pca)
print(y_pred)
y_pred_outliers = y_pred[:]


# In[18]:


y_pred_out=pd.DataFrame(y_pred_outliers,columns=['ana'] )
Load_final_pca=pd.concat([load_final_pca,y_pred_out],axis=1)
y_pred_out['ana'].value_counts()


# ## Creating dataset for just load_final2 without 10% outliers

# In[19]:


LOF_Anomaly_flag=Load_final_pca['ana']


# In[20]:


LOF_Anomalies=Load_final_pca[Load_final_pca['ana']==-1]
LOF_Anomalies.shape


# In[21]:


LOF_Anomaly_index=list(LOF_Anomalies.index)

#load_final3 LOF data
LOF_Anomaly_3=load_final3.iloc[LOF_Anomaly_index]

#load_final2 LOF data
LOF_Anomaly_2=load_final2.iloc[LOF_Anomaly_index]

Load_non_anomaly=Load_final_pca[Load_final_pca['ana']==1]
Load_non_anomaly=Load_non_anomaly.drop('ana',axis=1)
Index_out=list(Load_non_anomaly.index)
load_final2=load_final2.iloc[Index_out]
load_final3=load_final3.iloc[Index_out]
load_final2.reset_index(drop=True)
load_final3.reset_index(drop=True)
load_final3=load_final3.set_index('DATETIME')


# ## NOW OUR load_final3 and Load_final2 are without LOF outliers

# In[22]:


load_final_pca_without_out=Load_non_anomaly
load_final_pca_without_out=load_final_pca_without_out.reset_index(drop=True)


# # KMeans without LOF

# In[23]:


#KMeans without LOF
n_cluster = range(1, 12)
kmeans = [KMeans(n_clusters=i,random_state=0).fit(load_final_pca) for i in n_cluster]
scores = [kmeans[i].score(load_final_pca) for i in range(len(kmeans))]
fig, ax = plt.subplots()
ax.plot(n_cluster, scores)
plt.show()

df1=pd.DataFrame()
df1['Cluster'] = kmeans[10].predict(load_final_pca)
Cluster_label1=pd.DataFrame(df1['Cluster'])
print(load_final_pca.shape)
print(Cluster_label1.shape)

df1['Cluster']
print(df1['Cluster'].value_counts())

df1['principal_feature1'] = load_final_pca['PCA1']
df1['principal_feature2'] = load_final_pca['PCA2']

fig, ax = plt.subplots()
colors = {0:'red', 1:'blue', 2:'green', 3:'pink', 4:'black', 5:'orange', 6:'cyan', 7:'yellow', 8:'brown', 9:'purple', 10:'magenta', 11: 'grey', 12:'lightblue', 13:'lightgreen', 14: 'darkgrey',15: 'darkgreen'}
ax.scatter(df1['principal_feature1'], df1['principal_feature2'], c=df1['Cluster'].apply(lambda x: colors[x]))
plt.show()


# # DBSCAN Without LOF

# In[24]:


from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=40)
db.fit(load_final_pca)

Labels=pd.DataFrame(db.fit_predict(load_final_pca, y=None, sample_weight=None))
Labels.columns=['DB_Cluster']
print(Labels['DB_Cluster'].value_counts())
df=pd.DataFrame()
df['principal_feature1'] = load_final_pca['PCA1']
df['principal_feature2'] = load_final_pca['PCA2']

fig, ax = plt.subplots()
colors = {-1:'darkgrey',0:'red', 1:'blue', 2:'green', 3:'pink', 4:'black', 5:'orange', 6:'cyan', 7:'yellow', 8:'brown', 9:'purple', 10:'magenta', 11: 'violet', 12:'lightblue', 13:'lightgreen', 14: 'salmon',15: 'darkgreen'}
ax.scatter(df['principal_feature1'], df['principal_feature2'], c=Labels['DB_Cluster'].apply(lambda x: colors[x]))
plt.show()


# # Kmeans with LOF

# In[25]:


n_cluster = range(1, 18)
kmeans = [KMeans(n_clusters=i,random_state=0).fit(load_final_pca_without_out) for i in n_cluster]
scores = [kmeans[i].score(load_final_pca_without_out) for i in range(len(kmeans))]
fig, ax = plt.subplots()
ax.plot(n_cluster, scores)
plt.show()

df=pd.DataFrame()
df['Cluster'] = kmeans[16].predict(load_final_pca_without_out)
Cluster_label=pd.DataFrame(df['Cluster'])
print(load_final_pca_without_out.shape)
print(Cluster_label.shape)

df['Cluster']
print(df['Cluster'].value_counts())

df['principal_feature1'] = load_final_pca_without_out['PCA1']
df['principal_feature2'] = load_final_pca_without_out['PCA2']

fig, ax = plt.subplots()
colors = {0:'red', 1:'blue', 2:'green', 3:'pink', 4:'black', 5:'orange', 6:'cyan', 7:'yellow', 8:'brown', 9:'purple', 10:'magenta', 11: 'grey', 12:'lightblue', 13:'lightgreen', 14: 'darkgrey',15: 'darkgreen',16: 'violet',17:'blanchedalmond',18:'indigo'}
ax.scatter(df['principal_feature1'], df['principal_feature2'], c=df['Cluster'].apply(lambda x: colors[x]))
plt.show()


# In[26]:


Cluster_Label=pd.DataFrame(df['Cluster'])


# # DBSCAN WITH LOF

# In[27]:


from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.4, min_samples=60)
db.fit(load_final_pca_without_out)

Labels=pd.DataFrame(db.fit_predict(load_final_pca_without_out, y=None, sample_weight=None))
Labels.columns=['DB_Cluster']
print(Labels['DB_Cluster'].value_counts())
df1=pd.DataFrame()
df1['principal_feature1'] = load_final_pca_without_out['PCA1']
df1['principal_feature2'] = load_final_pca_without_out['PCA2']

fig, ax = plt.subplots()
colors = {-1:'darkgrey',0:'red', 1:'blue', 2:'green', 3:'pink', 4:'black', 5:'orange', 6:'cyan', 7:'yellow', 8:'brown', 9:'purple', 10:'magenta', 11: 'violet', 12:'lightblue', 13:'lightgreen', 14: 'salmon',15: 'darkgreen',16: 'violet',17:'blanchedalmond',18:'indigo'}
ax.scatter(df1['principal_feature1'], df1['principal_feature2'], c=Labels['DB_Cluster'].apply(lambda x: colors[x]))
plt.show()


# In[28]:


DBScan_Anomaly_Flag=pd.DataFrame(Labels['DB_Cluster'])


# # Load_final_pca_without_out definition

# In[29]:


print(DBScan_Anomaly_Flag.shape)
print(Cluster_Label.shape)
print(load_final_pca_without_out.shape)


# In[30]:


load_final_pca_without_out=pd.concat([DBScan_Anomaly_Flag,Cluster_Label,load_final_pca_without_out],axis=1)


# In[31]:


load_final_pca_without_out


# ## Cluster Anomalies through DBScan

# In[32]:


for j in range(0,17):
    k=0
    
    for i in range(0,len(load_final_pca_without_out)):
        
    
        if (load_final_pca_without_out['Cluster'].iloc[i]== j) & (load_final_pca_without_out['DB_Cluster'].iloc[i]==-1):
            k=k+1
    print(j)                   
    print (k/(sum(load_final_pca_without_out['Cluster']==j)))


# # From above data we raise Cluster 2, Cluster 10, Cluster 11,Cluster 13 and Cluster 15 as Cluster_Anomalies

# In[33]:


load_final_pca_without_out['Cluster_anomaly_flag']=0


# In[34]:


for i in range(0,len(load_final_pca_without_out)):
    if ((load_final_pca_without_out['Cluster'].iloc[i]==2) | (load_final_pca_without_out['Cluster'].iloc[i]==10) | (load_final_pca_without_out['Cluster'].iloc[i]==11) | (load_final_pca_without_out['Cluster'].iloc[i]==13)  | (load_final_pca_without_out['Cluster'].iloc[i]==15)):
        load_final_pca_without_out['Cluster_anomaly_flag'].iloc[i]=1
    else:
        pass


# ## Creating Final Datetime Columns

# In[35]:


load_final3.reset_index(inplace=True)
Date_Time_final=load_final3['DATETIME']
load_final_pca_without_out=pd.concat([Date_Time_final,load_final_pca_without_out],axis=1)
load_final_pca_without_out.shape


# ## Setting index of load_final3 again

# In[36]:


load_final3.set_index('DATETIME',inplace=True)


# ## Resetting index of load_final2 and load_final3

# In[37]:


print(load_final2.shape)
print(load_final3.shape)


# In[38]:


load_final2=load_final2.reset_index()
load_final3=load_final3.reset_index()


# ## Datetime is concated in load_final_pca_without_out and named load_final_pca_without_out_DT

# In[39]:


load_final_pca_without_out_DT=load_final_pca_without_out


# # Finding Switch Anomalies

# In[40]:


switch_anomaly =np.zeros(len(Cluster_label))
cluster_anomaly =np.zeros(len(Cluster_label))
for i in range(0,len(Cluster_label)):
    if((load_final_pca_without_out_DT['Cluster'].iloc[i]==2) | (load_final_pca_without_out_DT['Cluster'].iloc[i]==10) | (load_final_pca_without_out_DT['Cluster'].iloc[i]==11) |(load_final_pca_without_out_DT['Cluster'].iloc[i]==13) |(load_final_pca_without_out_DT['Cluster'].iloc[i]==15)):
        if((load_final_pca_without_out_DT['Cluster'].iloc[i-1]==2) | (load_final_pca_without_out_DT['Cluster'].iloc[i-1]==10) | (load_final_pca_without_out_DT['Cluster'].iloc[i-1]==11) | (load_final_pca_without_out_DT['Cluster'].iloc[i-1]==13)| (load_final_pca_without_out_DT['Cluster'].iloc[i-1]==15)):
            cluster_anomaly[i]=1
            print(cluster_anomaly[i])
        else:
            switch_anomaly[i]=1
            print(switch_anomaly[i])
        
Switch_Anomaly_flag=pd.DataFrame(switch_anomaly)
Cluster_Anomaly_flag=pd.DataFrame(cluster_anomaly)
Switch_Anomaly_flag.columns=['Switch_Anomaly']
Cluster_Anomaly_flag.columns=['Clus_Anomaly']


# # TOTAL SWITCH ANOMALIES

# In[41]:


Switch_Anomaly_flag['Switch_Anomaly'].sum()


# # TOTAL CLUSTER ANOMALIES

# In[42]:


Cluster_Anomaly_flag['Clus_Anomaly'].sum()


# # Defining Load_Load with Switch Anomaly Flag and Cluster Anomaly Flag

# In[43]:


Load_Load = pd.concat([Switch_Anomaly_flag,Cluster_Anomaly_flag,load_final_pca_without_out_DT],axis=1)


# ## Switch_Anomalies

# In[44]:


load_final3=load_final3.set_index('DATETIME')
Switch_Anomalies_pca=Load_Load[Load_Load['Switch_Anomaly']==1]
Switch_Anomalies_pca.set_index('DATETIME',inplace=True)
Switch_Anomalies=load_final3.loc[list(Switch_Anomalies_pca.index)]


# ## Cluster Anomalies

# In[45]:


Cluster_Anomalies_pca=Load_Load[Load_Load['Clus_Anomaly']==1]
Cluster_Anomalies_pca.set_index('DATETIME',inplace=True)
Cluster_Anomalies=load_final3.loc[list(Cluster_Anomalies_pca.index)]


# In[46]:


Cluster_0=Load_Load[Load_Load['Cluster']==0]
Cluster_1=Load_Load[Load_Load['Cluster']==1]
Cluster_2=Load_Load[Load_Load['Cluster']==2]
Cluster_3=Load_Load[Load_Load['Cluster']==3]
Cluster_4=Load_Load[Load_Load['Cluster']==4]
Cluster_5=Load_Load[Load_Load['Cluster']==5]
Cluster_6=Load_Load[Load_Load['Cluster']==6]
Cluster_7=Load_Load[Load_Load['Cluster']==7]
Cluster_8=Load_Load[Load_Load['Cluster']==8]
Cluster_9=Load_Load[Load_Load['Cluster']==9]
Cluster_10=Load_Load[Load_Load['Cluster']==10]
Cluster_11=Load_Load[Load_Load['Cluster']==11]
Cluster_12=Load_Load[Load_Load['Cluster']==12]
Cluster_13=Load_Load[Load_Load['Cluster']==13]
Cluster_14=Load_Load[Load_Load['Cluster']==14]
Cluster_15=Load_Load[Load_Load['Cluster']==15]
Cluster_16=Load_Load[Load_Load['Cluster']==16]
Cluster_17=Load_Load[Load_Load['Cluster']==17]


# In[47]:


load_final3.reset_index(inplace=True)


# ## Clustersas load_final3

# In[48]:


lf3_c0=load_final3.loc[list(Cluster_0.index)]
lf3_c1=load_final3.loc[list(Cluster_1.index)]
lf3_c2=load_final3.loc[list(Cluster_2.index)]
lf3_c3=load_final3.loc[list(Cluster_3.index)]
lf3_c4=load_final3.loc[list(Cluster_4.index)]
lf3_c5=load_final3.loc[list(Cluster_5.index)]
lf3_c6=load_final3.loc[list(Cluster_6.index)]
lf3_c7=load_final3.loc[list(Cluster_7.index)]
lf3_c8=load_final3.loc[list(Cluster_8.index)]
lf3_c9=load_final3.loc[list(Cluster_9.index)]
lf3_c10=load_final3.loc[list(Cluster_10.index)]
lf3_c11=load_final3.loc[list(Cluster_11.index)]
lf3_c12=load_final3.loc[list(Cluster_12.index)]
lf3_c13=load_final3.loc[list(Cluster_13.index)]
lf3_c14=load_final3.loc[list(Cluster_14.index)]
lf3_c15=load_final3.loc[list(Cluster_15.index)]
lf3_c16=load_final3.loc[list(Cluster_16.index)]
lf3_c17=load_final3.loc[list(Cluster_17.index)]


# In[49]:


load_final3.set_index('DATETIME',inplace=True)
lf3_c0.set_index('DATETIME',inplace=True)
lf3_c1.set_index('DATETIME',inplace=True)
lf3_c2.set_index('DATETIME',inplace=True)
lf3_c3.set_index('DATETIME',inplace=True)
lf3_c4.set_index('DATETIME',inplace=True)
lf3_c5.set_index('DATETIME',inplace=True)
lf3_c6.set_index('DATETIME',inplace=True)
lf3_c7.set_index('DATETIME',inplace=True)
lf3_c8.set_index('DATETIME',inplace=True)
lf3_c9.set_index('DATETIME',inplace=True)
lf3_c10.set_index('DATETIME',inplace=True)
lf3_c11.set_index('DATETIME',inplace=True)
lf3_c12.set_index('DATETIME',inplace=True)
lf3_c13.set_index('DATETIME',inplace=True)
lf3_c14.set_index('DATETIME',inplace=True)
lf3_c15.set_index('DATETIME',inplace=True)
lf3_c16.set_index('DATETIME',inplace=True)
lf3_c17.set_index('DATETIME',inplace=True)


# ## LOF Anomaly dataframe as load_final3

# In[50]:


load_final3=load_final3.reset_index()
lf3_lof=load_final3.loc[list(LOF_Anomalies.index)]
load_final3=load_final3.set_index('DATETIME')
lf3_lof=lf3_lof.set_index('DATETIME')


# In[51]:


print(load_final3.shape)
print(lf3_lof.shape)
print(lf3_c0.shape)
print(lf3_c1.shape)
print(lf3_c2.shape)
print(Switch_Anomalies.shape)
print(Cluster_Anomalies.shape)


# ## Now to find Factor Anomalies

# In[52]:


Mean=[]
Std=[]
for i in range(0,len(load_final3.columns)):
    Mean.append(load_final3[load_final3.columns[i]].mean())   
    Std.append(load_final3[load_final3.columns[i]].std())


# ## Defining Total Anomaly Dataframe

# In[53]:


Total_Anomalies=pd.concat([lf3_lof,Switch_Anomalies,Cluster_Anomalies],axis=0)


# In[54]:


Total_Anomalies.shape


# ## Sensorwise_Anomaly

# In[55]:


Anomaly_array=np.zeros((len(Total_Anomalies),len(Total_Anomalies.columns)))
Sensorwise_Anomaly=pd.DataFrame(Anomaly_array,columns=list(Total_Anomalies.columns))

for i in range(0,len(Total_Anomalies.columns)):
    for j in range(0,len(Total_Anomalies)):
        if Mean[i]-2*Std[i]<Total_Anomalies.iloc[j][i]<Mean[i]+2*Std[i]:
            pass
            
        else:
            Sensorwise_Anomaly.iloc[j][i]=1


# ## Columns-wise Anomaly flags

# In[56]:


for i in range(0,len(Sensorwise_Anomaly.columns)):
    print(Sensorwise_Anomaly[Sensorwise_Anomaly.columns[i]].sum())


# # GRAPHS

# In[57]:


load_final3.reset_index(inplace=True)


# In[58]:


load_final_graphs=load_final3


# In[59]:


print(Cluster_Anomaly_flag.shape)
print(Switch_Anomaly_flag.shape)


# In[60]:


load_final_graphs=pd.concat([load_final_graphs,Cluster_Anomaly_flag,Switch_Anomaly_flag],axis=1)


# ## Visualisation of Cluster Anonmalies on TS graph

# In[61]:


load_final_graphs.reset_index(inplace=True)


# In[62]:


fig, ax= plt.subplots()

a=load_final_graphs.loc[load_final_graphs['Clus_Anomaly'] ==1, ['index','COND_PMP_BRG_HORI_VIB']]

ax.plot(load_final_graphs['index'],load_final_graphs['COND_PMP_BRG_HORI_VIB'])
ax.scatter(a['index'],a['COND_PMP_BRG_HORI_VIB'], color='red')


# In[63]:


fig, ax= plt.subplots()

a=load_final_graphs.loc[load_final_graphs['Clus_Anomaly'] ==1, ['index','COND_PMP_BRG_VERT_VIB']]

ax.plot(load_final_graphs['index'],load_final_graphs['COND_PMP_BRG_VERT_VIB'])
ax.scatter(a['index'],a['COND_PMP_BRG_VERT_VIB'], color='red')


# In[64]:


fig, ax= plt.subplots()

a=load_final_graphs.loc[load_final_graphs['Clus_Anomaly'] ==1, ['index','CONDENSATE_PMP_INLET_PR']]

ax.plot(load_final_graphs['index'],load_final_graphs['CONDENSATE_PMP_INLET_PR'])
ax.scatter(a['index'],a['CONDENSATE_PMP_INLET_PR'], color='red')


# In[65]:


fig, ax= plt.subplots()

a=load_final_graphs.loc[load_final_graphs['Clus_Anomaly'] ==1, ['index','CONDENSATE_PMP_INLET_TEMP']]

ax.plot(load_final_graphs['index'],load_final_graphs['CONDENSATE_PMP_INLET_TEMP'])
ax.scatter(a['index'],a['CONDENSATE_PMP_INLET_TEMP'], color='red')


# ## Switch Anomalies

# In[66]:


fig, ax= plt.subplots()

a=load_final_graphs.loc[load_final_graphs['Switch_Anomaly'] ==1, ['index','CONDENSATE_PMP_INLET_PR']]

ax.plot(load_final_graphs['index'],load_final_graphs['CONDENSATE_PMP_INLET_PR'])
ax.scatter(a['index'],a['CONDENSATE_PMP_INLET_PR'], color='red')


# In[67]:


fig, ax= plt.subplots()

a=load_final_graphs.loc[load_final_graphs['Switch_Anomaly'] ==1, ['index','CONDENSATE_PMP_INLET_TEMP']]

ax.plot(load_final_graphs['index'],load_final_graphs['CONDENSATE_PMP_INLET_TEMP'])
ax.scatter(a['index'],a['CONDENSATE_PMP_INLET_TEMP'], color='red')


# In[68]:


fig, ax= plt.subplots()

a=load_final_graphs.loc[load_final_graphs['Switch_Anomaly'] ==1, ['index','COND_PMP_BRG_VERT_VIB']]

ax.plot(load_final_graphs['index'],load_final_graphs['COND_PMP_BRG_VERT_VIB'])
ax.scatter(a['index'],a['COND_PMP_BRG_VERT_VIB'], color='red')


# In[69]:


Total_Anomalies.to_csv('Crunch-a-thon New data.csv')


# In[70]:


Total_Anomalies.reset_index(inplace=True)


# # Now your data is trained

# ### For any new data 

# ### new= pd.read_csv(..........)
# new data pca
# new data clustering(predict)
# concat axis 0

# In[82]:


new_load = pd.read_csv("C:/Users/rishabh.malhotra/Desktop/Anomaly Detection - IoT/DataCEP_A.csv")


# In[84]:


new_load=new_load[0:1000]


# In[85]:


new_load.isnull().sum()

new_load=new_load.dropna()
new_load.set_index('DATETIME',drop=True,inplace=True)
new_load=new_load.reset_index()

## Load and Load_final and Load_final1 copies

new_load_final=load
new_load_final1=new_load_final

## Load_final1-Date_Time as a dataframe for datetime column

Date_Time=pd.DataFrame(new_load_final1['DATETIME'])
Date_Time.shape

## Importing packages necessary for K-means

import matplotlib
import seaborn
import matplotlib.dates as md
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

## Standardised Data as load_final2

new_load_final1.columns

new_load_final2 = new_load_final1[ ['COND_PMP_BRG_HORI_VIB', 'COND_PMP_BRG_VERT_VIB',
       'COND_PMP_DE_JRNL_BRG_TEMP', 'COND_PMP_MOT_STR__PH_TEMP',
       'COND_PMP_MOT_STR__PH_B_TEMP', 'COND_PMP_MOT_STR__PH_C_TEMP',
       'COND_PMP_MOTOR_CURRENT', 'COND_PMP_NDE_JRNL_BRG_TEMP',
       'COND_PMP_THR_BRG_MTL_TEMP', 'COND_PMP_OUTLET_PR',
       'COND_PMP_OUTLET_TEMP', 'COND_WATER_FLOW_1', 'COND_WATER_FLOW_2',
       'COND_WATER_PR', 'COND_WATER_RECIRC_FLOW', 'CONDENSATE_PMP_INLET_PR',
       'CONDENSATE_PMP_INLET_TEMP', 'CONDENSER_VACUUM',
       'CONDENSR_HOTWELL_WATER_LVL_1', 'CONDENSR_HOTWELL_WATER_LVL_2',
       'CONDENSR_HOTWELL_WATER_LVL_3']]
new_load_final3=new_load_final2
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(new_load_final2)
new_load_final2 = pd.DataFrame(np_scaled)

## Standardised Data load_final2 renaming columns

new_load_final2.columns=[ ['COND_PMP_BRG_HORI_VIB', 'COND_PMP_BRG_VERT_VIB',
       'COND_PMP_DE_JRNL_BRG_TEMP', 'COND_PMP_MOT_STR__PH_TEMP',
       'COND_PMP_MOT_STR__PH_B_TEMP', 'COND_PMP_MOT_STR__PH_C_TEMP',
       'COND_PMP_MOTOR_CURRENT', 'COND_PMP_NDE_JRNL_BRG_TEMP',
       'COND_PMP_THR_BRG_MTL_TEMP', 'COND_PMP_OUTLET_PR',
       'COND_PMP_OUTLET_TEMP', 'COND_WATER_FLOW_1', 'COND_WATER_FLOW_2',
       'COND_WATER_PR', 'COND_WATER_RECIRC_FLOW', 'CONDENSATE_PMP_INLET_PR',
       'CONDENSATE_PMP_INLET_TEMP', 'CONDENSER_VACUUM',
       'CONDENSR_HOTWELL_WATER_LVL_1', 'CONDENSR_HOTWELL_WATER_LVL_2',
       'CONDENSR_HOTWELL_WATER_LVL_3']]
new_load_final2.head()

## Checking load_final2 and load_final3 shape

if new_load_final2.shape==new_load_final3.shape:
    print('Oh, Yeah')
else:
    pass

## Resetting index to Datetime

Date_Time=Date_Time.reset_index(drop=True)
new_load_final3=pd.concat([Date_Time,new_load_final3],axis=1)
new_load_final2=pd.concat([Date_Time,new_load_final2],axis=1)

new_load_final3.head()

## Now we have Load_final2 and Load_final3 with datetime columns and standardised



## PCA

from sklearn.decomposition import PCA
new_load_final2=new_load_final2.set_index('DATETIME')
pca = PCA(n_components=2,random_state=5)
pca.fit(new_load_final2)
new_x_pca = pca.transform(new_load_final2)
new_load_final_pca=pd.DataFrame(new_x_pca,columns=['PCA1','PCA2'])


# In[87]:


new_df=pd.DataFrame()
new_df['Cluster'] = kmeans[16].predict(new_load_final_pca)


# In[91]:


new_df['Cluster'].unique()


# In[93]:


new_load_final_pca=pd.concat([new_load_final_pca,new_df],axis=1)


# In[94]:


new_load_final_pca


# ## Cluster Anomalies to be deleted as divided them into clusters

# In[ ]:


new_load_final_pca['Cluster_anomaly_flag']=0


# In[95]:




for i in range(0,len(new_load_final_pca)):
    if ((new_load_final_pca['Cluster'].iloc[i]==2) | (new_load_final_pca['Cluster'].iloc[i]==10) | (new_load_final_pca['Cluster'].iloc[i]==11) | (load_final_pca_without_out['Cluster'].iloc[i]==13)  | (load_final_pca_without_out['Cluster'].iloc[i]==15)):
        new_load_final_pca['Cluster_anomaly_flag'].iloc[i]=1
    else:
        pass


# In[1]:


LOF_Anomalies


# # END

# In[ ]:


Final_Dataframe


# In[ ]:


Load_Load


# In[ ]:


Date_Time


# In[ ]:


Load_final_pca.index

