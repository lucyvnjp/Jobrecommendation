#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import RAKE
import operator
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import wordcloud as w
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,make_scorer,classification_report,accuracy_score
import scikitplot as skplt
from sklearn import svm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings("ignore")


# ##1.Import the jobpostings.csv

# In[2]:


df_job = pd.read_csv("./dice_com-job_us_sample.csv")
df_job.head()


# #2. Cleaning of Data

#  2.1 Remove unnecessary columns

# #remove postdate, shift, sitename

# In[3]:


df_job.drop("postdate", inplace = True, axis = 1)
df_job.drop("shift", inplace = True, axis = 1)
df_job.drop("site_name", inplace = True, axis = 1)
df_job.drop("uniq_id", inplace = True, axis = 1)
df_job.drop("jobid", inplace = True, axis = 1)
df_job.head()


#  # 2.1 Remove dulicates

# In[4]:


df_job.drop_duplicates()
df_job.head(5)


# #2.3 Drop empty and rubbish words

# In[5]:


print(df_job.isnull().values.any())

df_job.isna().sum()


# In[6]:


# Total number of missing values
df_job.isnull().sum().sum()


# In[7]:


# drop all rows that have any NaN values
df_job.dropna(inplace = True)
# check that there is no more na
df_job.isna().sum()


# In[8]:


# Get index of all rows in skills that contains the value "Null"
jobNull = df_job[df_job["skills"] =="Null"].index
jobDesc1 = df_job[df_job["skills"] =="Please see job description"].index
jobDesc2 = df_job[df_job["skills"] =="(See job description)"].index
jobDesc3 = df_job[df_job["skills"] =="SEE BELOW"].index
jobDesc4 = df_job[df_job["skills"] =="Telecommuting not available Travel not required"].index
jobDesc5 = df_job[df_job["skills"] =="Refer to Job Description"].index
jobDesc6 = df_job[df_job["skills"] =="Please see Required Skills"].index

# Drop rows of index
df_job.drop(jobNull, inplace =True)
df_job.drop(jobDesc1, inplace =True)
df_job.drop(jobDesc2, inplace =True)
df_job.drop(jobDesc3, inplace =True)
df_job.drop(jobDesc4, inplace =True)
df_job.drop(jobDesc5, inplace =True)
df_job.drop(jobDesc6, inplace =True)


# In[9]:


# Print info of datafram
df_job.info()


# ##Data Visualization of the data

# #Top 5 most demand jobs¶

# In[10]:


# See the top 5 highest frequency of job titles
df_job["jobtitle"].value_counts()[:5]


# In[11]:


import nltk
nltk.download('punkt')
nltk.download('wordnet')


# In[12]:


from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[13]:


# Creating a stop_words list from the NLTK. We could also use the set of stopwords from Spacy or Gensim.
#from nltk.corpus import stopwords # Import stopwords from nltk.corpus
#stop_words = stopwords.words('english') # Create a list `stop_words` that contains the English stop words list


# In[14]:


# Create a CSV file to store a set of stopwords

#import csv # Import the csv module to work with csv files
#with open('./stopwordskills.csv', 'w', newline='') as f:
  #  writer = csv.writer(f)
   # writer.writerow(stop_words)


# In[15]:


# Open the CSV file and list the contents

with open('./stop_words.csv', 'r') as f:
    stop_words = f.read().strip().split(",")
stop_words[:-10]


# In[16]:


lemmatizer = WordNetLemmatizer()
for index, row in df_job.iterrows():
    filter_sentence = ''
    sentence = row['jobtitle']
    sentence = re.sub(r'[^\w\s]',' ', str(sentence))#cleaning
    words = nltk.word_tokenize(sentence) # tokenization
    words = [w for w in words if not w in stop_words] # stopwords removal
    for word in words:
         filter_sentence = filter_sentence + ' '+  lemmatizer.lemmatize(word)#append(lemmatizer.lemmatize(word))  
    print(filter_sentence)
    df_job.loc[index,'jobtitle'] = filter_sentence


# In[17]:


job = []
stopwordsList = []
cleanJobs = []
# Get the stopwords and store in list
with open('./stop_words.csv', 'r', encoding="utf-8") as f:
    for word in f:
        word = word.split(" \n")
        stopwordsList.append(word[0])
        
 # Tokenizing and Removing stop words from jobtitle
from nltk.tokenize import word_tokenize
import nltk
               
# Convert all words to lower case and change the shortform
for i in df_job["jobtitle"].values:
    jobs = i.lower()
    jobs = jobs.replace("QA", "Quality Assurance")
    jobs = jobs.replace("sr", "Senior")
    jobs = jobs.replace("jr", "Junior")
    jobs = jobs.replace("qm", "Quality Manager")
    job.append(jobs)
    
# tokenize and remove the words from stop words 
for j in job:
    text_tokens = word_tokenize(j)
    tokens_without_sw = [f for f in text_tokens if not f in stopwordsList]
    cleanJobs.append(' '.join(tokens_without_sw))


# In[18]:


from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# In[19]:


# Add the newly cleaned job title into the dataframe
df_job["clean_jobtitle"] = cleanJobs
df_job.head(5)


# In[20]:


#Get the Top 20 Jobs
qty = df_job["clean_jobtitle"].value_counts()[:10].tolist()
label = df_job["clean_jobtitle"].value_counts()[:10].index.tolist()
print(qty)
print("Top 10 Popular Jobs: " + str(label))


# In[21]:


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i],y[i], ha = 'center')
skillslabel = label
jobQty = qty
plt.figure(figsize = (10, 5))
plt.bar(skillslabel, jobQty, color = [ 'purple', 'red', 'green','blue', 'orange'])
addlabels(skillslabel, jobQty)
plt.title("Top 10 High demand jobs")
plt.xlabel("Name of Jobs")
plt.ylabel("Quantity")
# Visualizing the plot
plt.show()


# ## 2 Most Skills

# In[22]:


df_job["skills"]= df_job["skills"].str.lower()


# In[23]:


df_job.head()


# In[24]:


# importing Nltk stopword package
#import nltk
#nltk.download('stopwords')
#from nltk.corpus import stopwords

# Loading Stopwords into a list
#NLTK_stop_words_list = stopwords.words('english')
#print(NLTK_stop_words_list)
#print("Total numbers of stop words are")
#print(len(NLTK_stop_words_list))


# In[25]:


#custom_stop_word_list =['it','see','job','description','please','refer','to','job','description','tad','pgs','inc','specializes','in','delivering','secure','reliable','and','rapidly','implemented','workforce','solutions','to','the','u.s','federal','marketplace','including','u.s','government','agencies','and','their','prime','contractors','wi','see','below','qa','telecommuting','not','available','travel','not','arequired','full','time','lawson','chain']


# In[26]:


#final_stopword_list = custom_stop_word_list + NLTK_stop_words_list
#print(final_stopword_list)
#print(len(final_stopword_list))


# In[27]:


# Create a CSV file to store a set of stopwords

#import csv # Import the csv module to work with csv files
#with open('stopwordskills.csv', 'w', newline='') as f:
  #  writer = csv.writer(f)
   # writer.writerow(final_stopword_list)
    
    
        


# In[28]:


#with open('./stopwordskills.csv', 'r') as f:
    #  final_stopword_list = f.read().strip().split(",")
#final_stopword_list[-5:]


# In[29]:


#lemmatizer = WordNetLemmatizer()
#for index, row in df_job.iterrows():
   # filter_sentence = ''
  #  sentence = row["skills"]
  #  sentence = re.sub(r'[^\w\s]',' ', str(sentence))#cleaning
    #words = nltk.word_tokenize(sentence) # tokenization
   # words = [w for w in words if not w in stop_words] # stopwords removal
    #for word in words:
    #     filter_sentence = filter_sentence + ' '+  lemmatizer.lemmatize(word)#append(lemmatizer.lemmatize(word))  
    #print(filter_sentence)
   # df_job.loc[index,"skills"] = filter_sentence


# In[30]:


skillsTokenized = []
stopwordsSkills = []

# Get the stopwords and store in list 
with open('./stop_words.csv', 'r', encoding="utf-8") as f:
     for word in f:
        word.lower()
        word = word.split('\n')
        
        stopwordsSkills.append(word[0])
            
for k in df_job['skills'].values:
    k = str(k).split(',')
    # remove stopwords from skills
    skillstokens_without_sw = [f for f in k if not f.lower() in stopwordsSkills]
    for j in skillstokens_without_sw:
            skillsTokenized.append(j)
            
#put the cleaned skills into a new dataframe
df = pd.DataFrame({'skills' :skillsTokenized}) 


# In[31]:


#print(stopwordsSkills)


# In[32]:


lemmatizer = WordNetLemmatizer()
for index, row in df_job.iterrows():
    filter_sentence = []
    sentence = row["skills"]
    sentence = re.sub(r'[^\w\s]',' ', sentence)#cleaning
    words = nltk.word_tokenize(sentence) # tokenization
    words = [w for w in words if not w in stopwordsSkills] # stopwords removal
    for word in words:
         filter_sentence.append(lemmatizer.lemmatize(word))  
    print(filter_sentence)
    df_job.loc[index,"skills"] = filter_sentence


# In[33]:


lemmatizer = WordNetLemmatizer()
for index, row in df_job.iterrows():
    filter_sentence = ''
    sentence = row["skills"]
    sentence = re.sub(r'[^\w\s]',' ', str(sentence))#cleaning
    words = nltk.word_tokenize(sentence) # tokenization
    words = [w for w in words if not w in stopwordsSkills] # stopwords removal
    for word in words:
         filter_sentence = filter_sentence + ' '+  lemmatizer.lemmatize(word)#append(lemmatizer.lemmatize(word))  
    print(filter_sentence)
    df_job.loc[index,"skills"] = filter_sentence


# In[34]:



df['skills'].head()


# In[35]:


#df_job['skills'].head()


# In[36]:


#Get the Top 5 Jobs

qtySkills = df['skills'].value_counts().tolist()
labelSkills = df['skills'].value_counts().index.tolist()


# In[37]:


#df['skills_without_stopwords'] = df['skills'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwordsSkills)]))
#df['skills_without_stopwords'].head(5)


# In[38]:


print("Top 5 Skills mostly needed \n" + str(df['skills'].value_counts()[:5]))


# In[39]:


import wordcloud as w
import numpy as np
import matplotlib.pyplot as plt

lskills = labelSkills
frequencies = qtySkills
# Wordcloud aks for a strings, and I have tried separating the terms with ', '~'

d = dict(zip(lskills, frequencies))
wordcloud = w.WordCloud(collocations = False, random_state =1, background_color = 'white', width = 3000, height = 2000).generate_from_frequencies(d)

plt.imshow(wordcloud, interpolation = 'bilinear' )
plt.axis("off")
plt.figure(figsize = (3000,3000))
plt.show()


# # 3.Data Mining

# 3.1 TF-IDF

# In[40]:


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_job['clean_jobtitle'].values)
print(X.shape)
analyze = vectorizer.build_analyzer()
# print (' Job titles'), analyze(str(jobtitle))
# print ( 'Document transform', X.toarray())

#print(vectorizer.get_feature_names())
features = vectorizer.get_feature_names()
# indices = zip(*X.nonzero())

# for row, column in indices:
#   print('(%d, %s) %f' %(row, features[column], X[row, column]))


# 4. Clustering Using KMeans

# 4.1 Getting the Optimize cluster using elbow method

# In[41]:


# Using the elbow method to find the optimal number of clusters
# Within Cluster Sum of Squares (WCSS)

wcss =[]
for i in range(1, 18):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42, max_iter = 600, n_init = 1)
    kmeans.fit(X)
    # inertia method returns wcss for that model 
    wcss.append(kmeans.inertia_)


# In[42]:


#plotting the graph
plt.figure(figsize = (10,5))
sns.lineplot(range(1,18), wcss, marker = 'o', color = 'red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[43]:


true_k = 10
model = KMeans(n_clusters = true_k, init = 'k-means++', max_iter = 600, n_init = 1, random_state = 42)
pred = model.fit_predict(X)
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()


# In[44]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sklearn_pca = PCA(n_components = 2)

Y_sklearn = sklearn_pca.fit_transform(X.toarray())
kmeans = KMeans(n_clusters = 10, init = 'k-means++', max_iter = 600, n_init = 1, random_state = 42 )
fitted = kmeans.fit(Y_sklearn)
prediction = kmeans.predict(Y_sklearn)

plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c = prediction, s=50, cmap = 'viridis')

centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c = 'black', s =300, alpha = 0.6)


# In[45]:


#from yellowbrick.cluster import SilhouetteVisualizer

#Fig, ax = plt.subplots (2, 2, figsize = (15,8))
#[7, 8, 9, 10]:
  #  '' '
  #  Tạo phiên bản KMeans cho số lượng cụm khác nhau
  #  '' '
  #  km = KMeans (n_clusters = i, init = 'k-mean ++', n_init = 10, max_iter = 100, random_state = 42)
   # q, mod = divmod (i, 2)
   # '' '
   # Tạo phiên bản SilhouetteVisualizer với phiên bản KMeans
   # Vừa với trình hiển thị
   # '' '
    #visualizer = SilhouetteVisualizer (km, Colors = 'yellowbrick', ax = ax [q-1] [mod])
    #visualizer.fit (X)


# In[46]:


# Silhoutte score ranges from -1 to 1.
# metric used to calculate the goodness of a clustering technique
# 1: Means clusters are well apart from each other and clearly distinguished
from sklearn.metrics import silhouette_score
print('KMeans Scaled Silhouette Score: {}' .format(silhouette_score(X, model.labels_, metric='euclidean')))


# In[47]:


def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
        
get_top_keywords(X, pred, features, 10)        
        


# In[48]:


## Do further classification, Give the 0,1,2,3 a name to understand better
#Cluster 0 = `Project Management, cluster 1 = frontend, cluster 2 = Deveops/Software Engineer,
#cluster 3 = Business Solution Consultation, cluster 4 = Cloud Architect/Network,
#cluster 5 = Analyst, cluster 6 = IT Business Management

label = []
for i in df_job['clean_jobtitle'].values:
    
        vec = vectorizer.transform([i])
        pred = model.predict(vec)
        if pred == 0:
            label.append("Project Management")
        elif pred == 1:
            label.append("Cloud Architect/Network")
        elif pred == 2:
            label.append("Develope web")
        elif pred == 3:
            label.append("Software Engineer")
        elif pred == 4:
            label.append("Full stack developer")
        elif pred == 5:
            label.append("Analyst")
        elif pred == 6:
            label.append("Sap Consultant") 
        elif pred == 7:
            label.append("Business Solution Consultation")   
        elif pred == 8:
            label.append("Frontend")
        else:
            label.append("IT Business Management")
        
df_job['Label'] = label
df_job.head(5)


# In[49]:


jobSkills = []
for i in df_job['skills']:
    jobSkills.append(i.lower())
    
Xclass = vectorizer.fit_transform(jobSkills)

#Split data into test and train. Test size 20% Train Size 80%
X_train,X_test,y_train,y_test = train_test_split(Xclass,label,test_size=0.2,random_state=42)


# #5.1 Logistic Reression

# In[50]:


# Obtain the best C range
Cparamrange = [0.1,0.5,0.8,1,2]

trainAcc = []
testAcc =[]

for i in Cparamrange:
    lrg = LogisticRegression(penalty = 'l2' , C = i, random_state = 42)
    lrg.fit(X_train, y_train)
    lrg_predtrain = lrg.predict(X_train)
    lrg_predtest = lrg.predict(X_test)
    trainacc = accuracy_score(y_train, lrg_predtrain)
    testacc = accuracy_score(y_test, lrg_predtest)
    trainAcc.append(trainacc)
    testAcc.append(testacc)
    
plt.plot(Cparamrange, trainAcc, 'ro-', Cparamrange, testAcc, 'bv--')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Number of C')
plt.ylabel('Accuracy')

         
    


# # C =0.1 is able to get the optimal accuracy

# In[51]:


# using the best c param range 
lrg = LogisticRegression(penalty = 'l2' , C = 0.1, random_state = 42)
lrg.fit(X_train, y_train)
lrg_pred = lrg.predict(X_test)
lrg_acc = accuracy_score(y_test,lrg_pred)
print("Accuracy of Logistic Regression: " + str(lrg_acc))
print(classification_report(y_test, lrg_pred))
   


# In[52]:


#plot confusion matrix
skplt.metrics.plot_confusion_matrix(
       y_test,
       lrg_pred,
       x_tick_rotation = 90,
       figsize = (6,5))


# 5.2 Kneighbors Classifier

# In[53]:


get_ipython().run_line_magic('matplotlib', 'inline')

numNeighbors = [1, 5, 6, 7, 8, 10, 15, 20, 25, 30, 35, 40, 50]
trainAcc = []
testAcc = []

for k in numNeighbors:
    clf1 = KNeighborsClassifier(n_neighbors = k, metric = 'minkowski', p=2)
    clf1.fit(X_train, y_train)
    Y_predTrain = clf1.predict(X_train)
    Y_predTest = clf1.predict(X_test)
    trainAcc.append(accuracy_score(y_train, Y_predTrain))
    testAcc.append(accuracy_score(y_test, Y_predTest))
    
plt.plot(numNeighbors, trainAcc, 'ro-', numNeighbors, testAcc, 'bv--')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')


# # Let 50 be the n_neighbours

# In[54]:


knn =KNeighborsClassifier(n_neighbors = 50, metric = 'minkowski', p=2)
knn.fit(X_train, y_train)
knn_pred = clf1.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
print("Accuracy of KNN: "  + str(knn_acc))
print(classification_report(y_test, knn_pred))


# In[55]:


skplt.metrics.plot_confusion_matrix(
         y_test,
         knn_pred,
         x_tick_rotation = 90,
         figsize = (6,5))


# #Decision Tree

# In[56]:


maxdepths = [ 2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50]   # 17 different depths

trainAccuracy = np.zeros(len(maxdepths))
testAccuracy = np.zeros(len(maxdepths))

index = 0
for depth in maxdepths:
    clf2 = DecisionTreeClassifier(max_depth = depth)
    clf2 = clf2.fit(X_train, y_train)
    Y_predTrain = clf2.predict(X_train)
    Y_predTest = clf2.predict(X_test)
    trainAccuracy[index] = accuracy_score(y_train, Y_predTrain)
    testAccuracy[index] = accuracy_score(y_test, Y_predTest)
    index += 1
    
    
plt.plot(maxdepths, trainAccuracy, 'ro-', maxdepths, testAccuracy,'bv--')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
    


# # Let max_depth be 7

# In[57]:


dt = DecisionTreeClassifier(max_depth = 10)
dt2 = dt.fit(X_train, y_train)
dt_pred = dt2.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print(" Accuracy of Decision Tree: " + str(dt_acc))
print(classification_report(y_test, dt_pred))


# In[58]:


skplt.metrics.plot_confusion_matrix(
      y_test,
      dt_pred,
      x_tick_rotation = 90,
      figsize =(6,5))


# # 5.4 Support Vector Machines

# In[ ]:


Csvm = [0.1,0.5,0.8,1,1.5,2,2.5,3,3.5]

trainAcc = []
testAcc = []

for c in Csvm:
    modelsvm = svm.SVC(C=c, gamma = 1, kernel = 'rbf')
    svmfit = modelsvm.fit(X_train,y_train)
    Y_predTrain = modelsvm.predict(X_train)
    Y_predTest = modelsvm.predict(X_test)
    trainAcc.append(accuracy_score(y_train, Y_predTrain))
    testAcc.append(accuracy_score(y_test, Y_predTest))
    
plt.plot(Csvm, trainAcc, 'ro-', Csvm, testAcc,'bv--')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Number of C')
plt.ylabel('Accuracy')


# In[60]:


svm = svm.SVC(C=5, gamma = 1, kernel = 'rbf', probability = True)
svmfit = svm.fit(X_train, y_train)
svm_predictions = svmfit.predict(X_test)
svm_acc = accuracy_score(y_test, svm_predictions)
print("Accuracy of SVM: " + str(svm_acc))
print(classification_report(y_test, svm_predictions))


# In[61]:


skplt.metrics.plot_confusion_matrix(
       y_test,
       svm_predictions,
       x_tick_rotation = 90,
       figsize =(6,5))


#  #6. Determine Best Classification to use

# In[62]:


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
        
# initializing the labels and model accuracy. The model accuracy round up to 2dp
labels = ("Logistic Regression", "KNeighbors", "Decision Tree", "Support Vector Machines")
modelsAccuracy = [round(lrg_acc, 2), round(knn_acc, 2), round(dt_acc,2),round(svm_acc,2)]

# setting figure size by using figure () function
plt.figure(figsize = (10,5))

# making the bar chart on the data
plt.bar(labels, modelsAccuracy, color =['purple', 'yellow', 'green', 'blue'])

# calling the function to add value labels
addlabels(labels, modelsAccuracy)

# giving X and Y labels
plt.xlabel("Name of Model")
plt.ylabel("Accuracy")

# visualizing the plot
plt.show()


# #6.2 ROC

# In[63]:


# predict probabilites
lrg_prob = lrg.predict_proba(X_test)[::,1]
knn_prob = knn.predict_proba(X_test)[::,1]
dt_prob = dt.predict_proba(X_test)[::,1]
svm_prob = svm.predict_proba(X_test)[::,1]

# roc curve for models

fprlrg, tprlrg, threshlrd = roc_curve(y_test, lrg_prob, pos_label = 'Business Solution Consultant')
fprknn, tprknn, threshknn = roc_curve(y_test, knn_prob, pos_label = 'Business Solution Consultant')
fprdt, tprdt, threshdt = roc_curve(y_test,dt_prob, pos_label = 'Business Solution Consultant')
fprsvm, tprsvm, threshsvm = roc_curve(y_test, svm_prob, pos_label = 'Business Solution Consultant')

# roc curve for tpr = fpr
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label = 'Business Solution Consultant')


# In[64]:


plt.style.use('seaborn')

# plot roc curves
plt.plot(fprlrg, tprlrg, linestyle ='--',color='orange', label='Logistic Regression')
plt.plot(fprknn, tprknn, linestyle ='--',color='green', label='KNN')
plt.plot(fprdt, tprdt, linestyle ='--',color='red', label='Decison Tree')
plt.plot(fprsvm, tprsvm, linestyle ='--',color='purple', label='SVM')
plt.plot(p_fpr, p_tpr, linestyle ='--', color='blue')

#title
plt.title('ROC curve')
# x label
plt.xlabel('False Position Rate')
# y label
plt.ylabel('True Position rate')

plt.legend(loc ='best')
plt.savefig('ROC', dpi = 300)
plt.show()


# #7.Data analysis
# #7.1 Finding out what are the skills needed for each job role
# #7.1.1 Fronted
# 

# In[65]:


labelData = df_job[df_job['Label'] == "Frontend"]

skillsClass = []

for index,row in labelData.iterrows():
    skills = [row['skills']]
    skillstokens_without_sw = [ f for f in skills if not f.lower() in stopwordsSkills]
    for j in skillstokens_without_sw:
        skillsClass.append(j)
        
# put the cleanded skills into s new dataframe
df_frontend = pd.DataFrame({'skills': skillsClass})

qtySKills = df_frontend["skills"].value_counts().tolist()
labelSkills = df_frontend["skills"].value_counts().index.tolist()


# In[66]:


lskills = labelSkills
frequencies = qtySkills
# Wordcloud aks for a strings, and I have tried separating the terms with ', '~'

d = dict(zip(lskills, frequencies))
wordcloud = w.WordCloud(collocations = False, random_state =1, background_color = 'white', width = 3000, height = 2000).generate(str(d))

plt.imshow(wordcloud, interpolation = 'bilinear' )
plt.axis("off")
plt.figure(figsize = (3000,3000))
plt.show()


# #7.1.2 Analyst

# In[67]:


labelData = df_job[df_job['Label'] == "Analyst"]

skillsClass = []

for index,row in labelData.iterrows():
    skills = [row['skills']]
    skillstokens_without_sw = [f for f in skills if not f.lower() in stopwordsSkills]
    for j in skillstokens_without_sw:
          skillsClass.append(j)
            
            
# put the cleaned skills into a new dataframe
df_analyst = pd.DataFrame({'skills': skillsClass})

qtySkills = df_analyst["skills"].value_counts().tolist()
labelSkills = df_analyst["skills"].value_counts().index.tolist()


# In[68]:


lskills = labelSkills
frequencies = qtySkills
# aks for a strings, and I have tried separating the terms with ', '~'

d = dict(zip(lskills, frequencies))
wordcloud = w.WordCloud(collocations = False, random_state =1, background_color = 'white', width = 3000, height = 2000).generate(str(d))

plt.imshow(wordcloud, interpolation = 'bilinear' )
plt.axis("off")
plt.figure(figsize = (3000,3000))
plt.show()


# #7.1.3 Business Solution COnsultation

# In[69]:


labelData = df_job[df_job['Label']  == "Business Solution Consultation"]

skillsClass = []

for index,row in labelData.iterrows():
    skills = [row['skills']]
    skillstokens_without_sw = [f for f in skills if not f.lower() in stopwordsSkills]
    for j in skillstokens_without_sw:
          skillsClass.append(j)
            
            
# put the cleaned skills into a new dataframe
df_consultant = pd.DataFrame({'skills': skillsClass})

qtySkills = df_consultant["skills"].value_counts().tolist()
labelSkills = df_consultant["skills"].value_counts().index.tolist()


# In[70]:


lskills = labelSkills
frequencies = qtySkills
# Wordcloud aks for a strings, and I have tried separating the terms with ', '~'

d = dict(zip(lskills, frequencies))
wordcloud = w.WordCloud(collocations = False, random_state =1, background_color = 'white', width = 3000, height = 2000).generate(str(d))

plt.imshow(wordcloud, interpolation = 'bilinear' )
plt.axis("off")
plt.figure(figsize = (3000,3000))
plt.show()


# #7.1.4 Cloud Architect/Network

# In[71]:


labelData = df_job[df_job['Label']  == "Cloud Architect/Network"]

skillsClass = []

for index,row in labelData.iterrows():
    skills = [row['skills']]
    skillstokens_without_sw = [f for f in skills if not f.lower() in stopwordsSkills]
    for j in skillstokens_without_sw:
          skillsClass.append(j)
            
            
# put the cleaned skills into a new dataframe
df_cloud = pd.DataFrame({'skills': skillsClass})

qtySkills = df_cloud["skills"].value_counts().tolist()
labelSkills = df_cloud["skills"].value_counts().index.tolist()


# In[72]:


lskills = labelSkills
frequencies = qtySkills
# Wordcloud aks for a strings, and I have tried separating the terms with ', '~'

d = dict(zip(lskills, frequencies))
wordcloud = w.WordCloud(collocations = False, random_state =1, background_color = 'white', width = 3000, height = 2000).generate(str(d))

plt.imshow(wordcloud, interpolation = 'bilinear' )
plt.axis("off")
plt.figure(figsize = (3000,3000))
plt.show()


# # 7.1.5 Devops/Software Engineer

# In[73]:


labelData = df_job[df_job['Label']  == "Software Engineer"]

skillsClass = []

for index,row in labelData.iterrows():
    skills = [row['skills']]
    skillstokens_without_sw = [f for f in skills if not f.lower() in stopwordsSkills]
    for j in skillstokens_without_sw:
          skillsClass.append(j)
            
            
# put the cleaned skills into a new dataframe
df_engineer = pd.DataFrame({'skills': skillsClass})

qtySkills = df_engineer["skills"].value_counts().tolist()
labelSkills = df_engineer["skills"].value_counts().index.tolist()


# In[74]:


lskills = labelSkills
frequencies = qtySkills
# Wordcloud aks for a strings, and I have tried separating the terms with ', '~'

d = dict(zip(lskills, frequencies))
wordcloud = w.WordCloud(collocations = False, random_state =1, background_color = 'white', width = 3000, height = 2000).generate(str(d))

plt.imshow(wordcloud, interpolation = 'bilinear' )
plt.axis("off")
plt.figure(figsize = (3000,3000))
plt.show()


# #7.1.6 IT Business Management

# In[75]:


labelData = df_job[df_job['Label']  == "IT Business Management"]

skillsClass = []

for index,row in labelData.iterrows():
    skills = [row['skills']]
    skillstokens_without_sw = [f for f in skills if not f.lower() in stopwordsSkills]
    for j in skillstokens_without_sw:
          skillsClass.append(j)
            
            
# put the cleaned skills into a new dataframe
df_bus = pd.DataFrame({'skills': skillsClass})

qtySkills = df_bus["skills"].value_counts().tolist()
labelSkills = df_bus["skills"].value_counts().index.tolist()


# In[76]:


lskills = labelSkills
frequencies = qtySkills
# Wordcloud aks for a strings, and I have tried separating the terms with ', '~'

d = dict(zip(lskills, frequencies))
wordcloud = w.WordCloud(collocations = False, random_state =1, background_color = 'white', width = 3000, height = 2000).generate(str(d))

plt.imshow(wordcloud, interpolation = 'bilinear' )
plt.axis("off")
plt.figure(figsize = (3000,3000))
plt.show()


# #7.1.7 Project Management

# In[77]:


labelData = df_job[df_job['Label']  == "Project Management"]

skillsClass = []

for index,row in labelData.iterrows():
    skills = [row['skills']]
    skillstokens_without_sw = [f for f in skills if not f.lower() in stopwordsSkills]
    for j in skillstokens_without_sw:
          skillsClass.append(j)
            
            
# put the cleaned skills into a new dataframe
df_pro = pd.DataFrame({'skills': skillsClass})

qtySkills = df_pro["skills"].value_counts().tolist()
labelSkills = df_pro["skills"].value_counts().index.tolist()


# In[78]:


lskills = labelSkills
frequencies = qtySkills
# Wordcloud aks for a strings, and I have tried separating the terms with ', '~'

d = dict(zip(lskills, frequencies))
wordcloud = w.WordCloud(collocations = False, random_state =1, background_color = 'white', width = 3000, height = 2000).generate(str(d))

plt.imshow(wordcloud, interpolation = 'bilinear' )
plt.axis("off")
plt.figure(figsize = (3000,3000))
plt.show()


# # 7 Get User Input

# In[79]:


userInput = input("Enter your skills: ")
pred = vectorizer.transform([userInput.lower()])

output = svm.predict(pred)
print(output[0])


# # 8.1 From the classm find the one that suits most to the user's input by doing Cosine Similarity/Euclidean Distance

# # 8.1 Cosine Similarity

# In[80]:


cos = []
labelData = df_job[df_job['Label'] == output[0]]

for index,row in labelData.iterrows():
    skills = [row['skills']]
    skillVec = vectorizer.transform(skills)
    cos_lib = cosine_similarity(skillVec, pred)
#   print(cos_lib[0][0])    
    cos.append(cos_lib[0][0])
    
labelData['cosine_similarity']  = cos   
    


# In[81]:


# Display top 5 recommendation from cosine similaity 
top_5 = labelData.sort_values('cosine_similarity', ascending =False)[['company', 'employmenttype_jobstatus','jobdescription','joblocation_address','jobtitle','skills','Label','cosine_similarity' ]]
#'advertiserurl',
top_5.head(5)


# #8.2 Euclidean Distance

# In[82]:


euclidean = []
labelData = df_job[df_job['Label'] == output[0]]

for index,row in labelData.iterrows():
    skills = [row['skills']]
    skillVec = vectorizer.transform(skills)
    euclidean_dist = euclidean_distances(skillVec, pred)
#   print(cos_lib[0][0])    
    euclidean.append(euclidean_dist[0][0])
    
labelData['euclidean_distance']  = euclidean  
    


# In[83]:


# Display top 5 recommendation from cosine similaity 
top_5 = labelData.sort_values('euclidean_distance', ascending =False)[[ 'company', 'employmenttype_jobstatus','jobdescription','joblocation_address','jobtitle','skills','Label','euclidean_distance' ]]
#'advertiserurl',
top_5.head(5)


# #9 Final Job Reccommendation

# In[88]:


#userInput = input("Find your perfect role.: ")
userInput = input("Find your perfect role. : ")
pred = vectorizer.transform([userInput.lower()])

output = svm.predict(pred)
#print(" You are looking for the job that is under our job's list: + output[0] + " jobs")
print("You are looking for the job that is under our job's list: "  + output[0])


cos = []
labelData = df_job[df_job['Label'] == output[0]]

for index,row in labelData.iterrows():
    skills = [row['skills']]
    skillVec = vectorizer.transform(skills)
    cos_lib = cosine_similarity(skillVec, pred)
#   print(cos_lib[0][0])    
    cos.append(cos_lib[0][0])
    
labelData['cosine_similarity']  = cos   


# Display top 5 recommendation from cosine similaity 
top_5 = labelData.sort_values('cosine_similarity', ascending =False)[['company', 'employmenttype_jobstatus','jobdescription','joblocation_address','jobtitle','skills','Label','cosine_similarity' ]]
#'advertiserurl', 
top_5.head(5)


# In[ ]:




