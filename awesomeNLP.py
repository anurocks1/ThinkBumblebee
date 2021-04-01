import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
import time
from collections import Counter
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


folder1 = input("Enter input csv name  ")


with open('real_data0.csv',encoding = 'utf-8',errors = 'ignore') as file1:
    newdata1=file1.read()
from io import StringIO
data1=pd.read_csv(StringIO(newdata1))
#data1 = data1.drop_duplicates(keep='first')
#data1.to_csv('e1data.csv',encoding = 'utf-8-sig')
#del data1['Unnamed: 0']

with open(folder1,encoding = 'utf-8',errors = 'ignore') as file2:
    newdata2=file2.read()
from io import StringIO
data2=pd.read_csv(StringIO(newdata2))


#data1.to_csv('realdata.csv')

data2 = pd.DataFrame(data2.iloc[:,[9,77,7,36,45,88,47,49,51,50,83,84]].values) #change it to names of the columns whenever you get the time
data2[3] = data2[3] + data2[4] + data2[5]
del data2[4]
del data2[5]
Y = data1.iloc[:,10].values
del data1['label']

data2.columns = list(data1.columns.values)


my_dict={}

my_dict={"Afrikaans":":2",
         "Albanian":":3",
         "Amharic":":4",
         "Arabic":":5",
         "Armenian":":6",
         "Azerbaijani":":7",
         "Basque":":8",
         "Belarusian":":9",
         "Bengali":":a",
         "Bosnian":":b",
         "Bulgarian":":c",
         "Catalan":":d",
         "Cebuano":":e",
         "Chichewa":":f",
         "Chinese":":g",
         "Corsican":":h",
         "Croatian":":i",
         "Czech":":j",
         "Danish":":k",
         "Dutch":":l",
         "English":":m",
         "Esperanto":":n",
         "Estonian":":o",
         "Filipino":":p",
         "Finnish":":q",
         "French":":r",
         "Frisian":":s",
         "Galician":":t",
         "Georgian":":u",
         "German":":v",
         "Greek":":w",
         "Gujarati":":x",
         "Haitian Creole":":y",
         "Hausa":":z",
         "Hawaiian":":10",
         "Hebrew":":11",
         "Hindi":":12",
         "Hmong":":13",
         "Hungarian":":14",
         "Icelandic":":15",
         "Igbo":":16",
         "Indonesian":":17",
         "Irish":":18",
         "Italian":":19",
         "Japanese":":1a",
         "Javanese":":1b",
         "Kannada":":1c",
         "Kazakh":":1d",
         "Khmer":":1e",
         "Korean":":1f",
         "Kurdish (Kurmanji)":":1g",
         "Kyrgyz":":1h",
         "Lao":":1i",
         "Latin":":1j",
         "Latvian":":1k",
         "Lithuanian":":1l",
         "Luxembourgish":":1m",
         "Macedonian":":1n",
         "Malagasy":":1o",
         "Malay":":1p",
         "Malayalam":":1q",
         "Maltese":":1r",
         "Maori":":1s",
         "Marathi":":1t",
         "Mongolian":":1u",
         "Myanmar (Burmese)":":1v",
         "Nepali":":1w",
         "Norwegian":":1x",
         "Pashto":":1y",
         "Persian":":1z",
         "Polish":":20",
         "Portuguese":":21",
         "Punjabi":":22",
         "Romanian":":23",
         "Russian":":24",
         "Samoan":":25",
         "Scots Gaelic":":26",
         "Serbian":":27",
         "Sesotho":":28",
         "Shona":":29",
         "Sindhi":":2a",
         "Sinhala":":2b",
         "Slovak":":2c",
         "Slovenian":":2d",
         "Somali":":2e",
         "Spanish":":2f",
         "Sundanese":":2g",
         "Swahili":":2h",
         "Swedish":":2i",
         "Tajik":":2j",
         "Tamil":":2k",
         "Telugu":":2l",
         "Thai":":2m",
         "Turkish":":2n",
         "Ukrainian":":2o",
         "Urdu":":2p",
         "Uzbek":":2q",
         "Vietnamese":":2r",
         "Welsh":":2s",
         "Xhosa":":2t",
         "Yiddish":":2u",
         "Yoruba":":2v",
         "Zulu":":2w"
         }





content = data2['content']
trial = data2['content']


from selenium.webdriver.firefox.webdriver import WebDriver
driver=WebDriver()
driver.get('https://translate.google.co.in/')
driver.maximize_window()

remarks=[]
semarks = []
for i in range(0,len(content)):
    try:
        temp1= re.sub(r"[A-Za-z0-9\s]"," ",content[i])
        temp1=''.join(e for e in temp1 if e.isalnum())
        val=0
        if(temp1!=''):
            temp=trial[i]
            length_stat=len(trial[i])
            while(((temp1)!='') and (val<3) and ((len(temp1)/len(length_stat)) > 0.005)):
                search_field = driver.find_element_by_id("source")
                search_field.clear()
                #print("text_lang_detect",temp1)
                search_field.send_keys(temp1)
                #time.sleep(1)
                
                dropdown=driver.find_element_by_xpath("//*[@id='gt-sl-gms']")
                dropdown.click()
                time.sleep(0.4)
                dropdown1=driver.find_element_by_xpath("//*[@id=':0']/div")
                dropdown1.click()
                
                dropdown2=driver.find_element_by_xpath("//*[@id='gt-sl-gms']")
                dropdown2.click()
                time.sleep(0.4)
                dropdown3=driver.find_element_by_xpath("//*[@id=':0']/div")
                time.sleep(1)
                
                s=dropdown3.text
                #print(s)
                dropdown3.click()
                
                
                k=(s.split("-"))[0]
                l=k[:-1]
                print("language",l)
                #print(l)
                id_name=my_dict[l]
                search_field.clear()
                #print('translation_text',temp)
                search_field.send_keys(temp)
                dropdown=driver.find_element_by_xpath("//*[@id='gt-sl-gms']")
                dropdown.click()
    
                #dropdown1=driver.find_element_by_xpath("//*[@id=':g']")
                idd="//*[@id='"+id_name+"']"
                dropdown1=driver.find_element_by_xpath(idd)
                #time.sleep(1)
                dropdown1.click()
    
                dropdown2=driver.find_element_by_xpath("//*[@id='gt-submit']")
                dropdown2.click()
                time.sleep(1)
    
    
                temp=driver.find_element_by_id("gt-res-dir-ctr").text
                #print("final_result:",temp)
                temp1= re.sub(r"[A-Za-z0-9\s]"," ",temp)
                temp1=''.join(e for e in temp1 if e.isalnum())
                val=val+1
            remarks.append(temp)
        else:
            remarks.append(trial[i])
    except:
        remarks.append(trial[i])

#C = pd.DataFrame(c)

from googletrans import Translator
translator = Translator(service_urls=['translate.google.com'])
def remove(string):
    NON_BMP_RE = re.compile(u"[^\U00000000-\U0000d7ff\U0000e000-\U0000ffff]", flags=re.UNICODE)
    return NON_BMP_RE.sub(u'', string)
app =[]
pecus = []
#print('second check point')
for i in range(0,len(remarks)):
    try:
        p = translator.translate(remove(remarks[i]) , dest = 'en').text
        app.append(p)
    except:
        app.append(' ')
        pecus.append(i)
        continue
#print('third check point')        
honey = pd.concat([pd.DataFrame(app),pd.DataFrame(remarks)],axis = 1)

for i in range(0,len(honey)):
    if honey.iloc[i,0] == ' ':
        honey.iloc[i,0] = honey.iloc[i,1]
    #honey.iloc[:,0] = c   
data2['content'] = honey.iloc[:,0]

#data2.to_csv('japan_cons.csv')

################################################## Main Code #############################################################################3


X11 = pd.DataFrame(data1.iloc[:,2].values)
X11.columns = ['content']
X21 = pd.DataFrame(data2.iloc[:,2].values)
X21.columns = ['content']

X11=data1.iloc[:,2].values
X21 = data2.iloc[:,2].values
V = np.concatenate((X11,X21))




def words(text): return re.findall('[a-z]+', text.lower()) 
dictionary = Counter(words(open('big.txt').read())) #the big text has to contain dictionary of almost all the word in english language
def word_prob(word): return dictionary[word] / total
max_word_length = max(map(len, dictionary))
#making corpus2
def viterbi_segment(text):
    probs, lasts = [1.0], [0]
    for i in range(1, len(text) + 1):
        prob_k, k = max((probs[j] * word_prob(text[j:i]), j)
                        for j in range(max(0, i - max_word_length), i))
        probs.append(prob_k)
        lasts.append(k)
    words = []
    i = len(text)
    while 0 < i:
        words.append(text[lasts[i]:i])
        i = lasts[i]
    words.reverse()
    return words  #, probs[-1]

total = float(sum(dictionary.values()))

hash = []
for i in range(0,len(V)):
    r = re.findall(r'#[\w_]+',V[i])
    r = ' '.join(r)
    r = re.sub('[^a-zA-Z]',' ',r)
    hash.append(r)
    
corpus2 = []
for i in range(0,len(hash)):    
    pg = hash[i].split()
    o = []
    for j in pg:
        v = viterbi_segment(j)
        v = [word for word in v if word not in set(stopwords.words('english'))]
        w = ' '.join(v)
        o.append(w)
    x = ' '.join(o)
    corpus2.append(x) 
    
#tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf2 = TfidfVectorizer(ngram_range = (1,2),max_df=0.2,min_df=50/len(V))
#Xhash = pd.DataFrame(tfidf2.fit_transform(corpus2).toarray())
Xhash = tfidf2.fit_transform(corpus2).toarray()
X1hash,X2hash= Xhash[:len(data1),:],Xhash[len(data1):,:] #train test split

import time #158.9153699874878 seconds
start_time = time.time()      
corpus1 = []
word_count = []
hash_count = []
tag_count = []
url_count = []
iscontact = []
sale_count = []
ord_siz_count = []
price_currency_count = []

for i in range(0,len(V)):
    review = re.sub('http(s)?://t\.co/[\w0-9]+',' ',V[i]) 
    if re.findall(r'(size|order)',review,re.I):
        ord_siz_count.append(1)
    else:
        ord_siz_count.append(0)
    if re.findall(r'(price|\$)',review,re.I):
        price_currency_count.append(1)
    else:
       price_currency_count.append(0)
    if re.findall(r'([^#][0-9]{4}(\s|-)?[0-9]{4})|(([0-9]{2})?[0-9]{2}(\s|(-|,))?[0-9]{3}(\s|(-|,))?[0-9]{3})',review):
        iscontact.append(1)
    else:
        iscontact.append(0)
        
    if re.findall('(discount)|(% (off|OFF))',review):
        sale_count.append(1)
    else:
        sale_count.append(0)
    
    if re.search(r'(http(s)?://\w+\.)|(\.com)',review):
        url_count.append(1)
    else:
        url_count.append(0)
    review = re.sub('[^a-zA-Z#@]',' ',review)
    hash_count.append(len(re.findall(r'#',review)))
    tag_count.append(len(re.findall(r'@',review)))  
    review = re.sub(r'(\s)?(#|@)\w+',' ',review)                             
    review = review.lower()
    review = review.split()
    word_count.append(len(review))    #count words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus1.append(review) 
    




#corpus 1 ka kaam
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf1 = TfidfVectorizer(ngram_range = (1,2),max_df=0.50,min_df=0.004,max_features = 550)
X1 = tfidf1.fit_transform(corpus1).toarray()
X11new,X21new= X1[:len(data1),:],X1[len(data1):,:]

order_size_counts = pd.DataFrame(ord_siz_count)
order_size_counts.columns = ['order_size_count']
#price currency count
price_currency_counts = pd.DataFrame(price_currency_count)
price_currency_counts.columns = ['price_currency_count']


#pd.DataFrame(iscontact).to_csv('contact0.csv')
#contact count
iscontacts = pd.DataFrame(iscontact)
iscontacts.columns = ['iscontact']

#sale count
sale_counts = pd.DataFrame(sale_count)
sale_counts.columns = ['sale_count']

#url_count
url_counts =   pd.DataFrame(url_count)
url_counts.columns = ['url_count']

#word count
word_counts =   pd.DataFrame(word_count) 
word_counts.columns = ['word_count']
#hash_count
hash_counts =   pd.DataFrame(hash_count)
hash_counts.columns = ['hash_count']
#tag_count
tag_counts = pd.DataFrame(tag_count)
tag_counts.columns = ['tag_count']





ssr =[]
ssr = pd.concat([data1,data2])
del ssr['content']
#data 
ssr['word_count'] = word_counts['word_count']
ssr['hash_count'] = hash_counts['hash_count']
ssr['tag_count'] = tag_counts['tag_count']
ssr['url_count'] = url_counts['url_count']
ssr['iscontact'] = iscontacts['iscontact']
ssr['sale_count'] = sale_counts['sale_count']
ssr['price_currency_count'] = price_currency_counts['price_currency_count']
ssr['ord_size_count'] = order_size_counts['order_size_count']

print("--- %s seconds ---" % (time.time() - start_time))





from sklearn.preprocessing import LabelEncoder , OneHotEncoder#normal encoding of the dataset and the variable
labelencoder_ssr = LabelEncoder()
#ssr = ssr.fillna('Australia')  
#ssr.isna().sum()
ssr.iloc[:,0]=labelencoder_ssr.fit_transform(ssr.iloc[:,0])
ssr.iloc[:,1]=labelencoder_ssr.fit_transform(ssr.iloc[:,1]) 
#X2.iloc[:,2]=labelencoder_X2.fit_transform(X2.iloc[:,2])#sentiments#one hot encoding of the independent variable
onehotencoder = OneHotEncoder(categorical_features = [0,1])
ssr = onehotencoder.fit_transform(ssr).toarray()#one hot encoding of the dependent variable
labelencoder_Y = LabelEncoder()#chr(Y[:,0])#to make integer to character
Y=labelencoder_Y.fit_transform(Y)

X12new,X22new = ssr[:len(data1),:],ssr[len(data1):,:]
#X12new = np.delete(X12new,[14,15],1)#for  columng
#X22new = np.delete(X22new,[14,15],1)



Xtrain = np.concatenate((X12new,X11new,X1hash),axis =1)
Xtest = np.concatenate((X22new,X21new,X2hash),axis =1)
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
import time #0.19314002990722656 seconds
start_time = time.time()
X_resampled, Y_resampled = ros.fit_sample(Xtrain, Y)     
print("--- %s seconds ---" % (time.time() - start_time))


option = input('Which model to run ?xgb\tor\tnn?\t')
folder2 = input("Enter csv name you want to save:- ")
if(option=='xgb'):
    from xgboost import XGBClassifier
    xgb = XGBClassifier(learning_rate = 0.1 , n_estimators = 400 ,n_jobs = -1 , random_state = 0)
    import time
    start_time = time.time()
    xgb.fit(X_resampled,  Y_resampled) #224.76897954940796 seconds
    print("--- %s seconds ---" % (time.time() - start_time))
    Y_pred_xgb = xgb.predict(Xtest)
    Y_pred_xgb = pd.DataFrame(Y_pred_xgb)
    Y_pred_xgb.columns = ['prediction']
    Y_pred_xgb.to_csv(folder2)
    
else:
    if(option=='nn'):
        Xann = np.concatenate((X_resampled,Xtest))
        Xann = np.delete(Xann, [0,3], 1)
        Xtest_naya = np.delete(Xtest, [0,3], 1)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        Xann  = sc.fit_transform(Xann)
        Xtrain_new,Xtest_new= Xann[:len(X_resampled),:],Xann[len(X_resampled):,:]
        
        clf = Sequential()
        clf.add(Dropout(0.3,input_shape = (len(Xtrain_new[1,:]) ,)))
        clf.add(Dense(activation = 'relu'  ,units = int((len(Xtrain_new[1,:])+1)/2) , kernel_initializer = 'uniform' ))
        clf.add(Dropout(0.5 ))
        clf.add(Dense(activation = 'relu'  ,units = int((len(Xtrain_new[1,:])+1)/2) , kernel_initializer = 'uniform' ))
        clf.add(Dropout(0.5 ))
        clf.add(Dense(activation = 'sigmoid' ,units = 1 , kernel_initializer = 'uniform' ))
        clf.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
        clf.fit(Xtrain_new, Y_resampled , batch_size = int(len(Xtrain_new)/20) , epochs = 20 )
        
        Y_pred_nn = clf.predict(Xtest_new) 
        def formate(Y_pred_nn):
            for i in range(0,len(Y_pred_nn)):  
                if(Y_pred_nn[i][0] > 0.5):
                    Y_pred_nn[i][0] = 1
                else:
                    Y_pred_nn[i][0] = 0
                    
        formate(Y_pred_nn)
        Y_pred_nn = pd.DataFrame(Y_pred_nn)
        Y_pred_nn.columns = ['prediction']
        Y_pred_nn.to_csv(folder2)        
       
        



