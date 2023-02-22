import numpy as np
import pandas as pd 


from google.colab import files
uploaded = files.upload()
ED = pd.read_csv('/content/data.csv',encoding='unicode_escape')
ED = pd.read_csv('/content/data.csv',encoding='unicode_escape',
                 dtype ={'CustomerID' : str , 'InvoiceID' : str})
#InvoiceDate Dtype is object ===> must be datetime
ED['InvoiceDate'] = pd.to_datetime(ED['InvoiceDate'])
from pandas_profiling import ProfileReport
profile = ProfileReport(ED, title = 'Pandas Profiling Report of Ecommerce Dat')
profile
ED.dropna(axis = 0, subset = ['CustomerID'], inplace = True)
ED.isnull().sum()
temp = ED[['CustomerID', 'InvoiceNo', 'Country']].groupby(['CustomerID', 'InvoiceNo', 'Country']).count()
temp = temp.reset_index(drop = False) # Merge to Not Merge (like DB)
countries = temp['Country'].value_counts()
ED['CustomerID'] = ED['CustomerID'].astype('int64')
#Remove Quantity with negative values
ED = ED[ED.Quantity > 0]
#Add the column - amount_spent
ED['amount_spent'] = ED['Quantity'] * ED['UnitPrice']