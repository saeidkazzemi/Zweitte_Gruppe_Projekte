import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns 
import datetime
import warnings
warnings.filterwarnings('ignore')
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
from IPython.display import display, HTML
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
color = sns.color_palette()


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


#How many orders made by the customers?
orders = ED.groupby(by=['CustomerID','Country'], as_index=False)['InvoiceNo'].count()
plt.subplots(figsize=(15,6))
plt.plot(orders.CustomerID, orders.InvoiceNo)
plt.xlabel('Customers ID')
plt.ylabel('Number of Orders')
plt.title('Number of Orders for different Customers')
plt.show()



print('The TOP 5 customers with most number of orders...')
orders.sort_values(by='InvoiceNo', ascending=False).head()



money_spent = ED.groupby(by=['CustomerID','Country'], as_index=False)['amount_spent'].sum()

plt.subplots(figsize=(15,6))
plt.plot(money_spent.CustomerID, money_spent.amount_spent)
plt.xlabel('Customers ID')
plt.ylabel('Money spent (Dollar)')
plt.title('Money Spent for different Customers')
plt.show()



# rearrange all the columns for easy reference
ED_new = ED[['InvoiceNo','InvoiceDate','StockCode','Description',
                 'Quantity','UnitPrice','amount_spent','CustomerID','Country']]
ED_new.insert(loc=2, column='year_month', value=ED_new['InvoiceDate'].map(lambda x: 100*x.year + x.month))
ED_new.insert(loc=3, column='month', value=ED_new.InvoiceDate.dt.month)
# +1 to make Monday=1.....until Sunday=7
ED_new.insert(loc=4, column='day', value=(ED_new.InvoiceDate.dt.dayofweek)+1)
ED_new.insert(loc=5, column='hour', value=ED_new.InvoiceDate.dt.hour)

# Item sale with Quantity <=0 or unitPrice < 0
print (((ED_new['Quantity'] <= 0) | (ED_new['UnitPrice'] < 0)).value_counts())

#Delete the negative values 
ED_new = ED_new.loc[(ED_new['Quantity'] > 0) | (ED_new['UnitPrice'] >= 0)]
ax = ED_new.groupby('InvoiceNo')['year_month'].unique().value_counts().sort_index().plot(kind = 'bar',figsize=(15,6))
ax.set_xlabel('Month',fontsize=15)
ax.set_ylabel('Number of Orders',fontsize=15)
ax.set_title('Number of orders for different Months (1st Dec 2010 - 9th Dec 2011)',fontsize=15)
ax.set_xticklabels(('Dec_10','Jan_11','Feb_11','Mar_11','Apr_11','May_11','Jun_11','July_11','Aug_11',
                    'Sep_11','Oct_11','Nov_11','Dec_11'),rotation='horizontal', fontsize=13)
plt.show()
ED_new.groupby('InvoiceNo')['day'].unique().value_counts().sort_index
ax2 = ED_new.groupby('InvoiceNo')['day'].unique().value_counts().sort_index().plot(kind ='bar',color=color[0],figsize=(15,6))
ax2.set_xlabel('Day',fontsize=15)
ax2.set_ylabel('Number of Orders',fontsize=15)
ax2.set_title('Number of orders for different Days',fontsize=15)
ax2.set_xticklabels(('Mon','Tue','Wed','Thur','Fri','Sun'), rotation='horizontal', fontsize=15)
plt.show()
%%timeit

ED['yearmonth'] = ED['InvoiceDate'].apply(lambda x: (100*x.year) + x.month)
ED['Week'] = ED['InvoiceDate'].apply(lambda x: x.strftime('%W'))
ED['day'] = ED['InvoiceDate'].apply(lambda x: x.strftime('%d'))
ED['Weekday'] = ED['InvoiceDate'].apply(lambda x: x.strftime('%w'))
ED['hour'] = ED['InvoiceDate'].apply(lambda x: x.strftime('%H'))

plt.figure(figsize=(12,6))
plt.title("Frequency of order by Day", fontsize=15)
InvoiceDate = ED.groupby(['InvoiceNo'])['day'].unique()
InvoiceDate.value_counts().sort_index().plot.bar()


group_country_orders = ED_new.groupby('Country')['InvoiceNo'].count().sort_values()
# del group_country_orders['United Kingdom']

# plot number of unique customers in each country (with UK)
plt.subplots(figsize=(15,8))
group_country_orders.plot(kind='barh', fontsize=12, color=color[0])
plt.xlabel('Number of Orders', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.title('Number of Orders for different Countries', fontsize=12)
plt.show()