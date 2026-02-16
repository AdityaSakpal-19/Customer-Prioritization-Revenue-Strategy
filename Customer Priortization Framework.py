#!/usr/bin/env python
# coding: utf-8

# In[5]:


# A. load and combine the data.
import pandas as pd

df_1 = pd.read_csv("online_retail_2009_2010.csv")
df_2 = pd.read_csv("online_retail_2010_2011.csv")

df_raw = pd.concat([df_1, df_2], ignore_index = True)

print(df_raw.shape)


# In[7]:


print(df_raw.head())  #Checking the data


# In[ ]:


#B. Cleaning and structuring of the data


# In[8]:


df_raw.columns


# In[9]:


#Normalize the columns
df_raw.columns = df_raw.columns.str.strip().str.replace(' ' , '')
df_raw.columns


# In[10]:


#Checking Missing Values
df_raw.isnull().sum().sort_values(ascending=False)


# In[11]:


#Removing the missing data from Customer ID
df = df_raw.dropna(subset=['CustomerID'])
print("After removing missing CustomerID:", df.shape)


# In[12]:


# Remove cancelled invoices
df = df[~df['Invoice'].astype(str).str.startswith('C')]

# remove negative quantities
df = df[df['Quantity'] > 0]

print("After removing cancellations/returns:", df.shape)


# In[13]:


#Remove Invalid Prices
df = df[df['Price'] > 0]
print("After removing invalid prices:", df.shape)


# In[14]:


#Convert Date Column
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True)


# In[15]:


#Checking min and max dates
df['InvoiceDate'].min(), df['InvoiceDate'].max()


# In[16]:


#Creating a revenue column
df['Revenue'] = df['Quantity'] * df['Price']
df[['Quantity','Price','Revenue']].head()


# In[17]:


#Quick Sanity checks
df['Revenue'].describe() 


# In[18]:


df.info()


# In[37]:


#C Building the Customer Master Table (RFM base)


# In[19]:


#Define Analysis Date
analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
analysis_date


# In[20]:


#Aggregate to Customer Level
customer_df = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (analysis_date - x.max()).days,  # Recency
    'Invoice': 'nunique',                                     # Frequency
    'Revenue': 'sum',                                         # Monetary
    'Quantity': 'sum'
}).reset_index()

customer_df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'TotalQuantity']


# In[21]:


#Checks
customer_df.head()


# In[22]:


#Saving Customer master data
customer_df.to_csv("customer_master.csv", index=False)


# In[1]:


# D. Pareto / 80-20 analysis


# In[23]:


#Sort customers by revenue
pareto_df = customer_df.sort_values(by='Monetary', ascending=False).reset_index(drop=True)
pareto_df.head()


# In[24]:


#Compute cumulative revenue
pareto_df['CumRevenue'] = pareto_df['Monetary'].cumsum()
total_revenue = pareto_df['Monetary'].sum()

pareto_df['CumRevenuePct'] = pareto_df['CumRevenue'] / total_revenue * 100


# In[25]:


#Customer percentage column
pareto_df['CustomerPct'] = (pareto_df.index + 1) / len(pareto_df) * 100


# In[26]:


#Find the 80% revenue point
pareto_80 = pareto_df[pareto_df['CumRevenuePct'] >= 80].iloc[0]
pareto_80[['CustomerPct', 'CumRevenuePct']]


# In[27]:


#Visualization of Pareto Analysis
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(pareto_df['CustomerPct'], pareto_df['CumRevenuePct'])
plt.axhline(80, color='red', linestyle='--')
plt.xlabel('% of Customers')
plt.ylabel('% of Revenue')
plt.title('Revenue Concentration (Pareto Analysis)')
plt.show()


# In[ ]:


#E. Create RFM Scores


# In[28]:


#Recency score (lower recency = better)
customer_df['R_Score'] = pd.qcut(customer_df['Recency'], 5, labels=[5,4,3,2,1])


# In[29]:


#Frequency score (higher = better)
customer_df['F_Score'] = pd.qcut(customer_df['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])


# In[30]:


#Monetary score (higher = better)
customer_df['M_Score'] = pd.qcut(customer_df['Monetary'], 5, labels=[1,2,3,4,5])


# In[31]:


#Combine into RFM code
customer_df['RFM'] = (
    customer_df['R_Score'].astype(str) +
    customer_df['F_Score'].astype(str) +
    customer_df['M_Score'].astype(str)
)


# In[32]:


#F. Create Business Segments
def segment_customer(row):
    r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])

    if m >=4 and r >=4:
        return "VIP Loyal"

    elif m >=4 and r <=2:
        return "High Value At Risk"

    elif m >=3 and f >=3:
        return "Growth Potential"

    elif r >=4 and m <=2:
        return "New/Low Spend Active"

    else:
        return "Low Priority"

customer_df['Segment'] = customer_df.apply(segment_customer, axis=1)


# In[33]:


#Check
customer_df['Segment'].value_counts()


# In[34]:


#G. Map Segments to Priority
priority_map = {
    "High Value At Risk": "P1 - Immediate Retention",
    "VIP Loyal": "P2 - Maintain Loyalty",
    "Growth Potential": "P3 - Upsell Opportunity",
    "New/Low Spend Active": "P4 - Nurture",
    "Low Priority": "P5 - Minimal Effort"
}

customer_df['Priority'] = customer_df['Segment'].map(priority_map)


# In[35]:


#Inspect Distribution
customer_df['Priority'].value_counts()


# In[37]:


#Revenue by Priority
priority_revenue = customer_df.groupby('Priority')['Monetary'].sum().sort_values(ascending=False)
priority_revenue

priority_revenue_pct = priority_revenue / priority_revenue.sum() * 100
priority_revenue_pct


# In[38]:


#H. Checking the Revenue Impact by Setting Business Assumptions
impact_rates = {
    "P1 - Immediate Retention": 0.35,
    "P2 - Maintain Loyalty": 0.10,
    "P3 - Upsell Opportunity": 0.15,
    "P4 - Nurture": 0.05,
    "P5 - Minimal Effort": 0.00
}


# In[39]:


#Calculate Revenue Impact per Customer
customer_df['ImpactRate'] = customer_df['Priority'].map(impact_rates)

customer_df['PotentialImpactRevenue'] = (
    customer_df['Monetary'] * customer_df['ImpactRate']
)


# In[42]:


#Total Business Impact
total_impact = customer_df['PotentialImpactRevenue'].sum()
total_revenue = customer_df['Monetary'].sum()

total_impact, total_revenue



# In[43]:


impact_percent = total_impact / total_revenue * 100
impact_percent


# In[44]:


#Impact by Priority
impact_by_priority = customer_df.groupby('Priority')['PotentialImpactRevenue'].sum().sort_values(ascending=False)
impact_by_priority


# In[45]:


#I. Target Customers lists


# In[49]:


customer_df['CustomerID'] = customer_df['CustomerID'].astype(int)


# In[50]:


#Immediate Retention List (P1)

retention_list = (
    customer_df[customer_df['Priority'] == "P1 - Immediate Retention"]
    .sort_values(by=['Monetary','Recency'], ascending=[False, False])
)

retention_list.head(20)


# In[51]:


#Loyalty Protection List (P2)
loyalty_list = (
    customer_df[customer_df['Priority'] == "P2 - Maintain Loyalty"]
    .sort_values(by='Monetary', ascending=False)
)

loyalty_list.head(20)


# In[52]:


#Upsell Opportunity List (P3)
upsell_list = (
    customer_df[customer_df['Priority'] == "P3 - Upsell Opportunity"]
    .sort_values(by='Frequency', ascending=False)
)

upsell_list.head(20)


# In[53]:


#Exporting Target Customers lists
cols = ['CustomerID','Segment','Priority','Recency','Frequency','Monetary']

retention_list[cols].to_csv("P1_retention_customers.csv", index=False)
loyalty_list[cols].to_csv("P2_loyalty_customers.csv", index=False)
upsell_list[cols].to_csv("P3_upsell_customers.csv", index=False)


# In[54]:


#Exporting new customer master table
customer_df.to_csv("customer_master_final.csv", index=False)

