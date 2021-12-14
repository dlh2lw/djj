#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
train = pd.read_csv(r'C:\Users\zmy2d\Desktop\未来销售预测\sales_train.csv')
test= pd.read_csv(r'C:\Users\zmy2d\Desktop\未来销售预测\test.csv')
train.head()


# In[ ]:


test.head()


# In[ ]:


print('训练集的商店数量： %d ，商品数量： %d；\n' % (train['shop_id'].unique().size, train['item_id'].unique().size),
     '测试的商店数量： %d，商品数量： %d。' % (test['shop_id'].unique().size, test['item_id'].unique().size))


# In[ ]:


test[~test['shop_id'].isin(train['shop_id'].unique())]


# In[ ]:


test[~test['item_id'].isin(train['item_id'].unique())]['item_id'].unique()[:10]


# In[ ]:


shops = pd.read_csv(r'C:\Users\zmy2d\Desktop\未来销售预测\shops.csv')
shops.head()


# In[ ]:


# 查看测试集是否包含了这几个商店
test[test['shop_id'].isin([39, 40, 10, 11, 0, 57, 58, 1, 12 ,56])]['shop_id'].unique()


# In[ ]:


shop_id_map = {11: 10, 0: 57, 1: 58, 40: 39}
train.loc[train['shop_id'].isin(shop_id_map), 'shop_id'] = train.loc[train['shop_id'].isin(shop_id_map), 'shop_id'].map(shop_id_map)
train.loc[train['shop_id'].isin(shop_id_map), 'shop_id']


# In[ ]:


train.loc[train['shop_id'].isin([39, 40, 10, 11, 0, 57, 58, 1]), 'shop_id'].unique()


# In[ ]:


shops['shop_city'] = shops['shop_name'].map(lambda x:x.split(' ')[0].strip('!'))
shop_types = ['ТЦ', 'ТРК', 'ТРЦ', 'ТК', 'МТРЦ']
shops['shop_type'] = shops['shop_name'].map(lambda x:x.split(' ')[1] if x.split(' ')[1] in shop_types else 'Others')
shops.loc[shops['shop_id'].isin([12, 56]), ['shop_city', 'shop_type']] = 'Online'  # 12和56号是网上商店
shops.head(13)


# In[ ]:


# 对商店信息进行编码，降低模型训练的内存消耗
shop_city_map = dict([(v,k) for k, v in enumerate(shops['shop_city'].unique())])
shop_type_map = dict([(v,k) for k, v in enumerate(shops['shop_type'].unique())])
shops['shop_city_code'] = shops['shop_city'].map(shop_city_map)
shops['shop_type_code'] = shops['shop_type'].map(shop_type_map)
shops.head(7)


# In[ ]:


items = pd.read_csv(r'C:\Users\zmy2d\Desktop\未来销售预测\items.csv')
items


# In[ ]:


# 数据集比较大，只分析有没有重复名称不同ID的商品
items['item_name'] = items['item_name'].map(lambda x: ''.join(x.split(' ')))  # 删除空格
duplicated_item_name = items[items['item_name'].duplicated()]
duplicated_item_name 


# In[ ]:


duplicated_item_name_rec = items[items['item_name'].isin(duplicated_item_name['item_name'])]  # 6个商品相同名字不同id的记录
duplicated_item_name_rec


# In[ ]:


test[test['item_id'].isin(duplicated_item_name_rec['item_id'])]['item_id'].unique()


# In[ ]:


old_id = duplicated_item_name_rec['item_id'].values[::2]
new_id = duplicated_item_name_rec['item_id'].values[1::2]
old_new_map = dict(zip(old_id, new_id))
old_new_map


# In[ ]:


train.loc[train['item_id'].isin(old_id), 'item_id'] = train.loc[train['item_id'].isin(old_id), 'item_id'].map(old_new_map)
train[train['item_id'].isin(old_id)]


# In[ ]:


train[train['item_id'].isin(duplicated_item_name_rec['item_id'].values)]['item_id'].unique()  # 旧id成功替换成新id


# In[ ]:


items.groupby('item_id').size()[items.groupby('item_id').size() > 1]  # 检查同一个商品是否分了不同类目


# In[ ]:


cat = pd.read_csv(r'C:\Users\zmy2d\Desktop\未来销售预测\item_categories.csv')
cat


# In[ ]:


cat[cat['item_category_name'].duplicated()]


# In[ ]:


cat['item_type'] = cat['item_category_name'].map(lambda x: 'Игры' if x.find('Игры ')>0 else x.split(' -')[0].strip('\"')) 
cat.iloc[[32, 33, 34, -3, -2, -1]]  # 有几个比较特殊，需要另外调整一下


# In[ ]:


cat.iloc[[32,-3, -2], -1] = ['Карты оплаты', 'Чистые носители', 'Чистые носители' ]
cat.iloc[[32,-3, -2]]


# In[ ]:


item_type_map = dict([(v,k) for k, v in enumerate(cat['item_type'].unique())])
cat['item_type_code'] = cat['item_type'].map(item_type_map)
cat.head()


# In[ ]:


cat['sub_type'] = cat['item_category_name'].map(lambda x: x.split('-',1)[-1]) 
cat


# In[ ]:


cat['sub_type'].unique()


# In[ ]:


sub_type_map = dict([(v,k) for k, v in enumerate(cat['sub_type'].unique())])
cat['sub_type_code'] = cat['sub_type'].map(sub_type_map)
cat.head()


# In[ ]:


items = items.merge(cat[['item_category_id', 'item_type_code', 'sub_type_code']], on='item_category_id', how='left')
items.head()


# In[ ]:


import gc
del cat
gc.collect()


# In[ ]:


sns.jointplot('item_cnt_day', 'item_price', train, kind='scatter')


# In[ ]:


train_filtered = train[(train['item_cnt_day'] < 800) & (train['item_price'] < 70000)].copy()
sns.jointplot('item_cnt_day', 'item_price', train_filtered, kind='scatter')


# In[ ]:


outer = train[(train['item_cnt_day'] > 400) | (train['item_price'] > 40000)]
outer


# In[ ]:


outer_set = train_filtered[train_filtered['item_id'].isin(outer['item_id'].unique())].groupby('item_id')
 
fig, ax = plt.subplots(1,1,figsize=(10, 10))
colors = sns.color_palette() + sns.color_palette('bright')  # 使用调色板。默认颜色只有10来种，会重复使用，不便于观察
i = 1
for name, group in outer_set:
    ax.plot(group['item_cnt_day'], group['item_price'], marker='o', linestyle='', ms=12, label=name, c=colors[i])
    i += 1
ax.legend()

plt.show()


# In[ ]:


train[train['item_id'].isin([13403,7238, 14173])]


# In[ ]:


train.loc[train['item_id']==13403].boxplot(['item_cnt_day', 'item_price'])


# In[ ]:


m_400 = train[(train['item_cnt_day'] > 400) & (train['item_cnt_day'] < 520)]['item_id'].unique()
n = m_400.size
fig, axes = plt.subplots(1,n,figsize=(n*4, 6))
for i in range(n):
    train[train['item_id'] == m_400[i]].boxplot(['item_cnt_day'], ax=axes[i])
    axes[i].set_title('Item%d' % m_400[i])
plt.show()


# In[ ]:


filtered = train[(train['item_cnt_day'] < 400) & (train['item_price'] < 45000)].copy()
filtered.head()


# In[ ]:


filtered.drop(index=filtered[filtered['item_id'].isin([7238, 14173])].index, inplace=True)
del train, train_filtered
gc.collect()


# In[ ]:


(filtered[['date_block_num', 'shop_id','item_id', 'item_price']] < 0).any()


# In[ ]:


# 商品单价小于0的情况
filtered[filtered['item_price'] <= 0]


# In[ ]:


filtered.groupby(['date_block_num','shop_id', 'item_id'])['item_price'].mean().loc[4, 32, 2973]


# In[ ]:


filtered.loc[filtered['item_price'] <= 0, 'item_price'] = 1249.0  # 用了同一个月同一个商店该商品的均价
filtered[filtered['item_price'] <= 0]  # 检查是否替换成功


# In[ ]:


# 下面也给出替换的函数
def clean_by_mean(df, keys, col):
    """
    用同一月份的均值替换小于等于0的值
    keys 分组键；col 需要替换的字段
    """
    group = df[df[col] <= 0]
    # group = df[df['item_price'] <= 0]
    mean_price = df.groupby(keys)[col].mean()
    # mean_price = df.groupby(['date_block_num', 'shop_id', 'item_id'])['item_price'].mean()
    for i, row in group.iterrows:
        record = group.loc[i]
        df.loc[i,col] = mean_price.loc[record[keys[0]], record[keys[1]], record[keys[2]]]
        # df.loc[i,'item_price'] = mean_price.loc[record['date_block_num'], record['shop_id'], record['item_id']]
    return df
# 添加日营业额
filtered['turnover_day'] = filtered['item_price'] * filtered['item_cnt_day']
filtered


# In[ ]:


item_sales_monthly = filtered.pivot_table(columns='item_id',
                                          index='date_block_num', 
                                          values='item_cnt_day',
                                          fill_value=0,
                                          aggfunc=sum)
item_sales_monthly.head()


# In[ ]:


fig, axes = plt.subplots(1,2, figsize=(20, 8))

item_sales_monthly.sum(1).plot(ax=axes[0], title='Total sales of each month', xticks=[i for i in range(0,34,2)])  # 每月总销量
item_sales_monthly.sum(0).plot(ax=axes[1], title='Total sales of each item')  # 每个商品的总销量
plt.subplots_adjust(wspace=0.2)


# In[ ]:


top_sales = item_sales_monthly.sum().sort_values(ascending=False)
top_sales


# In[ ]:


test[test['item_id'].isin(top_sales[top_sales<=0].index)]


# In[ ]:


top_sales.iloc[0] / item_sales_monthly.sum().sum() * 100  # 销量占比


# In[ ]:


item_sales_monthly[top_sales.index[0]].plot(kind='bar', figsize=(12,6))  # 每月销量


# In[ ]:


item_turnover_monthly = filtered.pivot_table(index= 'date_block_num',
                                               columns= 'item_id',
                                               values='turnover_day',
                                               fill_value=0,
                                               aggfunc=sum)
item_turnover_monthly.head()


# In[ ]:


item_sales_monthly = item_sales_monthly.drop(columns=top_sales[top_sales<=0].index, axis=1)  # 去掉销量为0和负值的商品
item_turnover_monthly = item_turnover_monthly.drop(columns=top_sales[top_sales<=0].index, axis=1)
total_turnover = item_turnover_monthly.sum().sum()
item_turnover_monthly[top_sales.index[0]].sum() / total_turnover * 100


# In[ ]:


items[items['item_id']==20949 ]


# In[ ]:


(item_sales_monthly > 0).sum(1).plot(figsize=(12, 6))


# In[ ]:


item_sales_monthly.sum(1).div((item_sales_monthly > 0).sum(1)).plot(figsize=(12, 6))
# 商品月总销量 / 当月在售商品数量 = 当月在售商品平均销量


# In[ ]:


fig, axes = plt.subplots(1,2, figsize=(20, 8))
item_turnover_monthly.sum(1).plot(ax=axes[0], title='Total turnovers of each month', xticks=[i for i in range(0,34,2)])  # 每月总营收
item_turnover_monthly.sum(0).plot(ax=axes[1], title='Total turnovers of each item')  # 每个商品的总营收
plt.subplots_adjust(wspace=0.2)


# In[ ]:


top_turnover = item_turnover_monthly.sum().sort_values(ascending=False)
top_turnover


# In[ ]:


item_turnover_monthly[top_turnover.index[0]].sum() / total_turnover * 100


# In[ ]:


item_sales_monthly[top_turnover.index[0]].sum() / item_sales_monthly.sum().sum() * 100


# In[ ]:


item_turnover_monthly[top_turnover.index[0]].plot(kind='bar', figsize=(12, 6))


# In[ ]:


item_turnover_monthly[top_turnover.index[0]].div(item_turnover_monthly.sum(1)).plot(figsize=(12, 6),xticks=[i for i in range(0,34,2)])


# In[ ]:


items[items['item_id']==top_turnover.index[0]]


# In[ ]:


turnover_monthly = item_turnover_monthly.sum(1)
sales_monthly = item_sales_monthly.sum(1)
fig, axe1 = plt.subplots(1, 1, figsize=(16, 6))
axe2 = axe1.twinx()
axe1.plot(turnover_monthly.index, turnover_monthly.values, c='r')

axe2.plot(sales_monthly.index, sales_monthly.values, c='b')
axe2.grid(c='c', alpha=0.3)
axe1.legend(['Monthly Turnover'],fontsize=13, bbox_to_anchor=(0.95, 1))
axe2.legend(['Monthly Sales'],fontsize=13, bbox_to_anchor=(0.93, 0.9))
axe1.set_ylabel('Monthly Turnover', c='r')
axe2.set_ylabel('Monthly Sales', c='b')
plt.show()


# In[ ]:


sales_growth = item_sales_monthly.loc[23].sum() - item_sales_monthly.loc[11].sum()
sales_growth_rate = sales_growth / item_sales_monthly.loc[11].sum() * 100
turnover_growth = item_turnover_monthly.loc[23].sum() - item_turnover_monthly.loc[11].sum()
turnover_growth_rate = turnover_growth / item_turnover_monthly.loc[11].sum() * 100
print(
    ' 销售同比增长量为： %.2f ，同比增长率为： %.2f%%;\n' % (sales_growth, sales_growth_rate),
    '营收同比增长量为： %.2f ，同比增长率为： %.2f%%。' % (turnover_growth, turnover_growth_rate)
     )


# In[ ]:


dec_set = item_turnover_monthly.loc[[11, 23]]
dec_set


# In[ ]:


shop_sales_monthly = filtered.pivot_table(index='date_block_num',
                                          columns='shop_id',
                                          values='item_cnt_day',
                                          fill_value=0,
                                          aggfunc=sum)
shop_open_month_cnt = (shop_sales_monthly.iloc[-6:] >  0).sum()  # 有销量的记录
shop_open_month_cnt.head()  # 每个店铺最后半年里有几个月有销量


# In[ ]:


shop_c_n = shop_sales_monthly[shop_open_month_cnt[shop_open_month_cnt < 6].index]
shop_c_n.tail(12)
# 最后半年经营月数少于6个月的店铺


# In[ ]:


open_shop = shop_sales_monthly[shop_open_month_cnt[shop_open_month_cnt == 6].index]
open_shop.tail(7) # 最后半年都正常经营的商店


# In[ ]:


item_selling_month_cnt = (item_sales_monthly.iloc[-6:] >  0).sum() 
item_selling_month_cnt.head()  # 这些商品在最后半年有几个月有销量


# In[ ]:


item_zero = item_sales_monthly[item_selling_month_cnt[item_selling_month_cnt == 0].index]
# 这些商品在最后半年都没有销量
item_zero.tail(12)


# In[ ]:


selling_item = item_sales_monthly[item_selling_month_cnt[item_selling_month_cnt > 0].index]
selling_item.tail(12)  # 最后半年有销量的商品


# In[ ]:


cl_set = filtered[filtered['shop_id'].isin(open_shop.columns) & filtered['item_id'].isin(selling_item.columns)]
cl_set


# In[ ]:


from itertools import product
import time
ts = time.time()
martix = []
for i in range(34):
    record = cl_set[cl_set['date_block_num'] == i]
    group = product([i],record.shop_id.unique(),record.item_id.unique())
    martix.append(np.array(list(group)))
            
cols = ['date_block_num', 'shop_id', 'item_id']
martix = pd.DataFrame(np.vstack(martix), columns=cols)

martix


# In[ ]:


from itertools import product
import time
ts = time.time()
martix = []
for i in range(34):
    record = filtered[filtered['date_block_num'] == i]
    group = product([i],record.shop_id.unique(),record.item_id.unique())
    martix.append(np.array(list(group)))
            
cols = ['date_block_num', 'shop_id', 'item_id']
martix = pd.DataFrame(np.vstack(martix), columns=cols)

martix


# In[ ]:


del cl_set
gc.collect()


# In[ ]:


group = filtered.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': np.sum})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)
group


# In[ ]:


martix = pd.merge(martix, group, on=['date_block_num', 'shop_id', 'item_id'], how='left')
martix.head()


# In[ ]:


test['date_block_num'] = 34
test['item_cnt_month'] = 0
martix = pd.concat([martix.fillna(0), test.drop(columns='ID')], sort=False, ignore_index=True, keys=['date_block_num','shop_id','item_id'])
martix


# In[ ]:


martix = martix.merge(shops[['shop_id', 'shop_type_code', 'shop_city_code']], on='shop_id', how='left')
martix = martix.merge(items.drop(columns='item_name'), on='item_id', how='left')
martix


# In[ ]:


martix['year'] =  martix['date_block_num'].map(lambda x: x // 12 + 2013)
martix['month'] = martix['date_block_num'].map(lambda x: x % 12)
martix.head()


# In[ ]:


# 商品 月销量均值
group = martix.groupby(['date_block_num','item_id']).agg({'item_cnt_month':'mean'})
group.columns = ['item_cnt_month_avg']
group.reset_index(inplace=True)
martix = martix.merge(group, on=['date_block_num', 'item_id'], how='left')
martix.head()


# In[ ]:


# 商店 月销量均值
group = martix.groupby(['date_block_num','shop_id']).agg({'item_cnt_month':'mean'})
group.columns = ['shop_cnt_month_avg']
group.reset_index(inplace=True)
martix = martix.merge(group, on=['date_block_num', 'shop_id'], how='left')
martix.head()


# In[ ]:


# 类别 月销量均值
group = martix.groupby(['date_block_num','item_category_id']).agg({'item_cnt_month':'mean'})
group.columns = ['cat_cnt_month_avg']
group.reset_index(inplace=True)
martix = martix.merge(group, on=['date_block_num', 'item_category_id'], how='left')
martix.head()


# In[ ]:


# 商店-类别 月销量均值
group = martix.groupby(['date_block_num','shop_id','item_category_id']).agg({'item_cnt_month':'mean'})
group.columns = ['shop_cat_cnt_month_avg']
group.reset_index(inplace=True)
martix = martix.merge(group, on=['date_block_num','shop_id','item_category_id'], how='left')
martix.head()


# In[ ]:


# 大类 月销量均值
group = martix.groupby(['date_block_num', 'item_type_code']).agg({'item_cnt_month':'mean'})
group.columns = ['itemtype_cnt_month_avg']
group.reset_index(inplace=True)
martix = martix.merge(group, on=['date_block_num', 'item_type_code'], how='left')
martix.head()


# In[ ]:


# 小类 月销量均值
group = martix.groupby(['date_block_num', 'sub_type_code']).agg({'item_cnt_month':'mean'})
group.columns = ['subtype_cnt_month_avg']
group.reset_index(inplace=True)
martix = martix.merge(group, on=['date_block_num','sub_type_code'], how='left')
martix.head()


# In[ ]:


# 城市-商品 月销量均值
group = martix.groupby(['date_block_num','shop_city_code','item_id']).agg({'item_cnt_month':'mean'})
group.columns = ['city_item_cnt_month_avg']
group.reset_index(inplace=True)
martix = martix.merge(group, on=['date_block_num','shop_city_code','item_id'], how='left')
martix.head()


# In[ ]:


# 商店类型-商品 月销量均值
group = martix.groupby(['date_block_num','shop_type_code','item_id']).agg({'item_cnt_month':'mean'})
group.columns = ['shoptype_item_cnt_month_avg']
group.reset_index(inplace=True)
martix = martix.merge(group, on=['date_block_num','shop_type_code','item_id'], how='left')
martix.head()


# In[106]:


del group
gc.collect()


# In[ ]:


def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df
martix = lag_feature(martix, [1,2,3,6,12], 'item_cnt_month')
martix.head()


# In[ ]:


martix = lag_feature(martix, [1,2,3,6,12], 'item_cnt_month_avg')
martix = lag_feature(martix, [1,2,3,6,12], 'shop_cnt_month_avg')
martix.head()


# In[ ]:


martix.drop(columns=[ 'item_cnt_month_avg', 'shop_cnt_month_avg'], inplace=True)  # 只保留特征的历史信息
gc.collect()


# In[ ]:


martix = lag_feature(martix, [1,2,3,6,12], 'cat_cnt_month_avg')
martix = lag_feature(martix, [1,2,3,6,12], 'shop_cat_cnt_month_avg')
martix.head()


# In[ ]:


for col in train_set.columns:
    if col.find('code') >= 0:
        train_set[col] = train_set[col].astype(np.int8)
    elif train_set[col].dtype == 'float64':
        train_set[col] = train_set[col].astype(np.float32)
    elif train_set[col].dtype == 'int64':
        train_set[col] = train_set[col].astype(np.int16)
        
train_set['item_type_code'] = train_set['item_type_code'].astype('category')
train_set['sub_type_code'] = train_set['sub_type_code'].astype('category')
train_set.info()


# In[ ]:




