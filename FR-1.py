'''
1.探索性分析所有顧客資料的特徵(視覺化呈現)，並處理資料的格式與問題(例如: 遺失值等)。
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)
# pd.set_option('display.width',1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
plt.rc('font', family='Microsoft JhengHei') # 微軟正黑體
plt.rcParams['axes.unicode_minus'] = False # 正常顯示負號

# original data
df = pd.read_csv('customer_data_utf8.csv' ,encoding='utf-8')
# print(df.shape)             
# print(df.head())

print(df.info())
# print(df.isnull().sum()) # 空值個數
# missing_percentage = df.isnull().mean() * 100        # 缺失值的百分比
# print(missing_percentage)

# 數值型態描述
# print(df.describe())
# 字串型態描述
# print(df.describe(include=['object']))

df['優惠方式'] = df['優惠方式'].fillna('No Discount')

df['網路連線類型'] = df['網路連線類型'].fillna('No Type') # 處理1526筆空值，未訂閱網路服務，則為「無」
df['平均下載量( GB)'] = df['平均下載量( GB)'].fillna(0) # 未訂閱網路服務，則為 0
df['線上安全服務'] = df['線上安全服務'].fillna('No') # 未訂閱網路服務，則為否  以下同理
df['線上備份服務'] = df['線上備份服務'].fillna('No') 
df['設備保護計劃'] = df['設備保護計劃'].fillna('No')  
df['技術支援計劃'] = df['技術支援計劃'].fillna('No') 
df['電視節目'] = df['電視節目'].fillna('No') 
df['電影節目'] = df['電影節目'].fillna('No') 
df['音樂節目'] = df['音樂節目'].fillna('No') 
df['無限資料下載'] = df['無限資料下載'].fillna('No') 
df['客戶流失類別'] = df['客戶流失類別'].fillna('None')
df['客戶離開原因'] = df['客戶離開原因'].fillna('None')

df['平均長途話費'] = df['平均長途話費'].fillna(0) # 處理682筆空值，未訂閱家庭電話服務，則為 0
df['多線路服務'] = df['多線路服務'].fillna('No') # 未訂閱家庭電話服務，則為 'No'

# 每月費用、總費用、總退款    這三個一起看
# vars = ['每月費用', '總費用', '總退款']  # 先觀察圖形，找出特別明顯的異常值，再進一步處理
# numeric_df = df[vars]
# sns.pairplot(numeric_df, markers='o')
# plt.show()

# Colab 有用Facet來看，負值只有每月費用
negative_monthly_costs = df[df['每月費用'] < 0]
print(negative_monthly_costs) # 列出異常值
print(negative_monthly_costs.shape) # (120 ,38)
df = df[df['每月費用'] > 0] # 刪除異常值，只保留正常值

# vars = ['每月費用', '總費用', '總退款']
# numeric_df = df[vars]
# sns.pairplot(numeric_df, markers='o')
# plt.show()

df.to_csv('customer_data_handled.csv', index=False, encoding='utf_8_sig') # 輸出處理過的資料
df = pd.read_csv('customer_data_handled.csv' ,encoding='utf-8') 
print(df.shape) # (6923, 38)
# 數值型態描述
# print(df.describe())
# 字串型態描述
# print(df.describe(include=['object']))

'''
欄位說明 (1/4)
'''
# # 年齡
# age_counts = df['年齡'].value_counts()
# print("年齡的個數:")
# print(age_counts)

# plt.figure(figsize=(10, 6))
# sns.histplot(df['年齡'], bins=20 , color='skyblue')
# plt.title('年齡直方圖')
# plt.xlabel('年齡')
# plt.ylabel('計數')
# plt.show()

# # 分組為不同年齡段
# age_bins = [20, 30, 40, 50, 60, 70, 80, 90]
# age_labels = ['19-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']

# # 將年齡資料分組
# df['年齡分組'] = pd.cut(df['年齡'], bins=age_bins, labels=age_labels, right=False)

# # # 計算每個年齡段的人數
# age_group_counts = df['年齡分組'].value_counts().sort_index()

# plt.figure(figsize=(10, 6))
# sns.barplot(x=age_group_counts.index, y=age_group_counts.values, palette='viridis')
# plt.title('年齡分組計數圖')
# plt.xlabel('年齡段')
# plt.ylabel('計數')
# plt.show()

# # 扶養人數
# support_series = df['扶養人數']
# value_counts = support_series.value_counts()
# print(value_counts)

# plt.figure(figsize=(10, 6))
# sns.countplot(x='扶養人數', data=df, palette='muted')
# plt.title('扶養人數計數圖')
# plt.xlabel('扶養人數')
# plt.ylabel('計數')
# plt.show()

def plot_categorical_feature_counts(df, feature):
    # Calculate value counts
    feature_counts = df[feature].value_counts()

    # Print counts for each category
    for category, count in feature_counts.items():
        print(f"{category} 的個數: {count}")

    # Plot count plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x=feature, data=df, palette='Set2')
    plt.title(f'{feature}計數圖')
    plt.xlabel(feature)
    plt.ylabel('計數')

    # Plot pie chart
    plt.subplot(1, 2, 2)
    colors = ['lightskyblue', 'lightcoral']
    plt.pie(feature_counts, labels=feature_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title(f'{feature}圓餅圖')
    plt.show()

def plot_categorical_features(df, use_columns):
    for feature in use_columns:
        plot_categorical_feature_counts(df, feature)

use_columns = ['性別', '婚姻', '優惠方式', '電話服務', '網路服務', '多線路服務',
               '網路連線類型', '線上安全服務', '線上備份服務', '設備保護計劃',
               '技術支援計劃', '電視節目', '電影節目', '音樂節目', '無限資料下載',
               '合約類型', '無紙化計費', '支付帳單方式']
plot_categorical_features(df, use_columns)

  
# 城市
# city_counts = df['城市'].value_counts()
# print(city_counts)
# # 城市名稱太多，好像不好畫圖，故只取前10個城市
# top_cities = df['城市'].value_counts().head(10)

# plt.figure(figsize=(12, 8))
# sns.barplot(x=top_cities.values, y=top_cities.index, palette='Set2')
# plt.title('前 10 項城市計數圖')
# plt.xlabel('計數')
# plt.ylabel('城市')
# plt.show()

# 推薦次數
# recommendation_counts = df['推薦次數'].value_counts()
# print(recommendation_counts)

# plt.figure(figsize=(10, 6))
# sns.barplot(x=recommendation_counts.index, y=recommendation_counts.values, palette='Blues_r')
# plt.title('推薦次數計數圖')
# plt.xlabel('推薦次數')
# plt.ylabel('計數')
# plt.show()

# 加入期間 (月)
# join_counts = df['加入期間 (月)'].value_counts()
# print(join_counts)

# plt.figure(figsize=(10, 6))
# sns.histplot(df['加入期間 (月)'], bins=20, color='purple')
# plt.title('加入期間分布直方圖')
# plt.xlabel('加入期間 (月)')
# plt.ylabel('計數')
# plt.show()

# 優惠方式
discount_counts = df['優惠方式'].value_counts()
print(discount_counts)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='優惠方式', data=df, palette='muted')
plt.title('優惠方式計數圖')
plt.xlabel('優惠方式')
plt.ylabel('計數')
plt.xticks(rotation=25, ha='right')  # 如果x軸的標籤太長，可以選擇旋轉標籤以避免重疊

plt.subplot(1, 2, 2)
plt.pie(discount_counts, labels=discount_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('優惠方式比例圖')
plt.show()

# 平均長途話費
# plt.figure(figsize=(10, 6))
# sns.histplot(df['平均長途話費'], bins=10, kde=False, color='skyblue')
# plt.title('平均長途話費分布直方圖')
# plt.xlabel('平均長途話費')
# plt.ylabel('計數')
# plt.xticks(range(0, 51, 5))
# plt.show()

# 平均下載量( GB)
# plt.figure(figsize=(10, 6))
# sns.histplot(df['平均下載量( GB)'], bins=10, kde=False, color='skyblue')
# plt.title('平均下載量分布直方圖')
# plt.xlabel('平均下載量( GB)')
# plt.ylabel('計數')
# plt.xticks(range(0, 95, 5))
# plt.show()

# 額外數據費用、額外長途費用
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.hist(df['額外數據費用'], bins=20, color='skyblue', alpha=0.7)
# plt.title('額外數據費用分布直方圖')
# plt.xlabel('額外數據費用')
# plt.ylabel('計數')

# plt.subplot(1, 2, 2)
# plt.hist(df['額外長途費用'], bins=20, color='lightcoral', alpha=0.7)
# plt.title('額外長途費用分布直方圖')
# plt.xlabel('額外長途費用')
# plt.ylabel('計數')
# plt.tight_layout()
# plt.show()

# 總收入
# plt.figure(figsize=(8, 6))
# plt.hist(df['總收入'], bins=20, color='orange', alpha=0.7)
# plt.title('總收入分布直方圖')
# plt.xlabel('總收入')
# plt.ylabel('計數')
# plt.xticks(range(0, 13500, 1000))
# plt.show()

## 客戶狀態
# plt.figure(figsize=(8, 6))
# sns.countplot(x='客戶狀態', data=df, palette='muted')
# plt.title('客戶狀態計數圖')
# plt.xlabel('客戶狀態')
# plt.ylabel('計數')
# plt.show()

# # 客戶流失類別、客戶離開原因
# df_customer_left = df[['客戶流失類別', '客戶離開原因']]

# customer_churn_counts = df['客戶流失類別'].value_counts()
# print("客戶流失類別計數:\n", customer_churn_counts)
# customer_leave_reason_counts = df['客戶離開原因'].value_counts()
# print("\n客戶離開原因計數:\n", customer_leave_reason_counts)

# # df_customer_left = df[['客戶流失類別', '客戶離開原因']].dropna()
# # print(df_customer_left.shape) # (1839, 2)

# plt.figure(figsize=(15, 8))
# sns.countplot(x='客戶離開原因', hue='客戶流失類別', data=df_customer_left)
# plt.title('客戶離開原因分佈圖')
# plt.xlabel('客戶離開原因' ,fontsize=4)
# plt.ylabel('計數')
# plt.xticks(rotation=25, ha='right')  # 如果x軸的標籤太長，可以選擇旋轉標籤以避免重疊
# plt.show()
