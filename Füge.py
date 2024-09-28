###################################################
# Rating Products
###################################################

# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating


############################################
# Uygulama: Kullanıcı ve Zaman Ağırlıklı Kurs Puanı Hesaplama
############################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# (50+ Saat) Python A-Z™: Veri Bilimi ve Machine Learning
# Puan: 4.8 (4.764925)
# Toplam Puan: 4611
# Puan Yüzdeleri: 75, 20, 4, 1, <1
# Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6

df = pd.read_csv("datasets/course_reviews.csv")
df.head()
df.shape

# rating dagılımı
df["Rating"].value_counts()

df["Questions Asked"].value_counts()

df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating": "mean"})


df.head()

####################
# Average
####################

# Ortalama Puan
df["Rating"].mean()

####################
# Time-Based Weighted Average
####################
# Puan Zamanlarına Göre Ağırlıklı Ortalama

df.head()
df.info()

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

current_date = pd.to_datetime('2021-02-10 0:0:0')

df["days"] = (current_date - df["Timestamp"]).dt.days

df.loc[df["days"] <= 30, "Rating"].mean()

df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()

df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()

df.loc[(df["days"] > 180), "Rating"].mean()


df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
    df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
    df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
    df.loc[(df["days"] > 180), "Rating"].mean() * 22/100

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100

time_based_weighted_average(df)

time_based_weighted_average(df, 30, 26, 22, 22)



####################
# User-Based Weighted Average
####################

df.head()

df.groupby("Progress").agg({"Rating": "mean"})

df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
    df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100



def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100


user_based_weighted_average(df, 20, 24, 26, 30)


####################
# Weighted Rating
####################

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w/100 + user_based_weighted_average(dataframe)*user_w/100

course_weighted_rating(df)

course_weighted_rating(df, time_w=40, user_w=60)






--------------------------------------------------------------


###################################################
# Sorting Products
###################################################

###################################################
# Uygulama: Kurs Sıralama
###################################################
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/product_sorting.csv")
print(df.shape)
df.head(10)

####################
# Sorting by Rating
####################

df.sort_values("rating", ascending=False).head(20)

####################
# Sorting by Comment Count or Purchase Count
####################

df.sort_values("purchase_count", ascending=False).head(20)
df.sort_values("commment_count", ascending=False).head(20)

####################
# Sorting by Rating, Comment and Purchase
####################

df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])

df.describe().T

df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])

(df["comment_count_scaled"] * 32 / 100 +
 df["purchase_count_scaled"] * 26 / 100 +
 df["rating"] * 42 / 100)


def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["comment_count_scaled"] * w1 / 100 +
            dataframe["purchase_count_scaled"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)


df["weighted_sorting_score"] = weighted_sorting_score(df)

df.sort_values("weighted_sorting_score", ascending=False).head(20)

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)


####################
# Bayesian Average Rating Score
####################

# Sorting Products with 5 Star Rated
# Sorting Products According to Distribution of 5 Star Rating

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


df.head()

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)

df.sort_values("weighted_sorting_score", ascending=False).head(20)
df.sort_values("bar_score", ascending=False).head(20)

df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending=False)


####################
# Hybrid Sorting: BAR Score + Diğer Faktorler
####################

# Rating Products
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating
# - Bayesian Average Rating Score

# Sorting Products
# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated)
# - Hybrid Sorting: BAR Score + Diğer Faktorler


def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                     "3_point",
                                                                     "4_point",
                                                                     "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)

    return bar_score*bar_w/100 + wss_score*wss_w/100


df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.sort_values("hybrid_sorting_score", ascending=False).head(20)

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head(20)


############################################
# Uygulama: IMDB Movie Scoring & Sorting
############################################

import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/movies_metadata.csv",
                 low_memory=False)  # DtypeWarning kapamak icin

df = df[["title", "vote_average", "vote_count"]]

df.head()
df.shape

########################
# Vote Average'a Göre Sıralama
########################

df.sort_values("vote_average", ascending=False).head(20)

df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T

df[df["vote_count"] > 400].sort_values("vote_average", ascending=False).head(20)

from sklearn.preprocessing import MinMaxScaler

df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)). \
    fit(df[["vote_count"]]). \
    transform(df[["vote_count"]])

########################
# vote_average * vote_count
########################

df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

df.sort_values("average_count_score", ascending=False).head(20)


########################
# IMDB Weighted Rating
########################


# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)

# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)

# Film 1:
# r = 8
# M = 500
# v = 1000

# (1000 / (1000+500))*8 = 5.33


# Film 2:
# r = 8
# M = 500
# v = 3000

# (3000 / (3000+500))*8 = 6.85

# (1000 / (1000+500))*9.5

# Film 1:
# r = 8
# M = 500
# v = 1000

# Birinci bölüm:
# (1000 / (1000+500))*8 = 5.33

# İkinci bölüm:
# 500/(1000+500) * 7 = 2.33

# Toplam = 5.33 + 2.33 = 7.66


# Film 2:
# r = 8
# M = 500
# v = 3000

# Birinci bölüm:
# (3000 / (3000+500))*8 = 6.85

# İkinci bölüm:
# 500/(3000+500) * 7 = 1

# Toplam = 7.85

M = 2500
C = df['vote_average'].mean()

def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)

df.sort_values("average_count_score", ascending=False).head(10)

weighted_rating(7.40000, 11444.00000, M, C)

weighted_rating(8.10000, 14075.00000, M, C)

weighted_rating(8.50000, 8358.00000, M, C)

df["weighted_rating"] = weighted_rating(df["vote_average"],
                                        df["vote_count"], M, C)

df.sort_values("weighted_rating", ascending=False).head(10)

####################
# Bayesian Average Rating Score
####################

# 12481                                    The Dark Knight
# 314                             The Shawshank Redemption
# 2843                                          Fight Club
# 15480                                          Inception
# 292                                         Pulp Fiction



def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])

bayesian_average_rating([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905])

df = pd.read_csv("datasets/imdb_ratings.csv")
df = df.iloc[0:, 1:]


df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis=1)
df.sort_values("bar_score", ascending=False).head(20)


# Weighted Average Ratings
# IMDb publishes weighted vote averages rather than raw data averages.
# The simplest way to explain it is that although we accept and consider all votes received by users,
# not all votes have the same impact (or ‘weight’) on the final rating.

# When unusual voting activity is detected,
# an alternate weighting calculation may be applied in order to preserve the reliability of our system.
# To ensure that our rating mechanism remains effective,
# we do not disclose the exact method used to generate the rating.
#
# See also the complete FAQ for IMDb ratings.



----------------------------------------------------------------------

############################################
# SORTING REVIEWS
############################################

import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# Up-Down Diff Score = (up ratings) − (down ratings)
###################################################

# Review 1: 600 up 400 down total 1000
# Review 2: 5500 up 4500 down total 10000

def score_up_down_diff(up, down):
    return up - down

# Review 1 Score:
score_up_down_diff(600, 400)

# Review 2 Score
score_up_down_diff(5500, 4500)

###################################################
# Score = Average rating = (up ratings) / (all ratings)
###################################################

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

score_average_rating(600, 400)
score_average_rating(5500, 4500)

# Review 1: 2 up 0 down total 2
# Review 2: 100 up 1 down total 101

score_average_rating(2, 0)
score_average_rating(100, 1)


###################################################
# Wilson Lower Bound Score
###################################################

# 600-400
# 0.6
# 0.5 0.7
# 0.5



def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


wilson_lower_bound(600, 400)
wilson_lower_bound(5500, 4500)

wilson_lower_bound(2, 0)
wilson_lower_bound(100, 1)


###################################################
# Case Study
###################################################

up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]
comments = pd.DataFrame({"up": up, "down": down})



# score_pos_neg_diff
comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"],
                                                                             x["down"]), axis=1)

# score_average_rating
comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"], x["down"]), axis=1)

# wilson_lower_bound
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)



comments.sort_values("wilson_lower_bound", ascending=False)




------------------------------------------------------



######################################################
# Temel İstatistik Kavramları
######################################################

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


############################
# Sampling (Örnekleme)
############################

populasyon = np.random.randint(0, 80, 10000)
populasyon.mean()

np.random.seed(115)

orneklem = np.random.choice(a=populasyon, size=100)
orneklem.mean()


np.random.seed(10)
orneklem1 = np.random.choice(a=populasyon, size=100)
orneklem2 = np.random.choice(a=populasyon, size=100)
orneklem3 = np.random.choice(a=populasyon, size=100)
orneklem4 = np.random.choice(a=populasyon, size=100)
orneklem5 = np.random.choice(a=populasyon, size=100)
orneklem6 = np.random.choice(a=populasyon, size=100)
orneklem7 = np.random.choice(a=populasyon, size=100)
orneklem8 = np.random.choice(a=populasyon, size=100)
orneklem9 = np.random.choice(a=populasyon, size=100)
orneklem10 = np.random.choice(a=populasyon, size=100)

(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean()
 + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10


############################
# Descriptive Statistics (Betimsel İstatistikler)
############################

df = sns.load_dataset("tips")
df.describe().T

############################
# Confidence Intervals (Güven Aralıkları)
############################

# Tips Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
df = sns.load_dataset("tips")
df.describe().T

df.head()

sms.DescrStatsW(df["total_bill"]).tconfint_mean()
sms.DescrStatsW(df["tip"]).tconfint_mean()

# Titanic Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
df = sns.load_dataset("titanic")
df.describe().T
sms.DescrStatsW(df["age"].dropna()).tconfint_mean()

sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()


######################################################
# Correlation (Korelasyon)
######################################################


# Bahşiş veri seti:
# total_bill: yemeğin toplam fiyatı (bahşiş ve vergi dahil)
# tip: bahşiş
# sex: ücreti ödeyen kişinin cinsiyeti (0=male, 1=female)
# smoker: grupta sigara içen var mı? (0=No, 1=Yes)
# day: gün (3=Thur, 4=Fri, 5=Sat, 6=Sun)
# time: ne zaman? (0=Day, 1=Night)
# size: grupta kaç kişi var?

df = sns.load_dataset('tips')
df.head()

df["total_bill"] = df["total_bill"] - df["tip"]

df.plot.scatter("tip", "total_bill") # Yüksek bahşiş alanların toplam fiyatı yüksek oluyor.
plt.show()

df["tip"].corr(df["total_bill"])

df["total_bill"].corr(df["tip"])


######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)    ort-dönüsüm-oranlar
######################################################
###bu yol:ortalamalara yönelik ise
# 1. Hipotezleri Kur  örn:iki grup ort aras fark yoktur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı    ilgili gruplarin dagiliminin normalligi
#   - 2. Varyans Homojenliği    varyanslarinin dagiliminin benzer olmasi
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direk 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.


############################
# Uygulama 1: Sigara İçenler ile İçmeyenlerin Hesap Ortalamaları Arasında İst Ol An Fark var mı?
############################


df = sns.load_dataset("tips")
df.head()

df.groupby("smoker").agg({"total_bill": "mean"})


df.groupby("smoker").agg({"tip": "mean"})

############################
# 1. Hipotezi Kur
############################

# H0: M1 = M2  (fark yoktur demektir)
# H1: M1 != M2

############################
# 2. Varsayım Kontrolü
############################

# Normallik Varsayımı    :bir degiskenin dagiliminin standart normal olup olmadiginin hipotes testidir.
# Varyans Homojenliği varsayimi:

############################
# Normallik Varsayımı
############################

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.


test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"]) #normallik testidir
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.


test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


############################
# Varyans Homojenligi Varsayımı
############################

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

############################
# 3 ve 4. Hipotezin Uygulanması
############################

# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

############################bu saglanmadi ama saglanmis gibi hesapliyalim####
# 1.1 Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
############################

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
##ttest: normallik saglaniyorsa beni kullanabilirsin-
# iki varsayimda saglaniyorsa kullanabilirsin-
# normallik saglaniyor homojenlik saglanmiyorsadakullanabilirsin ama False gir der böylece arka tarafta welch testini yapar---


# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

############################bu olayda en basta  sartlar saglanmadigi icin bu yapilmali

# 1.2 Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)--ortalama -medyan kiyaslama testidir
############################

test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))






############################
# Uygulama 2: Titanic Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. var mıdır?
############################

df = sns.load_dataset("titanic")
df.head()

df.groupby("sex").agg({"age": "mean"})


# 1. Hipotezleri kur:
# H0: M1  = M2 (Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. Yoktur)
# H1: M1! = M2 (... vardır)


# 2. Varsayımları İncele

# Normallik varsayımı
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır


test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))   #birinci grup icin normallik saglaniyor mu? 0,007 cikti h0 reddedilir

test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))  #ho reddilir 0,05 ten kucuk oldugundan

# Varyans homojenliği   degiskenlerin dagilimlarinin birbirlerine benzerligini degerlendiriyor
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))   #0,9 ho kabul edilir varyanslar homojen

# Varsayımlar sağlanmadığı için nonparametrik

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) #0,02 ho fark yoktur diyordu.0,05 ten kucuk oldugundan reddilir





#fark vardir deriz



# 90 280




############################
# Uygulama 3: Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark var mıdır?
############################

df = pd.read_csv("datasets/diabetes.csv")
df.head()

df.groupby("Outcome").agg({"Age": "mean"})

# 1. Hipotezleri kur
# H0: M1 = M2
# Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark Yoktur
# H1: M1 != M2
# .... vardır.

# 2. Varsayımları İncele

# Normallik Varsayımı (H0: Normal dağılım varsayımı sağlanmaktadır.)
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))



# Normallik varsayımı sağlanmadığı için nonparametrik.(buna medyan-siralama  kiyasida denir)

# Hipotez (H0: M1 = M2)
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Ho reddedilir 0,05 ten kucuk oldugundan
#yas ile diyabet arasinda anlamli bir fark vardir  --bunlar ortalama testiydi simdiye kadar







###################################################
# İş Problemi: Kursun Büyük Çoğunluğunu İzleyenler ile İzlemeyenlerin Puanları Birbirinden Farklı mı?
###################################################

# H0: M1 = M2 (... iki grup ortalamaları arasında ist olarak .anlamli bir .fark yoktur.)
# H1: M1 != M2 (...vardır)

df = pd.read_csv("datasets/course_reviews.csv")
df.head()
df['Rating'].mean()
df[(df["Progress"] > 75)]["Rating"].mean()

df[(df["Progress"] < 25)]["Rating"].mean()

df[(df["Progress"]<10)]["Rating"].mean()


test_stat, pvalue = shapiro(df[(df["Progress"] > 75)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


test_stat, pvalue = shapiro(df[(df["Progress"] < 25)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
                                 df[(df["Progress"] < 25)]["Rating"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#kursu daha fazla izleyenlerin puan ortalamasi daha yuksek deriz-- ho reddedilir






######################################################
# AB Testing (İki Örneklem Oran Testi)
######################################################

# H0: p1 = p2
# Yeni Tasarımın Dönüşüm Oranı ile Eski Tasarımın Dönüşüm Oranı Arasında İst. Ol. Anlamlı Farklılık Yoktur.
# H1: p1 != p2
# ... vardır
###örnek sayisi 30 dan buyuk n>30 olmali
basari_sayisi = np.array([300, 250])
gozlem_sayilari = np.array([1000, 1100])

proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari)
#0.0001532232957772221<0,05  ho reddedilir
#iki oran arasinda anlamli bir farklilik vardrir deriz

basari_sayisi / gozlem_sayilari


############################
# Uygulama: Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Farklılık var mıdır?
############################

# H0: p1 = p2
# Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Fark yoktur

# H1: p1 != p2
# .. vardır

df = sns.load_dataset("titanic")
df.head()

df.loc[df["sex"] == "female", "survived"].mean()

df.loc[df["sex"] == "male", "survived"].mean()

female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()

test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#ho reddedilir..fark vardir



######################################################
# ANOVA (Analysis of Variance)
######################################################

# İkiden fazla grup ortalamasını karşılaştırmak için kullanılır.
#sig. anlamlilik degeri =p velue degeri

df = sns.load_dataset("tips")
df.head()

df.groupby("day")["total_bill"].mean()

# 1. Hipotezleri kur

# HO: m1 = m2 = m3 = m4
# Grup ortalamaları arasında fark yoktur.

# H1: .. fark vardır

# 2. Varsayım kontrolü

# Normallik varsayımı
# Varyans homojenliği varsayımı

# Varsayım sağlanıyorsa one way anova
# Varsayım sağlanmıyorsa kruskal    (nunparametrik)

# H0: Normal dağılım varsayımı sağlanmaktadır.

for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, 'p-value: %.4f' % pvalue)


# H0: Varyans homojenliği varsayımı sağlanmaktadır.

test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


# 3. Hipotez testi ve p-value yorumu

# Hiç biri sağlamıyor.
df.groupby("day").agg({"total_bill": ["mean", "median"]})


# HO: Grup ortalamaları arasında ist ol anl fark yoktur

# parametrik anova testi:
f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])

# Nonparametrik anova testi:
kruskal(df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Sun", "total_bill"])
#fark vardir sonuc



#ama farklilik hangi gruptan kaynaklaniyor
from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df['total_bill'], df['day'])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())


--------------------------------------------------------------------------------



#####################################################
# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchasemetriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsxexcel’ininayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBiddinguygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç



#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.




#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################

# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz.
# Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.


import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)





kontrol_df = pd.read_excel("datasets/ab_testing.xlsx", sheet_name='Control Group')
test_df = pd.read_excel("datasets/ab_testing.xlsx", sheet_name='Test Group')


kontrol_df.head()


test_df.head()






# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.












# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.


concat_df = pd.concat([kontrol_df, test_df], axis=0)

concat_df.head()

#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.


# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz


#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz




# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz


# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.



##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.




# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.









