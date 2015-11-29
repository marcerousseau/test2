'''
Created on 22 oct. 2015

@author: mrousseau
'''
from lifetimes.datasets import load_cdnow
from lifetimes.estimation import BetaGeoFitter, GammaGammaFitter, ParetoNBDFitter
data = load_cdnow(index_col=[0])
print data.head()
print type(data) 
from scikits.recommenders.knn import UserBasedRecommender

bgf = ParetoNBDFitter(penalizer_coef=0.0)
bgf.fit(data['frequency'], data['recency'], data['T'])
print bgf
print bgf.conditional_expected_number_of_purchases_up_to_time(1,2,45,90)
print bgf.conditional_probability_alive(2,45,90)

from lifetimes.plotting import plot_frequency_recency_matrix
plot_frequency_recency_matrix(bgf)

#from matplotlib import pyplot as plt
#plt.show()

#from lifetimes.plotting import plot_probability_alive_matrix
#plot_probability_alive_matrix(bgf)


#plt.show()

t = 1
data['predicted_purchases'] = data.apply(lambda r: bgf.conditional_expected_number_of_purchases_up_to_time(t, r['frequency'], r['recency'], r['T']), axis = 1)
print data.sort('predicted_purchases').tail(5)

from lifetimes.datasets import load_transaction_data
from lifetimes.utils import summary_data_from_transaction_data

transaction_data = load_transaction_data()
print transaction_data.head()
print type(transaction_data)
print transaction_data.columns

print data.columns
print data.head()

t = 10
data['predicted_purchases'] = data.apply(lambda r: bgf.conditional_expected_number_of_purchases_up_to_time(t, r['frequency'], r['recency'], r['T']), axis = 1)
print data

from pandas import DataFrame
d = [{'id': 1, 'R':23, 'F':12, 'M':12.5}, {'id': 2,'R':43, 'F':1, 'M':120.5}, {'id': 3,'R':203, 'F':2, 'M':19.5}]
test = DataFrame(d)
print test
print test.info()
print test['R']

ggf = GammaGammaFitter(penalizer_coef=0)