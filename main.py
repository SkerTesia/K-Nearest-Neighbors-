import random
import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', None)
dataset = "TV_show_data.csv" #import file
df = pd.read_csv(dataset)

sample_data = df.sample(n=1000)  #set number of samples to 1000 #need to remove random state to be able to generate samples randomly in the end
new_data = df.sample(n=1) #set 1 new sample

min_limit = 7.5 #set rating edge to 7.5


#print(columns)
#print("ratings", columns['Rating'])
#print("ratings", sample_data['Rating'].size,  sample_data['Rating'].unique().size)


#weights: 60--> 1.2, 3000 --> 60, 200 -> 4 #lahko ugotovis z min,max
def season_weight(weight1):  #columns = (columns - mean) / std. deviation --> can do this way too
    return weight1 / 1.2  #1-60

def episode_weight(weight2): #1-3000
    return weight2 / 60

def runtime_weight(weight3): #1-200
    return weight3 / 4

def weigh_data(data):
    data = data[['Rating', 'Total Seasons', 'Total Episodes', 'Average Runtime']]  # import columns with limited samples
    weighted_s = data['Total Seasons'].apply(season_weight)
    weighted_e = data['Total Episodes'].apply(episode_weight)
    weighted_run = data['Average Runtime'].apply(runtime_weight)
    data.insert(4, "Weighted Seasons", weighted_s, True)
    data.insert(5, "Weighted Episodes", weighted_e, True)
    data.insert(6, "Weighted Runtime", weighted_run, True)
    return data


#columns = weigh_data(columns)

#print("weight")
#print(columns['Weighted Seasons'])
#print(columns.head())

#get distance of points based on ratings
def euclid_dist(x, y):
    #print((x["Weighted Seasons"] - y["Weighted Seasons"])**2)
    return np.sqrt((x["Weighted Seasons"] - y["Weighted Seasons"])**2 + (x["Weighted Episodes"] - y["Weighted Episodes"])**2 + (x["Weighted Runtime"] - y["Weighted Runtime"])**2)

def find_KNN(sample, k):
    knn = sample.nsmallest(k, 'Distance')
    return knn

def get_ratings(knn):
    avg_rating = knn['Rating'].mean()
    return avg_rating

def get_KNN(k, sample, new): #compare to get k nearest neighbors, new column to save distance values
    sample = weigh_data(sample)
    new = weigh_data(new)
    #print(new)
    #print(new.iloc(0))
    #vrstica = new.iloc[0]
    #print(vrstica)
    euclid_dist_helper = lambda x: euclid_dist(x, new.iloc[0])
    distance = sample.apply(euclid_dist_helper, axis = 1)
    sample.insert(7, ["Distance"], distance, True)

    k_nearest = find_KNN(sample, k)
    print("sample:", sample[['Distance']].head())
    average_rating = get_ratings(k_nearest)
    print("nearest:", k_nearest[['Rating', 'Distance']])


    if average_rating >= min_limit:
        print(f"The predicted rating is {average_rating:.2f}. The show is good.")
    else:
        print(f"The predicted rating is {average_rating:.2f}. The show is bad.")

    return average_rating
    #print(sample)
    #sample now has distance column.


k = 5
get_KNN(k, sample_data, new_data) #its best to have uneven number of neighbors (best of uneven is five, but can use others too)


#random state control!! --> in the beginning helped with generating, in the end it stopped generating and so had to be removed. but i dont understand why.
#what i want to make is the percentage of good vs percentage of bad predictions