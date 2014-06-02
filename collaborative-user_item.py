__author__ = 'Shaun'
from math import sqrt
import sys
import numpy as np


def user_based_pred(train_file,test_file):

    """

    :param train_file: training set file name
    :param test_file: test set file name

    """
    input_ori = {}

    train_input = open(train_file)
    max_user = 0
    max_item = 0
    for all_rec in train_input:   # get user_item dict {'user':[item1:rating,item2:rating]}}
        temp_rec = all_rec.split("\t")
        if temp_rec[0] in input_ori:
            input_ori[temp_rec[0]][temp_rec[1]] = temp_rec[2]
            if (int(temp_rec[1]) > max_item):
                max_item = int(temp_rec[1])
        else:
            input_ori[temp_rec[0]] = {temp_rec[1]: temp_rec[2]}
            if (int(temp_rec[1]) > max_item):
                max_item = int(temp_rec[1])
    max_user = len(input_ori)


    #initialize user_item matrix and user_mean
    user_item = [[0 for i in range(max_item + 1)] for i in range(max_user + 1)]     # user-item matrix
    user_item_sub = [[0 for i in range(max_item + 1)] for i in range(max_user + 1)] # user-item matrix subtracted user-mean
    user_sim = [[0 for i in range(max_user + 1)] for i in range(max_user + 1)]      # user-user similarity matrix
    user_count = [0 for i in range(max_user + 1)]                                   # user-rated item number
    user_mean = [0 for i in range(max_user + 1)]                                    # user-mean vector

    for user_key, all_user in input_ori.items():                                    # user-item dict transform into  user-item matrix
        for item_key, item in all_user.items():
            user_item[int(user_key)][int(item_key)] = int(item)
            user_count[int(user_key)] += 1
    for i in range(1, max_user + 1):
        user_mean[i] = 1.0 * sum(user_item[i]) / (user_count[i])                    # calculate user-mean

    #calculate user_item_sub matrix
    for user_key, all_user in input_ori.items():
        for item_key, item in all_user.items():
            if user_item[int(user_key)][int(item_key)] != 0:
                user_item_sub[int(user_key)][int(item_key)] = user_item[int(user_key)][int(item_key)] - user_mean[
                    int(user_key)]

    eps = sys.float_info.min  # EPS

    #calculate similarity
    for i in range(1, max_user + 1):
        for j in range(i + 1, max_user + 1):
            tp_m1 = user_item[i]
            tp_m2 = user_item[j]
            tmp_u1 = user_item_sub[i]
            tmp_u2 = user_item_sub[j]
            s1 = sum([x * y for x, y in zip(tmp_u1, tmp_u2)])
            s2 = sqrt(sum([x ** 2 for x, y, xi, yi in zip(tmp_u1, tmp_u2, tp_m1, tp_m2) if xi * yi != 0]))
            s3 = sqrt(sum([y ** 2 for x, y, xi, yi in zip(tmp_u1, tmp_u2, tp_m1, tp_m2) if xi * yi != 0]))
            sim = s1 * 1.0 / ((s2 * s3) + eps)
            user_sim[i][j] = sim # store into user_sim matrix

    #fill up similarity matrix
    for i in range(1, max_user + 1):
        for j in range(1, i):
            user_sim[i][j] = user_sim[j][i]

    # sim_out = open("sim_out_user.txt","w")
    # for i in range(1,max_user+1):
    #     for j in range(1,max_user+1):
    #         a = "%d %d %f\n" % (i, j, user_sim[i][j])
    #         sim_out.write(a)
    # sim_out.close()
    # Prediction
    out = open(train_file+"_user_prd", "w")  # Predict result

    test_vet = open(test_file)
    user_item_mat = np.array(user_item)
    user_np = np.array(user_sim)

    k = 10
    test_vec = []
    Pred = []
    for all_rec in test_vet:
        temp_rec = all_rec.split("\t")
        test_vec.append(int(temp_rec[2]))
        PCCsimilarityUser = []
        #get top-k most similar user

        if int(temp_rec[1]) >= max_item:  # if item_n in test-set but not in train-set Pred = user-mean
            pre_pred = user_mean[int(temp_rec[0])]
            out.write(str(pre_pred))
            out.write("\n")
            Pred.append(pre_pred)
            continue

        PCCsimilarityUser = user_np[int(temp_rec[0]), :].copy()  # get current user similarity
        current_item_user = user_item_mat[:, int(temp_rec[1])] < 1  # get user unrated item

        PCCsimilarityUser[current_item_user] = -2  # set unrated item similarity to 0
        PCCsimilarityUser[0] = -9  # first element [0] in array
        PCCSorted = PCCsimilarityUser.argsort()     # sort similarity
        PCCSorted = PCCSorted[::-1]                 # reverse sorted
        user = PCCSorted[0:k]                       # get top-k user id
        sims = PCCsimilarityUser[PCCSorted][0:k]    # get top-k similarity
        pre_pred = 0
        sum_mul = 0
        for auser in zip(user, sims):               # SUM
            sum_mul += auser[1] * user_item_sub[auser[0]][int(temp_rec[1])]

        pre_pred = user_mean[int(temp_rec[0])] + sum_mul / sum(sims)  # Predict
        if pre_pred > 5:
            pre_pred = 5
        if pre_pred < 1:
            pre_pred = 1
        out.write(str(pre_pred))
        out.write("\n")
        Pred.append(pre_pred)
    out.close()

    #Evaluate
    Pred = np.array(Pred)
    test_vec = np.array(test_vec)
    rmse = np.sqrt(((test_vec - Pred) ** 2).mean())
    print rmse

def item_based_pred(train_file,test_file):

    """

    use set instead of array to accelerate the calculations of similarity

    user-item set: to find user1 and user2 both rated item
    {user1:[item1,item2],user2:[item3,item4]...}


    """
    input_ori = {}

    # get user_item dict
    train_input = open(train_file)
    item_user_dictset={}
    user_item_dictset={}
    max_user = 0
    max_item = 0
    for all_rec in train_input:  # get user_item dict {'user':[item1:rating,item2:rating]}}
        temp_rec = all_rec.split("\t")
        if temp_rec[0] in input_ori:
            input_ori[temp_rec[0]][temp_rec[1]] = temp_rec[2]
            if int(temp_rec[1]) > max_item:
                max_item = int(temp_rec[1])
        else:
            input_ori[temp_rec[0]] = {temp_rec[1]: temp_rec[2]}
            if max_item < int(temp_rec[1]):
                max_item = int(temp_rec[1])
    max_user = len(input_ori)

    item_sim = np.zeros((max_item+1,max_item+1))  # item-item similarity
    user_mean = [0 for i in range(max_user + 1)]  # user-mean

    for user,row in input_ori.items():            # get user-item set and item-item set
        current_mean=[]
        for item,rating in row.items():
            current_mean.append(int(rating))

            # get item_user_dict_set
            if item in item_user_dictset:
                item_user_dictset[item].add(user)
            else:
                item_user_dictset[item] = set()
                item_user_dictset[item].add(user)

            # get user_item_dict_set
            if user in user_item_dictset:
                user_item_dictset[user].add(item)
            else:
                user_item_dictset[user] = set()
                user_item_dictset[user].add(item)

        user_mean[int(user)] = sum(current_mean)*1.0/len(current_mean)  # calculate user-mean

    #calculate similarity
    eps = sys.float_info.epsilon  # EPS

    for i in range(1, max_item+1):
        for j in range(i+1, max_item+1):
            if str(i) not in item_user_dictset or str(j) not in item_user_dictset:
                continue
            both_rated = item_user_dictset[str(i)] & item_user_dictset[str(j)]
            s1 = 0
            s2 = 0
            s3 = 0
            for i_u in both_rated:
                s1 += (int(input_ori[i_u][str(i)]) - user_mean[int(i_u)])*(int(input_ori[i_u][str(j)])-user_mean[int(i_u)])
                s2 += (int(input_ori[i_u][str(i)]) - user_mean[int(i_u)])**2
                s3 += (int(input_ori[i_u][str(j)]) - user_mean[int(i_u)])**2
            s2 = sqrt(s2)
            s3 = sqrt(s3)
            sim = s1*1.0/(s2*s3+eps)
            item_sim[i][j] = sim
            item_sim[j][i] = sim  # fill up

    # sim_out = open("sim_out_item.txt","w")
    # for i in range(1,max_item+1):
    #     for j in range(1,max_item+1):
    #         a = "%d %d %f\n" % (i, j, item_sim[i][j])
    #         sim_out.write(a)
    # sim_out.close()

    #Prediction
    test_vec = open(test_file)
    k = 10
    out = open(train_file+"_item_prd","w")
    Pred = []
    test_evaluate = []
    for p_rec in test_vec:
        temp_rec = p_rec.split("\t")
        c_u = int(temp_rec[0])
        c_i = int(temp_rec[1])
        c_r = int(temp_rec[2])
        test_evaluate.append(int(c_r))
        if c_i >= max_item:
            pred_pre = user_mean[c_u]
            out.write(str(pred_pre))
            out.write("\n")
            Pred.append(pred_pre)
            continue

        item_sim_cur = item_sim[c_i]  # similarity between item-c_i and other items

        arr = [int(i) for i in user_item_dictset[str(c_u)]]  # get all user-c_u rated item
        sims = []
        rating = []
        for item in arr:
            sims.append(item_sim_cur[item])
            rating.append(int(input_ori[str(c_u)][str(item)]))

        sims = np.array(sims)  # item_n: vector of similarity
        rating = np.array(rating)  # vector of rating score
        sorted_index_sim = sims.argsort()[::-1][0:k]  # sort - reverse - top-k
        sorted_sim = sims[sorted_index_sim]  # sorted top-k similarity
        sorted_rating = rating[sorted_index_sim]  # sorted top-k rating score

        pred_pre = sum(sorted_sim*sorted_rating)/(sum(sorted_sim)+eps)  # predict
        if pred_pre > 5:
            pred_pre=5
        if pred_pre < 1:
            pred_pre=1
        out.write(str(pred_pre))
        out.write("\n")
        Pred.append(pred_pre)

    out.close()

    #evaluate
    Pred = np.array(Pred)
    test_evaluate = np.array(test_evaluate)
    rmse = np.sqrt(((test_evaluate - Pred) ** 2).mean())
    print rmse




if __name__ == '__main__':
    #t1=Timer("cal_sim()","from __main__ import cal_sim")
    #print t1.timeit(1)
    #t1=Timer("item_based_pred()","from __main__ import item_based_pred")
    #print t1.timeit(1)

    #item_based_pred("u1.base","u1.test")
   # user_based_pred("u1.base","u1.test")

    for i in range(1,6):
        tr_file = "u"+str(i)+".base"
        t_file = "u"+str(i)+".test"
        print tr_file,t_file,"item_based"
        item_based_pred(tr_file,t_file)  # about 40s
        print tr_file,t_file,"user_based"
        user_based_pred(tr_file,t_file)  # about 5min