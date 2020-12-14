import pickle
from CONFIG import *
import os
import pandas as pd


with open(os.path.join(LINK_DATA,"score.pickle"),"rb") as output:
    score=pickle.load(output)
with open(os.path.join(LINK_DATA,"data.picke"),"rb") as out_put_file:
    user_dict, item_dict, event_dict, ui_dict, iu_dict, ur_dict,ir_dict = pickle.load(out_put_file)
def create_user_item_rating_train_test(file_name):
    link_data = os.path.join(LINK_DATA,file_name)
    with open(os.path.join(LINK_DATA,"content.picke"),"rb") as contents:
        data2=pickle.load(contents)

    data = pd.read_csv(link_data)
    print(list(data2.keys()))
    data = data[data["postId"].isin(item_dict.keys())]
    training = dict()
    for i1,i in data.iterrows():
        #print(i1)
        if tuple([user_dict[i["userId"]],item_dict[i["postId"]]]) not in training.keys():
            training[tuple([user_dict[i["userId"]],item_dict[i["postId"]]])] = score[i["eventId"]]
            #print("s")
        else:
            training[tuple([user_dict[i["userId"]],item_dict[i["postId"]]])] = min(5,training[tuple([user_dict[i["userId"]],item_dict[i["postId"]]])] + score[i["eventId"]])
    with open(os.path.join(LINK_DATA,"training.pickle"),"wb") as output:
        pickle.dump(training,output)
    print(len(training))
    #print(training)
    #print(training)
create_user_item_rating_train_test("user_history.csv")