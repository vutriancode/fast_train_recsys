#from code.preprocess_data.CONFIG import LINK_DATA
import pandas as pd
import numpy as np
import pickle
import os

from pandas.core.indexes.api import union_indexes

from CONFIG import *


class User_Item:

    def __init__(self,file_name):
        self.link_pickle = None
        self.link_data = os.path.join(LINK_DATA,file_name)

    def create_user_item_history(self):
        
        with open(os.path.join(LINK_DATA,"contens.picke"),"rb") as contents:
            data2=pickle.load(contents)

        data = pd.read_csv(self.link_data)
        print(list(data2.keys()))
        data = data[data["postId"].isin(data2.keys())]
        
        user = data["userId"].unique()
        item = data["postId"].unique()
        event = data["eventId"].unique()
        print(len(user))
        print(len(item))
        print(len(data))

        user_dict = dict((x,y) for y,x in enumerate(user))
        item_dict = dict((x,y) for y,x in enumerate(item))
        event_dict = dict((x,y) for y,x in enumerate(event))


        ui_dict = dict()
        iu_dict = dict()
        ur_dict = dict()
        ir_dict = dict()


        for i1,i in data.iterrows():
            #print(i1)
            #creat ui_dict
            if user_dict[i["userId"]] not in ui_dict.keys():
                ui_dict[user_dict[i["userId"]]] = [item_dict[i["postId"]]]
            else:
                ui_dict[user_dict[i["userId"]]].append(item_dict[i["postId"]])
            
            #creat iu_dict
            if item_dict[i["postId"]] not in iu_dict.keys():
                iu_dict[item_dict[i["postId"]]] = [user_dict[i["userId"]]]
            else:
                iu_dict[item_dict[i["postId"]]].append(user_dict[i["userId"]])
            
            #create ur_dict
            if user_dict[i["userId"]] not in ur_dict.keys():
                ur_dict[user_dict[i["userId"]]] = [event_dict[i["eventId"]]]
            else:
                ur_dict[user_dict[i["userId"]]].append(event_dict[i["eventId"]])

            #create ir_dict
            if item_dict[i["postId"]] not in ir_dict.keys():
                ir_dict[item_dict[i["postId"]]] = [event_dict[i["eventId"]]]
            else:
                ir_dict[item_dict[i["postId"]]].append(event_dict[i["eventId"]])
            
        with open(os.path.join(LINK_DATA,"data.picke"),"wb") as out_put_file:
                pickle.dump([user_dict, item_dict, event_dict,\
                    ui_dict, iu_dict, ur_dict,ir_dict],out_put_file)

a=User_Item("user_history.csv")
a.create_user_item_history()
