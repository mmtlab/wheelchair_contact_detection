# -*- coding: utf-8 -*-
"""
Created on Sun May 14 17:10:15 2023

@author: giamp
"""
import pandas as pd
import numpy as np
def decode_data(df1):
    for j in range(24):
        df1["el"+str(j+1)]=range(len(df1["Data"]))
    for i in range(len(df1["Data"])):
        df1["Data"][i]= df1["Data"][i][2:]
        df1["Data"][i]= df1["Data"][i][:-4]+'000'
        for j in range(24):
            a=df1["Data"][i][j*8:j*8+7]
            b=str(a[1])+str(a[3])+str(a[5])
            df1["el"+str(j+1)][i]=b
    df1 = df1.drop('Data', axis=1)
    df1.rename(columns={'el1':'el5','el5':'el1'},inplace=True)
    df1 = df1.astype(float)
    df1.iloc[:,1:]=df1.iloc[:,1:]-df1.iloc[1,1:]
    return df1
def find_first_bump(cap_data,stdweight=4):
    mean=cap_data["Accelerometer Y (g)"].iloc[0:60].mean()
    std=cap_data["Accelerometer Y (g)"].iloc[0:60].std()
    for i in range(len(cap_data["Time (s)"])):
       if abs(cap_data["Accelerometer Y (g)"][i]-mean)>6*std:
           syncimpulsetime=cap_data["Time (s)"][i]
           break
    return syncimpulsetime
def find_contact(cap_data,contact_threshold=-20):
    contact_data=pd.DataFrame(columns=['Time (s)','Contact','Position'])
    contact_data['Time (s)']=cap_data['Time (s)']
    for j in range(len(cap_data['Time (s)'])):
        position=[]
        for i in range(24):
           if cap_data['el'+str(i+1)][j]<contact_threshold:
               contact_data['Contact'][j]=True
               position.append('el'+str(i+1))
           if position==[]: 
                contact_data['Contact'][j]=False
        if position==[]:
            contact_data['Position'][j]=np.nan
        else:
            contact_data['Position'][j]=position
    contact_data['Contact']=contact_data['Contact'].astype(float)
    return contact_data
# rawdata=r"C:\Users\giamp\OneDrive\Desktop\prova bottone 4\NGIMU - 0035EEA3\sensors.csv"
# testname=
# csvfilefolder=r'G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230511\02_preprocessing\capacitive sleeve'
# csvfullpath=os.path.join(csvfilefolder,testname+'.csv')
# df1=pd.read_csv(rawdata).astype(float)
# mean=df1["Accelerometer Y (g)"].iloc[0:60].mean()
# std=df1["Accelerometer Y (g)"].iloc[0:60].std()
# for i in range(len(df1["Time (s)"])):
#    if df1["Accelerometer Y (g)"][i]>(mean*1.2):
#        led_time=df1["Time (s)"][i]
#        break
# df1.to_csv(csvfullpath)
# df1.plot(x="Time (s)",y="Accelerometer Y (g)")
# cap_data=pd.read_csv(r"G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230511\02_preprocessing\capacitive sleeve\hand_pattern_semi.csv")
# led_data=pd.read_csv(r"G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230511\02_preprocessing\realsenseRight\ledstatus\hand_pattern_slop.csv")
# contact_time=find_first_contact(cap_data)
# led_time=led_data["Time(s)"].iloc[-1].astype(float)
# csini=led_time-contact_time