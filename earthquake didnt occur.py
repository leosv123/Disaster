# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 07:36:34 2019

@author: lingraj
"""

#Earthquake detected:
import requests
import json
import random
import time

headers = {
    'appKey' : "f6779157-b946-4074-babe-c4174f37bf4e",
    'Content-Type' : "application/json"
    }

while True:
    url = "https://ptcu-thingworx74-fundamentals.portal.ptc.io/Thingworx/Things/WeatherRover1_ankit2003/Properties/Raw_Destination"
    #speed=random.randint(0,1)
   # print(speed)
    speed =0
    payload=json.dumps({"Raw_Destination":speed})
    response=requests.request("PUT", url, data=payload, headers=headers)
    print(response.text)
    
    
    
    url = "https://ptcu-thingworx74-fundamentals.portal.ptc.io/Thingworx/Things/WeatherRover1_ankit2003/Properties/Genergy"
    #speed=random.randint(0,1)
   # print(speed)
    Genergy =259000
    payload=json.dumps({"Genergy":Genergy})
    response=requests.request("PUT", url, data=payload, headers=headers)
    print(response.text)
    
    url = "https://ptcu-thingworx74-fundamentals.portal.ptc.io/Thingworx/Things/WeatherRover1_ankit2003/Properties/Gpulse"
    #speed=random.randint(0,1)
   # print(speed)
    Gpulse =2000
    payload=json.dumps({"Gpulse":Gpulse})
    response=requests.request("PUT", url, data=payload, headers=headers)
    print(response.text)
    
    url = "https://ptcu-thingworx74-fundamentals.portal.ptc.io/Thingworx/Things/WeatherRover1_ankit2003/Properties/Gdenergy"
    #speed=random.randint(0,1)
   # print(speed)
    Gdenergy =400
    payload=json.dumps({"Gdenergy":Gdenergy})
    response=requests.request("PUT", url, data=payload, headers=headers)
    print(response.text)
    
    
    url = "https://ptcu-thingworx74-fundamentals.portal.ptc.io/Thingworx/Things/WeatherRover1_ankit2003/Properties/NBumps"
    #speed=random.randint(0,1)
   # print(speed)
    NBumps =3
    payload=json.dumps({"NBumps":NBumps})
    response=requests.request("PUT", url, data=payload, headers=headers)
    print(response.text)

    #url = "http://192.168.84.250:8080/Thingworx/Things/vehicle/Properties/Battery"
    #Battery=random.randrange(0,72,3)
   # payload=json.dumps({"Battery":Battery})
   # response=requests.request("PUT", url, data=payload, headers=headers)
   # print(response.text)
   # url = "https://ptcu-thingworx74-fundamentals.portal.ptc.io/Thingworx/Things/ Pressure_Lingraj_ankit2003/Properties/GdPulse"
    #speed=random.randint(0,1)
    #print(speed)
   # GdPulse=2595648
    #payload=json.dumps({"GdPulse":GdPulse})
   # response=requests.request("PUT", url, data=payload, headers=headers)
   # print(response.text)


   
    time.sleep(30)
    print("hello")