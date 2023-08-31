# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 21:16:51 2023
@author: YANG Guoming
@email: yangguoming1995@gmail.com

"""
import pvlib
import pandas as pd
#reads the TMY data downloaded from NSRDB for Harbin  
with open('D:/Doctor/paper/2023third/Python_Code/tmy_2020.csv', 'r') as f:
    data, metadata =  pvlib.iotools.psm3.parse_psm3(f, map_variables=True)
lat = metadata['latitude']
lon = metadata['longitude']
times = data.index

#acquires the information relating to the solar position
solpos = pvlib.solarposition.get_solarposition(
    times, 
    lat, 
    lon,
    temperature=data['temp_air'])

#gets the surface tilt and surface azimuth, systemtype == 'fixed':
surface_tilt = 45.77
surface_azimuth = 180

#Determines total in-plane irradiance and its beam, sky diffuse and ground reflected components using the default model:'isotropic'
poa_irradiance = pvlib.irradiance.get_total_irradiance(
    surface_tilt, 
    surface_azimuth, 
    solpos['apparent_zenith'],
    solpos['azimuth'], 
    data['dni'], 
    data['ghi'], 
    data['dhi'])
poa_irradiance.fillna(0, inplace=True) #替换nan为0，在原数据框上进行

#电池模型
T_noct = 46 #单位°C
cell_temp = data['temp_air'].values + poa_irradiance['poa_global'].values/800 * (T_noct - 20)  #草稿公式2

#草稿公式3
gamma_mod = -0.42/100  #光伏组件功率的温度系数unit:%/C
eta_inv = 0.985 #https://www.energysage.com/solar-inverters/tmeic/2866/pvl-l0833gr/
P_s = 1000000 #光伏电站装机容量P_dc0  单位W
pv_tradional = P_s * eta_inv * poa_irradiance['poa_global'].values/1000 * (1 + gamma_mod * (cell_temp - 25) ) 
pv_tradional[pv_tradional<0] = 0
pv_tradional[pv_tradional>833000] = 833000
pv_tradional = pd.Series(pv_tradional)
pv_tradional.to_csv('D:/Doctor/paper/2023third/Python_Code/pv_traditional.csv')


#一天内总和
daily_pv = [sum(pv_tradional[24*x:24*x+23]) for x in range(0,365)]
daily_pv = pd.Series(daily_pv)/1000/1000
daily_pv.to_csv('D:/Doctor/paper/2023third/Python_Code/daily_pv_power_pv_traditional.csv')

#日负荷数据
rou_HP = 2   #Heat pump供应热负荷
rou_EC = 4   #Electric chiller供应冷负荷
Load_data = pd.read_csv(r'D:\Doctor\paper\2023third\Python_Code\723740_TMY3_BASE.csv', index_col=0)  #光伏物理模型链模拟的光伏出力 单位W
Elload = 400 * (Load_data['Fans:Electricity [kW](Hourly)'] + Load_data['General:InteriorLights:Electricity [kW](Hourly)'] + \
                     Load_data['General:ExteriorLights:Electricity [kW](Hourly)'] + Load_data['Appl:InteriorEquipment:Electricity [kW](Hourly)'] + \
                     Load_data['Misc:InteriorEquipment:Electricity [kW](Hourly)'] + Load_data['Cooling:Electricity [kW](Hourly)']/rou_EC + \
                     Load_data['Heating:Gas [kW](Hourly)']/rou_HP)
Elload.index = range(8760)




daily_Elload = [sum(Elload[24*x:24*x+23]) for x in range(0,365)]
daily_Elload = pd.Series(daily_Elload)/1000
daily_Elload.to_csv('D:/Doctor/paper/2023third/Python_Code/daily_Elload.csv')


