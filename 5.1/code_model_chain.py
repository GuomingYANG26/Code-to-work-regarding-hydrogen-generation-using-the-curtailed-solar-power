# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:26:35 2022
@author: YANG Guoming
yangguoming1995@gmail.com
"""
import pvlib
import numpy as np
import pandas as pd
from pvlib.tools import cosd, sind, tand, asind

#载入数据
with open('D:/Doctor/paper/2023third/Python_Code/tmy_2020.csv', 'r') as f:
    data, metadata =  pvlib.iotools.psm3.parse_psm3(f, map_variables=True)
lat = metadata['latitude']
lon = metadata['longitude']
times = data.index

#太阳位置模型
solpos = pvlib.solarposition.get_solarposition(
    times, 
    lat, 
    lon,
    altitude=np.mean(pvlib.atmosphere.pres2alt(data['pressure'])),
    pressure=data['pressure'],
    method='nrel_numpy',
    temperature=data['temp_air'])

#extraterrestrial irradiation
ETI = pvlib.irradiance.get_extra_radiation(
    times, 
    solar_constant=1361.1, 
    method='nrel')

#相对气团和绝对气团
airmass = pvlib.atmosphere.get_relative_airmass(
    solpos['apparent_zenith'], 
    model='kastenyoung1989')
#airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass, data['pressure'])


#光伏板朝向
surface_tilt = 45.77
surface_azimuth = 180
#转换模型
poa_irradiance = pvlib.irradiance.get_total_irradiance(
    surface_tilt, 
    surface_azimuth, 
    solpos['apparent_zenith'],
    solpos['azimuth'],  #看例子 https://pvlib-python.readthedocs.io/en/stable/gallery/adr-pvarray/plot_simulate_system.html#sphx-glr-gallery-adr-pvarray-plot-simulate-system-py
    data['dni'], 
    data['ghi'], 
    data['dhi'], 
    dni_extra=ETI, 
    airmass=airmass,  #相对气团
    albedo=data['albedo'] , 
    model='perez')
#入射角
aoi = pvlib.irradiance.aoi(surface_tilt, 
                           surface_azimuth, 
                           solpos['apparent_zenith'], 
                           solpos['azimuth'])

#反射损失模型              ---------------------------增加这部分内容-----------------
n_pv = 1.526  #光伏模块的反射率
n_air = 1   #大气反射率
K = 4.0   #The glazing extinction coefficient in units of 1/meter.
L = 0.002 #The glazing thickness in units of meters.
n_T = 1.4585 #The refeaction index of the pyranometer cover

refractive_angle = asind(n_air / n_pv * (sind(aoi)))
Rd = sind(refractive_angle-aoi)**2 / sind(refractive_angle+aoi)**2 + tand(refractive_angle-aoi)**2 / tand(refractive_angle+aoi)**2
tau_zero = np.exp(-K * L)
#relative transmittance for beam radiation
tau_b = np.exp(-K * L / cosd(refractive_angle))/tau_zero * (1-Rd/2)/(1-((n_pv-1)/(n_pv+1))**2)   #师姐RSER式4
tau_b = np.where(tau_b<0, 0, tau_b)  #令负数为0l     
#tau_b_pvlib = pvlib.iam.physical(aoi)         #直接调用函数也可

# Xie et al The “Fresnel Equations” for Diffuse radiation on Inclined photovoltaic Surfaces (FEDIS)
w = (n_pv*(n_T+1)**2)/(n_T*(n_pv+1)**2) * (2.77526*10**(-9) + 3.74953*n_pv - 5.18727*(n_pv)**2 + 3.41186*(n_pv)**3 - 1.08794*(n_pv)**4 + 0.13606*(n_pv)**5)
#relative transmittance for diffuse radiation
tau_d = 2*w/(np.pi*(1+cosd(surface_tilt)))*(30/7*np.pi - 160/21*surface_tilt/180*np.pi - 10/3*np.pi*cosd(surface_tilt) + 160/21*cosd(surface_tilt)*sind(surface_tilt) - \
     5/3*np.pi*cosd(surface_tilt)*(sind(surface_tilt))**2 + 20/7*cosd(surface_tilt)*(sind(surface_tilt))**3 - 5/16*np.pi*cosd(surface_tilt)*(sind(surface_tilt))**4 + 16/105*cosd(surface_tilt)*(sind(surface_tilt))**5)

#relative transmittance for ground-reflected radiation
tau_g = 40*w/(21*(1-cosd(surface_tilt))) - tau_d*(1+cosd(surface_tilt))/(1-cosd(surface_tilt))


absorbed_radiation = tau_b*poa_irradiance.poa_direct + tau_d*poa_irradiance.poa_sky_diffuse + tau_g*poa_irradiance.poa_ground_diffuse



#电池模型
cell_temp0 = pvlib.temperature.sapm_cell(
    poa_global=poa_irradiance['poa_global'], 
    temp_air=data['temp_air'], 
    wind_speed=data['wind_speed'],
    a=-3.56, 
    b=-0.075, 
    deltaT=3, 
    irrad_ref=1000.0)

#PV module model
pv_dc = pvlib.pvsystem.pvwatts_dc(
    g_poa_effective=absorbed_radiation, 
    temp_cell=cell_temp0, 
    pdc0=1000000,                     #------------------------从ECM的999000改成现在1000000----------------
    gamma_pdc=-0.42/100, 
    temp_ref=25.0)
#PVWatts loss model
loss = pvlib.pvsystem.pvwatts_losses(
    soiling=2, 
    shading=0, 
    snow=3, 
    mismatch=0, 
    wiring=2, 
    connections=0, 
    lid=0, 
    nameplate_rating=0, 
    age=0, 
    availability=3)/100
#inverter model
pv_ac0 = pvlib.inverter.pvwatts(
    pdc=pv_dc*(1 - loss), 
    pdc0=833000/0.985,
    eta_inv_nom=0.985, 
    eta_inv_ref=0.9637)

#final PV AC power output
pv_ac = pv_ac0.fillna((0))
pv_ac[pv_ac<0] = 0
pv_ac.index = range(8760)

#保存数据
pv_ac.to_csv('D:/Doctor/paper/2023third/Python_Code/pv_modelchain.csv')


#一天内总和
daily_pv = [sum(pv_ac[24*x:24*x+23]) for x in range(0,365)]
daily_pv = pd.Series(daily_pv)/1000/1000
daily_pv.to_csv('D:/Doctor/paper/2023third/Python_Code/daily_pv_power_model_chain.csv')
'''
daily_pv_x = np.zeros((365))
for i in range(365):
    if daily_pv[i] <= 0.17*24*1000*1000:
        daily_pv_x[i] = daily_pv[i]
    else:
        daily_pv_x[i] = 0.17*24*1000*1000
daily_pv_x = pd.Series(daily_pv_x)
daily_pv_x.to_csv('D:\Doctor\paper\2023third\code\daily_x_pv_power.csv')


daily_pv_2x = np.zeros((365))
for i in range(365):
    if 2*daily_pv[i] <= 0.17*24*1000*1000:
        daily_pv_2x[i] = 2*daily_pv[i]
    else:
        daily_pv_2x[i] = 0.17*24*1000*1000
daily_pv_2x = pd.Series(daily_pv_2x)
daily_pv_2x.to_csv('D:\Doctor\paper\2023third\code\daily_2x_pv_power.csv')
'''

