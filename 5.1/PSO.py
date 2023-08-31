# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:53:40 2023
@author: YANG Guoming
@email: yangguoming1995@gmail.com

"""
#####备注  电解槽，压缩机，储氢罐，电池储能和光伏电站的缩写分别为Elz, Cmp, HS, BS, PV
import pandas as pd
from gurobipy import GRB
from gurobipy import * 
import gurobipy as gp
import numpy as np
import pvlib
import math
from scipy.stats import linregress
import pwlf
import xlsxwriter
import time

###---------------------------------模型参数-------------------------------------######
tao = 8/100   #贴现率
c_Elz, OM_Elz, Y_Elz = 1027.5, 5/100, 10 #电解槽单位投资成本 $/kW, 运维成本因子值, 寿命 yr.
xi_Elz = tao*(1+tao)**Y_Elz /((1+tao)**Y_Elz - 1)  #电解槽资本回收系数
c_Cmp, OM_Cmp, Y_Cmp = 730, 1/100, 20 #压缩机单位投资成本 $/kW, 运维成本因子值, 寿命 yr.
xi_Cmp = tao*(1+tao)**Y_Cmp /((1+tao)**Y_Cmp - 1)  #压缩机资本回收系数
c_HS, OM_HS, Y_HS = 9292.5, 1/100, 20  #储氢罐单位投资成本 $/m^3, 运维成本因子值, 寿命 yr.
xi_HS = tao*(1+tao)**Y_HS /((1+tao)**Y_HS - 1)  #储氢罐资本回收系数
c_BS, OM_BS, Y_BS = 137, 0.02/100, 15  #电池单位投资成本 $/kWh, 单次充放电运维成本因子值，寿命 yr.
xi_BS = tao*(1+tao)**Y_BS/((1+tao)**Y_BS - 1) #电池资本回收系数
c_PV, OM_PV, Y_PV = 833, 1/100, 30 #光伏电站单位投资成本  $/kW, 运维成本因子值, 寿命
xi_PV = tao*(1+tao)**Y_PV / ((1+tao)**Y_PV - 1)  #光伏电站资本回收系数

hydrogen_price = 5   #氢气售价 $/kg
#负荷需求
rou_HP = 2   #Heat pump供应热负荷
rou_EC = 4   #Electric chiller供应冷负荷
Load_data = pd.read_csv('D:/Doctor/paper/2023third/Python_Code/723740_TMY3_BASE.csv', index_col=0)  #单栋居民住宅电力负荷数据 kW
P_Ld = 400 * (Load_data['Fans:Electricity [kW](Hourly)'] + Load_data['General:InteriorLights:Electricity [kW](Hourly)'] + \
                     Load_data['General:ExteriorLights:Electricity [kW](Hourly)'] + Load_data['Appl:InteriorEquipment:Electricity [kW](Hourly)'] + \
                     Load_data['Misc:InteriorEquipment:Electricity [kW](Hourly)'] + Load_data['Cooling:Electricity [kW](Hourly)']/rou_EC + \
                     Load_data['Heating:Gas [kW](Hourly)']/rou_HP)
P_Ld.index = range(8760)
'''
P_Ld0 = P_Ld.copy()
k = 2   #向前移动k个时间段
for i in range(k):
    P_Ld[i] = P_Ld0[8760-(k-i)] 
for i in range(k, 8760):
    P_Ld[i] = P_Ld0[i-k]
'''
#光伏出力
P_power = pd.read_csv('D:/Doctor/paper/2023third/Python_Code/pv_modelchain.csv', index_col=0)   #物理模型链模拟的1MW光伏电站的光伏出力 单位W
P_power = P_power.iloc[:,0].values/1000   #单位kW
#reads the TMY data downloaded from NSRDB for Harbin  
with open('D:/Doctor/paper/2023third/Python_Code/tmy_2020.csv', 'r') as f:
    data, metadata =  pvlib.iotools.psm3.parse_psm3(f, map_variables=True)
#以1 MW光伏电站的模拟出力为基准，确定光伏初始装机容量下的出力
PGF = 0.621*np.sum(data['ghi'])/365/1000  #光伏板生成因子
Cap_PV = np.sum(P_Ld)/365 /PGF  #初始装机容量  单位kW
P_PV = P_power*Cap_PV/1000   #未扩建光伏电站的实际出力，单位kW    

Time = 8760      #一年的小时数
Hh = 39 #氢气高热值  kWh/kg
Pre0 = 1  #标准大气压 bar
Pre_Cmp = 200 #压缩机工作压力 bar
Pre_Cmp0 = 350 #压缩机参考工作压力 bar
P_Cmp0 = 2.1  #压缩机在标准工作压力下压缩单位千克氢气单位小时消耗的电能 kWh/kg/h
Niu_H = 30  #kg/m^3 氢气的体积质量分数 0.03kg/L
delt_Time = 1 #时间间隔一小时
sig_BS = 0.0001 #电池自放电率
eta_Ch_BS = 0.95 #电池充电效率
eta_Dis_BS = 0.95 #电池放电效率

#电池信息
one_battery_capacity = 40*133/1000 #单个电池额定容量 单位kWh
#电池放电样本，单位标幺值,W,W
N_j = 14  #样本点个数
SoC_j = [0, 0.01, 0.07, 0.13, 0.19, 0.25, 0.31, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 1]
Pminus_j = [0, 235.47, 5143.03, 10474.3, 15807.6, 21248.7, 26807.2, 305.03, 6100.49, 12201, 18301.5, 24402, 30502.5, 0]  #单位W
Pdis_j = [0, 234.69, 4829.89, 9222.88, 12994.5, 16252.3, 19007.4, 304.25, 5791.81, 10966.2, 15523.3, 19463, 22785.3, 0]  #单位W
Pminus_j = np.array(Pminus_j)/1000  #单位kW
Pdis_j = np.array(Pdis_j)/1000   #单位kW
#电池充电样本，单位标幺值,W,W
N_k = 20  #样本点个数
SoC_k = [0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 1]
Pplus_k = [0, 234.67, 929.33, 1833.6, 2712.78, 3566.9, 4395.93, 261.08, 1034.92, 2044.8, 3029.66, 3989.49, 4924.29, 303.62, 1202.88, 2375.54, 3517.02, 4626.75, 5686.86, 0] #单位W
Pcha_k = [0, 235.47, 941.87, 1883.74, 2825.62, 3767.49, 4709.36, 261.86, 1047.43, 2094.86, 3142.29, 4189.72, 5237.15, 304.39, 1215.23, 2424.95, 3628.22, 4824.46, 5995.89, 0] #单位W
Pplus_k = np.array(Pplus_k)/1000   #单位kW
Pcha_k = np.array(Pcha_k)/1000   #单位kW


#电解槽效率曲线
theoretical_eta_curve = pd.read_excel(r'D:\Doctor\paper\2023third\code\theoretical_eta_curve.xlsx')
power_per_unit =   theoretical_eta_curve.iloc[:,1]
effi_UI = theoretical_eta_curve.iloc[:,0]

def PSO(P_Elz, N_BS):
    ####-----------------------制氢量分段线性化----------------------
    # initialize piecewise linear fit with your x and y data with a random seed
    my_pwlf = pwlf.PiecewiseLinFit(P_Elz*power_per_unit, power_per_unit*P_Elz/Hh*effi_UI, seed=123)

    # fit the data for five line segments
    # force the function to go through the data points (0.0, 0.0)
    # where the data points are of the form (x, y)
    x_c = [0.0]
    y_c = [0.0]
    res = my_pwlf.fit(5, x_c, y_c)

    # predict for the determined points
    xpoint = my_pwlf.fit_breaks
    ypoint  = my_pwlf.predict(xpoint)
    line_num = len(xpoint) - 1
    line_slope = np.zeros(line_num)
    line_intercept = np.zeros(line_num)
    for i in range(line_num):
        slope0, intercept0, r_value0, p_value0, std_err0 = linregress(xpoint[i:i+2], ypoint[i:i+2])
        line_slope[i] = slope0
        line_intercept[i] = intercept0

    #------------------------------------------创建模型----------------------------------------
    firm_PH=gp.Model("firm_PH")
    #----------------------------------
    #变量定义
    #N_BS = firm_PH.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='number_of_BS')
    P_Cmp = firm_PH.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='capacity_of_Cmp')
    #P_Elz = firm_PH.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='rated_power_of_Elz')
    V_HS = firm_PH.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='capacity_of_HS')
    Oversizing_PV = firm_PH.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='PV_oversizing_ratio')

    P_PV_toElz = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='P_PV_toElz')    #光伏至电解槽功率
    P_PV_toLd = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='P_PV_toLd')    #光伏至负荷功率
    P_PV_Cur = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='P_PV_Cur')    #弃光功率
    P_PV_toCmp = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='P_PV_Cmp')   #压缩机消耗的电功率
    H_Elz = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='H_Elz')   #制氢量
    H_HS = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='H_HS')   #储氢罐储存氢气量
    H_Demand = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='H_Demand')   #氢气需求量

    P_BS_Ch = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='P_BS_ch')    #电池储能充电功率
    P_BS_plus = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='P_BS_ch')    #考虑电池储能充电效率之后的储能充电功率
    P_BS_Dis = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='P_BS_Dis')    #电池储能放电功率
    P_BS_minus = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='P_BS_Dis')    #考虑电池储能放电效率之前的储能放电功率
    E_BS = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='E_BS')    #电池可用能量 kWh
    SoC_BS = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='SoC_BS')    #电池荷电状态 kWh
    Xtj = firm_PH.addVars(Time, N_j, vtype=GRB.CONTINUOUS, name='Xtj')    #充电功率权重
    Xtk = firm_PH.addVars(Time, N_k, vtype=GRB.CONTINUOUS, name='XtK')    #放电功率权重

    P_Elz_disaggregate = firm_PH.addVars(Time*line_num, 1, vtype=GRB.CONTINUOUS, name='P_Elz_disaggregate')    ##电解槽制氢消耗功率分解项
    B_Elz = firm_PH.addVars(Time*line_num, 1, vtype=GRB.BINARY, name='B_Elz')  #制氢效率曲线的二进制变量
    #--------------------------------------------------------------------------
    #目标函数
    obj = LinExpr()
    obj.addConstant((xi_Elz*c_Elz + OM_Elz*c_Elz)*P_Elz)    #电解槽等年值成本、等年值运维成本
    #obj.addTerms(xi_Elz*c_Elz + OM_Elz*c_Elz, P_Elz)   #电解槽等年值成本、等年值运维成本
    obj.addTerms(xi_Cmp*c_Cmp + OM_Cmp*c_Cmp, P_Cmp)   #压缩机等年值成本、等年值运维成本
    obj.addTerms(xi_HS*c_HS + OM_HS*c_HS, V_HS)        #储氢罐等年值成本，等年值运维成本
    obj.addTerms(xi_PV*c_PV*Cap_PV + OM_PV*c_PV*Cap_PV, Oversizing_PV)   #光伏电站等年值成本，等年值运维成本
    obj.addConstant(xi_BS*c_BS*one_battery_capacity*N_BS)  #电池储能等年值投资成本
    for t in range(Time):
        obj.addTerms(OM_BS*c_BS, P_BS_Ch[t,0])  #电池储能的运维成本
    for t in range(Time):
        obj.addTerms(-hydrogen_price, H_Demand[t,0])       #售氢收入

    #--------------------------------------------------------------------------
    #添加约束
    #光伏扩建倍数约束
    firm_PH.addConstr((Oversizing_PV >= 1), name='oversizing_cons')
    firm_PH.addConstr((Oversizing_PV <= 10), name='oversizing_cons')
    #光伏出力平衡约束
    for t in range(Time):
        firm_PH.addConstr((P_BS_Ch[t,0] + P_PV_toElz[t,0] + P_PV_toLd[t,0] + P_PV_toCmp[t,0] + P_PV_Cur[t,0]  == Oversizing_PV*P_PV[t]), name='PV_cons_%s' % (t))
        firm_PH.addConstr((P_PV_toElz[t,0] <= Oversizing_PV*P_PV[t]), name='PV_cons1_%s' % (t))
        firm_PH.addConstr((P_PV_toLd[t,0] <= Oversizing_PV*P_PV[t]), name='PV_cons2_%s' % (t))
        firm_PH.addConstr((P_PV_Cur[t,0] <= Oversizing_PV*P_PV[t]), name='PV_cons3_%s' % (t))
        firm_PH.addConstr((P_BS_Ch[t,0] <= Oversizing_PV*P_PV[t]), name='PV_cons4_%s' % (t))
        firm_PH.addConstr((P_PV_toCmp[t,0] <= Oversizing_PV*P_PV[t]), name='PV_cons5_%s' % (t))
    #负荷平衡约束
    for t in range(Time):
        firm_PH.addConstr((P_BS_Dis[t,0] + P_PV_toLd[t,0]  == P_Ld[t]), name='Load_cons_%s' % (t))
        firm_PH.addConstr((P_BS_Dis[t,0] <= P_Ld[t]), name='Load_cons1_%s' % (t))

    #电解槽约束
    for t in range(Time):
        expr = LinExpr()
        for k in range(line_num):
            expr.addTerms(1, P_Elz_disaggregate[line_num*t + k, 0])
        firm_PH.addConstr((P_PV_toElz[t,0] == expr), name='cons1_Elz_%s' %(t))
    for t in range(Time):
        expr = LinExpr()
        for k in range(line_num):
            expr.addTerms(1, B_Elz[line_num*t + k, 0])
        firm_PH.addConstr((expr <= 1), name='cons2_Elz_%s' %(t))
    for t in range(Time):
        for k in range(line_num):
            firm_PH.addConstr((P_Elz_disaggregate[line_num*t + k, 0] <= xpoint[k+1] * P_Elz * B_Elz[line_num*t + k, 0]), name='cons3_Elz_%s_%s' %(t, k))
            firm_PH.addConstr((P_Elz_disaggregate[line_num*t + k, 0] >= xpoint[k] * P_Elz * B_Elz[line_num*t + k, 0]), name='cons4_Elz_%s_%s' %(t, k))
    for t in range(Time):
        expr = LinExpr()
        for k in range(line_num):
            expr.addTerms(line_slope[k], P_Elz_disaggregate[t*line_num+k, 0])
            expr.addTerms(line_intercept[k], B_Elz[t*line_num+k, 0])
        firm_PH.addConstr((H_Elz[t,0] ==expr),name='cons5_Elz_%s'%(t))
    for t in range(Time):
        firm_PH.addConstr((P_PV_toElz[t,0] <= P_Elz), name='operation_Cons_Elz_%s' %(t))
    #firm_PH.addConstr((P_Elz == 0), name='operation_Cons_Elz_%s' %(t))    
    #压缩机约束
    for t in range(Time):
        firm_PH.addConstr((P_PV_toCmp[t,0] == P_Cmp0*H_Elz[t,0]*math.log(Pre_Cmp/Pre0)/math.log(Pre_Cmp0/Pre0)), name='operation_Cons_Cmp_%s' %(t))
        firm_PH.addConstr((P_PV_toCmp[t,0] <= P_Cmp), name='Power_limit_Cons_Cmp_%s' %(t))
        
    #储氢罐约束
    for t in range(Time-1):
        firm_PH.addConstr((Niu_H*H_HS[t+1,0] == Niu_H*H_HS[t,0] + delt_Time*(H_Elz[t,0] - H_Demand[t,0])), name='Operation_Cons_HS_%s' %(t))
    firm_PH.addConstr((Niu_H*H_HS[0,0] == Niu_H*H_HS[Time-1,0] + delt_Time*(H_Elz[Time-1,0] - H_Demand[Time-1,0])), name='Operation_Cons_HS_%s' %(8759))    
    for t in range(Time):
        firm_PH.addConstr((H_HS[t,0] <= V_HS), name='Storage_level_Cons_HS_%s' %(t))
    firm_PH.addConstr((H_HS[0,0] == 0), name='Initial_hydrogen_storage_level')
    for t in range(Time):
        if (t+1) %24 ==0:
            firm_PH.addConstr((H_Demand[t,0] == Niu_H*H_HS[t,0] + H_Elz[t,0]), name='Hydrogen_demand_%s' %(t))
        else:
            firm_PH.addConstr((H_Demand[t,0] == 0), name='Hydrogen_demand_%s' %(t))

    #电池储能约束
    for t in range(Time-1):
        firm_PH.addConstr((E_BS[t+1,0]  == E_BS[t,0] + P_BS_plus[t,0] - P_BS_minus[t,0]), name= 'BS_balance_cons_%s' %(t))   #式18
    firm_PH.addConstr((E_BS[0,0] == E_BS[Time-1,0] + P_BS_plus[Time-1,0] - P_BS_minus[Time-1,0]), name= 'BS_balance_cons_8759')

    for t in range(Time):
        expr = LinExpr()
        for k in range(N_k):
            expr.addTerms(Pplus_k[k]*N_BS, Xtk[t,k])
        firm_PH.addConstr((expr == P_BS_plus[t,0]), name= 'Xtk_cons1_%s'%(t))    #式19 
    for t in range(Time):
        expr = LinExpr()
        for k in range(N_k):
            expr.addTerms(Pcha_k[k]*N_BS, Xtk[t,k])
        firm_PH.addConstr((expr == P_BS_Ch[t,0]), name= 'Xtk_cons2_%s'%(t))   #式20

    for t in range(Time):
        expr = LinExpr()
        for j in range(N_j):
            expr.addTerms(SoC_j[j], Xtj[t,j])
        for k in range(N_k):
            expr.addTerms(SoC_k[k], Xtk[t,k])
        firm_PH.addConstr((expr == SoC_BS[t,0]), name= 'BS_SoC_cons1_%s'%(t)) #式21

    for t in range(Time):
        expr = LinExpr()
        for k in range(N_k):
            expr.addTerms(1, Xtk[t,k])
        firm_PH.addConstr((expr == 1), name= 'Xtk_cons3_%s'%(t))  #式22
    for t in range(Time):
        expr = LinExpr()
        for j in range(N_j):
            expr.addTerms(Pminus_j[j]*N_BS, Xtj[t,j])
        firm_PH.addConstr((expr == P_BS_minus[t,0]), name= 'Xtj_cons1_%s'%(t))   #式23   
    for t in range(Time):
        expr = LinExpr()
        for j in range(N_j):
            expr.addTerms(Pdis_j[j]*N_BS, Xtj[t,j])
        firm_PH.addConstr((expr == P_BS_Dis[t,0]), name= 'Xtj_cons2_%s'%(t))   #式24    
    for t in range(Time):
        expr = LinExpr()
        for j in range(N_j):
            expr.addTerms(1, Xtj[t,j])
        firm_PH.addConstr((expr == 1), name= 'Xtj_cons3_%s'%(t))    #式25
    for t in range(Time):
        firm_PH.addConstr((E_BS[t,0] == SoC_BS[t,0]*one_battery_capacity*N_BS), name= 'BS_energy_cons_%s'%(t))    #式26
    for t in range(Time):
        firm_PH.addConstr((E_BS[t,0] <= one_battery_capacity * N_BS), name= 'BS_SoC_cons2_%s' %(t))  #式27
    firm_PH.addConstr((E_BS[0,0] == 0.8 * one_battery_capacity * N_BS), name= 'Initial_SoC_cons')   #式16
    for t in range(Time):
        firm_PH.addConstr((SoC_BS[t,0] <= 1), name= 'BS_SoC_cons3_%s'%(t)) #式15

    # 调用求解器
    firm_PH.setObjective(obj, GRB.MINIMIZE)
    firm_PH.setParam('MIPFocus', 2)  # 将焦点集中在寻找更好的下界
    firm_PH.optimize()
    if firm_PH.status == 2:
        obj = firm_PH.objVal   #目标函数 
    else:
        obj = 10**10
    return obj



# 初始化粒子群参数，每个粒子的速度和位置边界
N = 20 # 种群数目
D = 2 # 维度
T = 50 # 最大迭代次数
c1 = c2 = 1.5 # 个体学习因子与群体学习因子
w_max = 0.8 # 权重系数最大值
w_min = 0.4 # 权重系数最小值

x_max = [0, 0] # 每个维度最大取值范围
x_max[0] = Cap_PV  #电解槽最大额定功率不超过光伏初始装机容量
x_max[1] = 80000 #电池数量上限值
x_min = [0, 0] # 每个维度最小取值范围


v_max = [0,0]  # 每个维度粒子的最大速度
v_max[0] = 0.1 * x_max[0] 
v_max[1] = 0.1 * x_max[1]
v_min = [0,0] # 每个维度粒子的最小速度
v_min[0] = -0.1 * x_max[0] 
v_min[1] = -0.1 * x_max[1]


# 初始化种群个体
# 初始化每个粒子的位置
x = np.random.rand(N, D)
for i in range(N):
    for j in range(D):
        x[i,j] = x[i,j] * (x_max[j] - x_min[j]) + x_min[j]
# 初始化每个粒子的速度
v = np.random.rand(N, D)
for i in range(N):
    for j in range(D):
        v[i,j] = v[i,j] * (v_max[j] - v_min[j]) + v_min[j]
        
# 初始化个体最优位置和最优值
p = x.copy() # 用来存储每一个粒子的历史最优位置
p_best = np.zeros((N, 1))  # 每行存储的是最优值
for i in range(N): # 初始化每个粒子的最优值，此时就是把位置带进去，把适应度值计算出来
    p_best[i] = PSO(x[i, 0], x[i, 1])
    time.sleep(0.2)
    print('---------------------------------------------')

# 初始化全局最优位置和全局最优值
gb = np.zeros(T) # 用于记录每一次迭代的全局最优值
g_best = 10**10 #设置全局最优值
x_best = np.zeros(D) # 用于存储最优粒子的取值
for i in range(N):
    if g_best > p_best[i]:
        g_bset = p_best[i].copy()
        x_best = x[i,:].copy()
# 按照公式依次迭代直到满足精度或者迭代次数
for i in range(T):
    for j in range(N):
        # 计算动态惯性权重
        w = w_max - (w_max - w_min) * i / T
        # 更新速度
        v[j, :] = w * v[j, :] + c1 * np.random.rand(1) * (p[j, :] - x[j, :]) + c2 * np.random.rand(1) * (x_best - x[j, :])
        # 速度边界条件处理
        for k in range(D):
            if (v[j, k] > v_max[k]) or (v[j, k] < v_min[k]):
                v[j, k] = v_min[k] + np.random.rand(1) * (v_max[k] - v_min[k])
        # 更新位置
        x[j, :] = x[j, :] + v[j, :]
        # 位置边界条件处理
        for k in range(D):
            if (x[j, k] > x_max[k]) or (x[j, k] < x_min[k]):
                x[j, k] = x_min[k] + np.random.rand(1) * (x_max[k] - x_min[k])
        # 更新个体最优值和全局最优值
        print("--------迭代次数为：{0}---------".format(i))
        print("--------粒子数为：{0}---------".format(j))
        temp_p = PSO(x[j, 0], x[j, 1])
        print('---------------------------------------------')
        time.sleep(0.2)
        if p_best[j] > temp_p:
            p_best[j] = temp_p
            p[j,:] = x[j,:].copy()
        if g_best > p_best[j]:
            g_best = p_best[j].copy()
            x_best = x[j,:].copy()   # 一定要加copy，否则后面x[j,:]更新也会将x_best更新        
    # 记录历代全局最优值
    gb[i] = g_best
    time.sleep(5)
print("最优值为", gb[T - 1], "最优位置为", x_best)
#保存数据
workbook = xlsxwriter.Workbook('D:/Doctor/paper/2023third/Python_Code/PSO_results.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, 'g_best')
#写入数据
for i in range(T):
    worksheet.write(i+1, 0, gb[i])   
#写入标题
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, 'P_Elz(kW)')
worksheet.write(0, 1, 'N_BS')
worksheet.write(1, 0, x_best[0])
worksheet.write(1, 1, x_best[1])   
workbook.close()  

