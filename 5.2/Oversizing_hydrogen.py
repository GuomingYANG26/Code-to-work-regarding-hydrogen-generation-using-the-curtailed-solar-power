 -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:42:50 2023
@email: yangguoming1995@gmail.com
@author: YANG Guoming
"""

#####---------采用传统方法对灵活光伏电站、电池储能和电解槽进行建模,有氢气系统------------------
import pandas as pd
from gurobipy import GRB
import pvlib
from gurobipy import * 
import gurobipy as gp
import numpy as np
import math
import xlsxwriter
import pwlf
from scipy.stats import linregress
###--------------------------模型参数---------------------------------------######
tao = 8/100   #贴现率
c_Elz, OM_Elz, Y_Elz = 1027.5, 5/100, 10 #电解槽单位投资成本 $/kW, 运维成本因子值, 寿命 yr.
xi_Elz = tao*(1+tao)**Y_Elz /((1+tao)**Y_Elz - 1)  #电解槽资本回收系数
c_Cmp, OM_Cmp, Y_Cmp = 730, 1/100, 20 #压缩机单位投资成本 $/kW, 运维成本因子值, 寿命 yr.
xi_Cmp = tao*(1+tao)**Y_Cmp /((1+tao)**Y_Cmp - 1)  #压缩机资本回收系数
c_HS, OM_HS, Y_HS = 9292.5, 1/100, 20  #储氢罐单位投资成本 $/m^3, 运维成本因子值, 寿命 yr.
xi_HS = tao*(1+tao)**Y_HS /((1+tao)**Y_HS - 1)  #储氢罐资本回收系数
c_BS, OM_BS, Y_BS, Cap_OneBS = 137, 0.02/100, 15, 5.32  #电池单位投资成本 $/kWh, 单次充放电运维成本因子值，寿命 yr., 单个电池储能额定容量 kWh
xi_BS = tao*(1+tao)**Y_BS/((1+tao)**Y_BS - 1) #电池资本回收系数
c_PV, OM_PV, Y_PV = 833, 1/100, 30  #光伏电站单位投资成本  $/kW, 运维成本因子值, 寿命 
xi_PV = tao*(1+tao)**Y_PV / ((1+tao)**Y_PV - 1)  #光伏电站资本回收系数
eta_Elz = 67/100     #电解槽制氢效率 % 制氢效率曲线的平均值
hydrogen_price = 5   #氢气售价 $/kg
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

#负荷需求
rou_HP = 2   #Heat pump供应热负荷
rou_EC = 4   #Electric chiller供应冷负荷
Load_data = pd.read_csv('D:/Doctor/paper/2023third/Python_Code/723740_TMY3_BASE.csv', index_col=0)  #单栋居民住宅电力负荷数据 kW
P_Ld = 400 * (Load_data['Fans:Electricity [kW](Hourly)'] + Load_data['General:InteriorLights:Electricity [kW](Hourly)'] + \
                     Load_data['General:ExteriorLights:Electricity [kW](Hourly)'] + Load_data['Appl:InteriorEquipment:Electricity [kW](Hourly)'] + \
                     Load_data['Misc:InteriorEquipment:Electricity [kW](Hourly)'] + Load_data['Cooling:Electricity [kW](Hourly)']/rou_EC + \
                     Load_data['Heating:Gas [kW](Hourly)']/rou_HP)
P_Ld.index = range(8760)

  

def curtailment_multiplier (multiplier):
    #光伏出力
    P_power = pd.read_csv('D:/Doctor/paper/2023third/Python_Code/pv_traditional.csv', index_col=0)   #传统光伏建模模拟的1MW光伏电站的光伏出力 单位W
    P_power = P_power.iloc[:,0].values/1000   #单位kW
    #reads the TMY data downloaded from NSRDB for Harbin  
    with open('D:/Doctor/paper/2023third/Python_Code/tmy_2020.csv', 'r') as f:
        data, metadata =  pvlib.iotools.psm3.parse_psm3(f, map_variables=True)
    #以1 MW光伏电站的模拟出力为基准，确定光伏初始装机容量下的出力
    PGF = 0.621*np.sum(data['ghi'])/365/1000  #光伏板生成因子
    Cap_PV = np.sum(P_Ld)/365 /PGF  #初始装机容量
    P_PV = P_power/1000*Cap_PV   #单位kW  
    #-------------------------------创建模型-----------------------------
    firm_PH=gp.Model("firm_PH")
    #----------------------------------
    #变量定义
    N_BS = firm_PH.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='number_of_BS')
    P_Cmp = firm_PH.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='capacity_of_Cmp')
    P_Elz = firm_PH.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='rated_power_of_Elz')
    V_HS = firm_PH.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='capacity_of_HS')
    Oversizing_PV = multiplier
    #Oversizing_PV = firm_PH.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='PV_oversizing_ratio')

    P_BS_Ch = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='P_BS_ch')    #电池储能充电功率
    P_BS_Dis = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='P_BS_Dis')    #电池储能放电功率
    SoC_BS = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='SoC_BS')    #电池可用能量 kWh
    P_PV_toElz = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='P_PV_toElz')    #光伏至电解槽功率
    P_PV_toLd = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='P_PV_toLd')    #光伏至负荷功率
    P_PV_Cur = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='P_PV_Cur')    #弃光功率
    P_PV_toCmp = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='P_PV_Cmp')   #压缩机消耗的电功率
    H_Elz = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='H_Elz')   #制氢量
    H_HS = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='H_HS')   #储氢罐储存氢气量
    H_Demand = firm_PH.addVars(Time, 1, vtype=GRB.CONTINUOUS, name='H_Demand')   #氢气需求量

    BS_Ch = firm_PH.addVars(Time, 1, vtype=GRB.BINARY, name='BS_Ch')    #表示电池储能充电的二进制变量
    BS_Dis = firm_PH.addVars(Time, 1, vtype=GRB.BINARY, name='BS_Dis')  #表示电池储能放电的二进制变量
    #----------------------------------
    #目标函数
    obj = LinExpr()
    obj.addTerms(xi_Elz*c_Elz + OM_Elz*c_Elz, P_Elz)   #电解槽等年值成本、等年值运维成本
    obj.addTerms(xi_Cmp*c_Cmp + OM_Cmp*c_Cmp, P_Cmp)   #压缩机等年值成本、等年值运维成本
    obj.addTerms(xi_HS*c_HS + OM_HS*c_HS, V_HS)        #储氢罐等年值成本，等年值运维成本
    obj.addConstant((xi_PV*c_PV*Cap_PV + OM_PV*c_PV*Cap_PV)*Oversizing_PV)   #光伏电站等年值成本，等年值运维成本
    obj.addTerms(xi_BS*c_BS*Cap_OneBS, N_BS)  #电池储能等年值投资成本
    for t in range(Time):
        obj.addTerms(OM_BS*c_BS, P_BS_Ch[t,0])  #电池储能的运维成本
    for t in range(Time):
        obj.addTerms(-hydrogen_price, H_Demand[t,0])       #售氢收入

    #-----------------------------------------------
    #添加约束
    #光伏扩建倍数约束
    #firm_PH.addConstr((Oversizing_PV >= 1), name='oversizing_cons')
    #firm_PH.addConstr((Oversizing_PV <= 10), name='oversizing_cons')
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
        firm_PH.addConstr((H_Elz[t,0] == P_PV_toElz[t,0]*eta_Elz/Hh), name='hydrogen_production_%s' % (t))
        firm_PH.addConstr((P_PV_toElz[t,0] <= P_Elz), name='operation_Cons_Elz_%s' %(t))
    #firm_PH.addConstr((P_Elz == 0), name='operation_Cons_Elz2')
      
    #压缩机约束
    for t in range(Time):
        firm_PH.addConstr((P_PV_toCmp[t,0] == P_Cmp0*H_Elz[t,0]*math.log(Pre_Cmp/Pre0)/math.log(Pre_Cmp0/Pre0)), name='operation_Cons_Cmp_%s' %(t))
        firm_PH.addConstr((P_PV_toCmp[t,0] <= P_Cmp), name='Power_limit_Cons_Cmp_%s' %(t))
        
    #储氢罐约束
    for t in range(Time-1):
        firm_PH.addConstr((Niu_H*H_HS[t+1,0] == Niu_H*H_HS[t,0] + delt_Time*(H_Elz[t,0] - H_Demand[t,0])), name='Operation_Cons_HS_%s' %(t))
    firm_PH.addConstr((Niu_H*H_HS[0,0] == Niu_H*H_HS[8759,0] + delt_Time*(H_Elz[8759,0] - H_Demand[8759,0])), name='Operation_Cons_HS_%s' %(8759))    
    for t in range(Time):
        firm_PH.addConstr((H_HS[t,0] <= V_HS), name='Storage_level_Cons_HS_%s' %(t))
    firm_PH.addConstr((H_HS[0,0] == 0), name='Initial_hydrogen_storage_level')
    for t in range(Time):
        if (t+1) %24 ==0:
            firm_PH.addConstr((H_Demand[t,0] == Niu_H*H_HS[t,0] + H_Elz[t,0]), name='Hydrogen_demand_%s' %(t))
        else:
            firm_PH.addConstr((H_Demand[t,0] == 0), name='Hydrogen_demand_%s' %(t))

    #电池储能约束
    firm_PH.addConstr((SoC_BS[0,0] == 0.8 * N_BS * Cap_OneBS), name='BS_cons1')
    for t in range(Time-1):
        firm_PH.addConstr((SoC_BS[t+1,0] == (1 - sig_BS) * SoC_BS[t,0] + eta_Ch_BS * P_BS_Ch[t,0] - P_BS_Dis[t,0]/eta_Dis_BS ), name='BS_cons2_%s' %(t))
    firm_PH.addConstr((SoC_BS[0,0] == (1 - sig_BS) * SoC_BS[8759,0] + eta_Ch_BS * P_BS_Ch[8759,0] - P_BS_Dis[8759,0]/eta_Dis_BS), name='BS_cons2_%8759')
    for t in range(Time):
        firm_PH.addConstr((SoC_BS[t,0] <=  N_BS * Cap_OneBS ), name='BS_cons3_%s' %(t))
        firm_PH.addConstr((BS_Ch[t,0] + BS_Dis[t,0] <=  1 ), name='BS_cons4_%s' %(t))
        firm_PH.addConstr((P_BS_Ch[t,0] <=  0.25 * BS_Ch[t,0]*N_BS * Cap_OneBS), name='BS_cons5_%s' %(t))
        firm_PH.addConstr((P_BS_Dis[t,0] <=  0.25 * BS_Dis[t,0]*N_BS * Cap_OneBS), name='BS_cons6_%s' %(t))
    #firm_PH.addConstr((P_Elz == 0), name='ELZ_cons%s')

     # 调用求解器
    firm_PH.setObjective(obj, GRB.MINIMIZE)
    firm_PH.setParam('MIPFocus', 2)  # 将焦点集中在寻找更好的下界
    firm_PH.optimize()
    P_BS_Ch_value = np.zeros((Time))  #读取优化结果的数据
    H_Demand_value = np.zeros((Time))
    for t in range(Time):
        P_BS_Ch_value[t] = P_BS_Ch[t,0].X
        H_Demand_value[t] = H_Demand[t,0].X 
       
    obj = firm_PH.ObjVal #目标函数 $
    P_Elz_value = P_Elz.X #电解槽额定功率 kW
    P_Cmp_value = P_Cmp.X #压缩机额定功率 kW
    V_HS_value = V_HS.X  #储氢罐容量 kg
    N_BS_value = N_BS.X*Cap_OneBS/1000 #电池储能容量 MWh
    P_PV_value = Cap_PV*Oversizing_PV #光伏电站额定功率 MW
    Hydrogen_sale = sum(H_Demand_value)*hydrogen_price #售氢收入 $
    Hydrogen_sale_contribution =   -Hydrogen_sale / sum(P_Ld) / ((xi_PV*c_PV*Cap_PV + OM_PV*c_PV*Cap_PV)/sum(P_PV)) #售氢收入占比
    PV_cost = (xi_PV*c_PV*Cap_PV + OM_PV*c_PV*Cap_PV)*Oversizing_PV #光伏电站等年值成本 $
    PV_cost_contribution = PV_cost / sum(P_Ld) / ((xi_PV*c_PV*Cap_PV + OM_PV*c_PV*Cap_PV)/sum(P_PV)) #光伏电站成本占比
    Battery_cost = xi_BS*c_BS*Cap_OneBS * N_BS.X + OM_BS*c_BS*sum(P_BS_Ch_value) #电池储能等年值成本 $
    Battery_cost_contribution = Battery_cost / sum(P_Ld) / ((xi_PV*c_PV*Cap_PV + OM_PV*c_PV*Cap_PV)/sum(P_PV))  #电池成本占比
    Hydrogen_cost = (xi_Elz*c_Elz + OM_Elz*c_Elz)*P_Elz.X + (xi_Cmp*c_Cmp + OM_Cmp*c_Cmp)*P_Cmp.X + (xi_HS*c_HS + OM_HS*c_HS)*V_HS.X #氢系统等年值成本
    Hydrogen_cost_contribution = Hydrogen_cost / sum(P_Ld) / ((xi_PV*c_PV*Cap_PV + OM_PV*c_PV*Cap_PV)/sum(P_PV)) #氢系统成本占比
    Firm_kWh_Premium = firm_PH.ObjVal/sum(P_Ld) / ((xi_PV*c_PV*Cap_PV + OM_PV*c_PV*Cap_PV)/sum(P_PV)) #稳固千瓦时溢价
    return  obj, P_Elz_value, P_Cmp_value, V_HS_value, N_BS_value, P_PV_value, Hydrogen_sale, Hydrogen_sale_contribution, PV_cost, PV_cost_contribution, Battery_cost, Battery_cost_contribution, Hydrogen_cost, Hydrogen_cost_contribution, Firm_kWh_Premium




    



##----程序正式开始------
length_1_number = 201
length_2_number = 51
multiplier1 = np.linspace(1, 3, length_1_number)
multiplier2 = np.linspace(3, 8, length_2_number)
multiplier2 = np.delete(multiplier2,[0])    #删除第一个元素
multiplier = np.hstack((multiplier1, multiplier2))  #合并两个array
number = length_1_number + length_2_number - 1

obj_0 = np.zeros(number) 
P_Elz_value_0 = np.zeros(number)
P_Cmp_value_0 = np.zeros(number)
V_HS_value_0 = np.zeros(number)
N_BS_value_0 = np.zeros(number)
P_PV_value_0 = np.zeros(number) 
Hydrogen_sale_0 = np.zeros(number)
Hydrogen_sale_contribution_0 = np.zeros(number)
PV_cost_0 = np.zeros(number)
PV_cost_contribution_0 = np.zeros(number)
Battery_cost_0 = np.zeros(number)
Battery_cost_contribution_0 = np.zeros(number)
Hydrogen_cost_0 = np.zeros(number) 
Hydrogen_cost_contribution_0 = np.zeros(number) 
Firm_kWh_Premium_0 = np.zeros(number)


for n in range(number):
    print('--------------------------------------------------------------------')
    print(n)
    obj_0[n],  P_Elz_value_0[n], P_Cmp_value_0[n], V_HS_value_0[n], N_BS_value_0[n], P_PV_value_0[n], Hydrogen_sale_0[n], Hydrogen_sale_contribution_0[n], \
        PV_cost_0[n], PV_cost_contribution_0[n], Battery_cost_0[n], Battery_cost_contribution_0[n], Hydrogen_cost_0[n], Hydrogen_cost_contribution_0[n], Firm_kWh_Premium_0[n] = curtailment_multiplier (multiplier[n])
#保存数据
workbook = xlsxwriter.Workbook('D:/Doctor/paper/2023third/Last_Code/5.2/Oversizing_hydrogen.xlsx')
worksheet = workbook.add_worksheet()
#写入行名
row = 0
col = 0
worksheet.write(row, col, 'Objective_function_value')
worksheet.write(row, col+1, 'P_Elz(kW)')
worksheet.write(row, col+2, 'P_cmp(kW)')
worksheet.write(row, col+3, 'V_HS(kg)')
worksheet.write(row, col+4, 'N_BS(MWh)')
worksheet.write(row, col+5, 'P_PV(MW)')
worksheet.write(row, col+6, 'Hydrogen_sale')
worksheet.write(row, col+7, 'Hydrogen_sale_contribution')
worksheet.write(row, col+8, 'PV_cost')
worksheet.write(row, col+9, 'PV_cost_contribution')
worksheet.write(row, col+10, 'Battery_cost')
worksheet.write(row, col+11, 'Battery_cost_contribution')
worksheet.write(row, col+12, 'Hydrogen_cost')
worksheet.write(row, col+13, 'Hydrogen_cost_contribution')
worksheet.write(row, col+14, 'Firm_kWh_Premium')
worksheet.write(row, col+15, 'Oversizing')


row += 1
#一行行写入数据
length = number
buffer = 0
while buffer < length:
    worksheet.write(row, col, obj_0[buffer])
    worksheet.write(row, col+1, P_Elz_value_0[buffer])
    worksheet.write(row, col+2, P_Cmp_value_0[buffer])
    worksheet.write(row, col+3, V_HS_value_0[buffer])
    worksheet.write(row, col+4, N_BS_value_0[buffer])
    worksheet.write(row, col+5, P_PV_value_0[buffer])
    worksheet.write(row, col+6, Hydrogen_sale_0[buffer])
    worksheet.write(row, col+7, Hydrogen_sale_contribution_0[buffer])
    worksheet.write(row, col+8, PV_cost_0[buffer])
    worksheet.write(row, col+9, PV_cost_contribution_0[buffer])
    worksheet.write(row, col+10, Battery_cost_0[buffer])
    worksheet.write(row, col+11, Battery_cost_contribution_0[buffer])
    worksheet.write(row, col+12, Hydrogen_cost_0[buffer])
    worksheet.write(row, col+13, Hydrogen_cost_contribution_0[buffer])
    worksheet.write(row, col+14, Firm_kWh_Premium_0[buffer])
    worksheet.write(row, col+15, multiplier[buffer])
    row += 1
    buffer += 1
workbook.close()
    