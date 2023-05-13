#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 18:23:51 2023

@author: sunyenpeng
"""
import pandas as pd
import numpy as np
import pulp as pl
import plotly.graph_objects as go
import plotly.io as pio
import random
from geopy.distance import geodesic
pio.renderers.default='browser'
#Loading the data
#population data: for calculating the demand of each local authorities
populationdata=pd.read_excel("ukpopestimatesmid2021on2021geographyfinal.xls", sheet_name="MYE2 - Persons",skiprows=7)
#LA data: for lontitude and latitude
localauthorities=pd.read_csv("Local_Authority_Districts_(December_2022)_Boundaries_UK_BFC.csv")
#facility data
facilities=pd.read_excel("facilities.xlsx")
facilities["New_name"]=["North Lanarkshire",'County Durham','Halton','Doncaster','Sandwell','Peterborough','Bridgend','Bristol, City of','Enfield','Bexley','Gravesham','Exeter','Southampton']
#population=populationdata.iloc[0:5,:]

#Data preprocessing
#Generate a dataset called "la", which contains the code, name, longitude, latitude of each localauthorities
code=populationdata["Code"]#Use code to subset essential local authorities(customers)
la=localauthorities[localauthorities["LAD22CD"].isin(code)]# Use the unique code to filter out local authorities
la=la[["LAD22CD","LAD22NM","LONG","LAT"]]#Subset the useful data, Code, Name, Longitude and latitude of each LA


#Generate capacity for each facility
capacity=facilities[["Capacity (1000 visits)","Name"]].set_index("Name")
capacity=capacity*1000

#Longititude and latitude of the facilities
fall=la[la.LAD22NM.isin(facilities.New_name)][["LAD22NM","LONG","LAT"]] #Match the name of facility and la, so we don't need to find every facility's longitude and latitude
fdata = pd.merge(facilities, fall, how="left",left_on=['New_name'],right_on=['LAD22NM'])
fdata=fdata.set_index("Name")

pop=populationdata[populationdata["Code"].isin(localauthorities.LAD22CD)][['Name',"Code","All ages"]]
pop['Demand']=pop['All ages']*0.001
pop=pop.drop("All ages",axis=1).set_index("Code")


"""Using a**2+b**2=c**2 to calculate the distance between each customer and facility"""

customerdistance =pd.DataFrame(index=fdata.index,columns=localauthorities.LAD22CD)

for i in range(len(localauthorities)):
    pop_ = localauthorities.iloc[i]
    for j in range(len(fdata)):
        facility = fdata.iloc[j]
        distance = geodesic((pop_['LAT'], pop_['LONG']),
                            (facility['LAT'], facility['LONG'])).km/60
        customerdistance.iloc[j,i]=distance
cd=[]
for i in customerdistance.columns:
    for j in customerdistance.index:
        val=customerdistance.loc[j,i]
        cd.append(val)
"""Data preparation for Heuristic algorithm"""
#Transform all data into dictionary

popname=pop.index.tolist()
popdemand=pop.Demand.tolist()
demand_dict = {(n): d for n, d in zip(popname,popdemand)}

cn=capacity.index.tolist()
cc=capacity["Capacity (1000 visits)"].tolist()
capacity_dict = {(n): c for n, c in zip(cn,cc)}

NM=localauthorities.LAD22CD.tolist()
FN=["Motherwell" for i in range(374)]+["Newton Aycliffe" for i in range(374)]+["Runcorn" for i in range(374)]+["Doncaster" for i in range(374)]+["Wednesbury" for i in range(374)]+["Peterborough" for i in range(374)]+["Bridgend" for i in range(374)]+["Avonmouth" for i in range(374)]+["Enfield" for i in range(374)]+["Belvedere" for i in range(374)]+["Northfleet" for i in range(374)]+["Exeter" for i in range(374)]+["Southampton" for i in range(374)]
dist=cd
distance_dict = {(i,j):d for i,j,d in zip(FN,NM*13,dist)}


"""This part is for Constructive Heuristic"""

#Random shuffle the demand dictionary for constructive heuristic method
keys=list(demand_dict.keys())
random.seed(1)#set a seed
random.shuffle(keys)
demand_dict_random = {key: demand_dict[key] for key in keys}

assigned_demand_constructive = []
# Loop over each customer demand and assign them to the nearest available facility
for demand, demand_value in demand_dict_random.items():
    # Initialize a list to store the available facilities for the current demand
    available_facilities = []
    for facility, Capacity in capacity_dict.items():
        # Check if the current facility has enough capacity
        if demand_value <= Capacity:
            # Append the index of the available facility and its distance to the list
            distance = distance_dict[(facility, demand)]
            available_facilities.append((facility, distance))
    # Sort the available facilities by distance in ascending order
            available_facilities_sorted = sorted(available_facilities, key=lambda x: x[1])
    # Loop over the sorted available facilities and assign the demand to the first available facility
    for facility, distance in available_facilities_sorted:
        if demand_value <= capacity_dict[facility]:
            # Assign the demand to the available facility
            assigned_demand_constructive.append((demand, facility))
            # Subtract the demand from the capacity of the assigned facility
            capacity_dict[facility] -= demand_value
            break
# Print the assigned demand and the remaining capacity of each facility
for i, j in capacity_dict.items():
    print(f"Facility {i} has a remaining capacity of {j}")
print("Assigned demand:")
for demand in assigned_demand_constructive:
    print(f"Customer {demand[0]} assigned to Facility {demand[1]}")

#------------Constructive Heuristic ends here--------------#

"""This part is for Random adaptive algorithms"""

popname=pop.index.tolist()
popdemand=pop.Demand.tolist()
demand_dict = {(n): d for n, d in zip(popname,popdemand)}#create a demand dictionary

cn=capacity.index.tolist()
cc=capacity["Capacity (1000 visits)"].tolist()
capacity_dict = {(n): c for n, c in zip(cn,cc)}#create a dictionary of capacity

NM=localauthorities.LAD22CD.tolist()
FN=["Motherwell" for i in range(374)]+["Newton Aycliffe" for i in range(374)]+["Runcorn" for i in range(374)]+["Doncaster" for i in range(374)]+["Wednesbury" for i in range(374)]+["Peterborough" for i in range(374)]+["Bridgend" for i in range(374)]+["Avonmouth" for i in range(374)]+["Enfield" for i in range(374)]+["Belvedere" for i in range(374)]+["Northfleet" for i in range(374)]+["Exeter" for i in range(374)]+["Southampton" for i in range(374)]
dist=cd
distance_dict = {(i,j):d for i,j,d in zip(FN,NM*13,dist)} #create a dictionary of the distance between customers and facilities

# Sort the demand dictionary in descending order
demand_dict_sorted = {k: v for k, v in sorted(demand_dict.items(), key=lambda item: item[1], reverse=True)}

# Set the value of n


# Initialize an empty list to store the assigned demand
assigned_demand_random = []

# Loop until all customers are assigned
while len(demand_dict_sorted) > 0:
    n = random.randint(1, 30)#select a random number to the short list
    # Select the top n customers from the sorted demand dictionary
    selected_customers = list(demand_dict_sorted.items())[:n]
    random.shuffle(selected_customers)#random shuffle the list
    # Remove the selected customers from the demand dictionary
    for customer in selected_customers:
        del demand_dict_sorted[customer[0]]
        
    # Loop over each selected customer and assign them to the nearest available facility
    for demand, demand_value in selected_customers:
        # Initialize a list to store the available facilities for the current demand
        available_facilities = []
        for facility, capacity in capacity_dict.items():
            # Check if the current facility has enough capacity
            if demand_value <= capacity:
                # Append the index of the available facility and its distance to the list
                distance = distance_dict[(facility, demand)]
                available_facilities.append((facility, distance))
        # Sort the available facilities by distance in ascending order
        available_facilities_sorted = sorted(available_facilities, key=lambda x: x[1])
        # Loop over the sorted available facilities and assign the demand to the first available facility
        for facility, distance in available_facilities_sorted:
            if demand_value <= capacity_dict[facility]:
                # Assign the demand to the available facility
                assigned_demand_random.append((demand, facility))
                # Subtract the demand from the capacity of the assigned facility
                capacity_dict[facility] -= demand_value
                break

# Print the assigned demand and the remaining capacity of each facility
for facility, capacity in capacity_dict.items():
    print(f"Facility {facility} has a remaining capacity of {capacity}")
print("Assigned demand:")
for demand in assigned_demand_random:
    print(f"Customer {demand[0]} assigned to Facility {demand[1]}")

#------------Random Adaptive ends here---------------#

f={}#Create an empty dictionary to store the dataframe of each route
for i in fdata.index:
    f[i]=localauthorities[localauthorities.LAD22CD.isin([assigned_demand_random[j][0] for j in range(374) if assigned_demand_random[j][1]==i])][['LONG','LAT','LAD22NM']]
for i in f:
    f[i]["LONG_t"]=fdata.loc[i,"LONG"]
    f[i]["LAT_t"]=fdata.loc[i,"LAT"]
    f[i]["D"]=i
f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13 = [None]*13
for i, key in enumerate(f):
    nf = f[key]
    new_df = nf.copy()
    if i == 0:
        f1 = new_df
    elif i == 1:
        f2 = new_df
    elif i == 2:
        f3 = new_df
    elif i == 3:
        f4 = new_df
    elif i == 4:
        f5 = new_df
    elif i == 5:
        f6 = new_df
    elif i == 6:
        f7 = new_df
    elif i == 7:
        f8 = new_df
    elif i == 8:
        f9 = new_df
    elif i == 9:
        f10 = new_df
    elif i == 10:
        f11 = new_df
    elif i == 11:
        f12 = new_df
    elif i == 12:
        f13 = new_df
p_h=pd.concat([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13])

"""The following part is for first improvement """

"""Modeling part by using Pulp"""
#Reset the capacity for pulp
capacity=facilities[["Capacity (1000 visits)","Name"]].set_index("Name")
capacity=capacity*1000

model=pl.LpProblem("Test_Problem",pl.LpMinimize)

x=pl.LpVariable.dicts("route",(localauthorities.LAD22CD,fdata.index),cat="Binary")#Generate decision variables
#Create a dataframe to store solution of constructive heuristic:
current_allocation = [(x, y, 1) for x, y in assigned_demand_constructive]
current_solution=pd.DataFrame()
for col, row, value in current_allocation:
    current_solution.loc[row, col] = value
current_solution=current_solution.fillna(0)


model+= pl.lpSum(x[j][i]*customerdistance.loc[i,j]*pop.loc[j,"Demand"] for i in fdata.index for j in x)#Objective function

for j in x:
    model+= pl.lpSum(x[j][i] for i in fdata.index)==1 #Restriction of each customer must be served
    
for j in capacity.index:
    model+= pl.lpSum(x[i][j]*pop.loc[i,"Demand"] for i in x)<=capacity.loc[j,"Capacity (1000 visits)"] #Restriction of not exceeding the capacity of each facility
reallocation_cost=sum(demand_dict.values())/len(demand_dict)
##Ignore the last constraint to get the optimal solution
#for i in x:
#    for j in capacity.index:
#        model += pop.loc[i,"Demand"] * customerdistance.loc[j,i] * (current_solution.loc[j,i] - x[i][j]) >= (1 - pl.lpSum([current_solution.loc[k,i] * x[i][k] for k in capacity.index])) * reallocation_cost


model.solve(pl.PULP_CBC_CMD(maxSeconds=10))#Set maximum solution time to 10 seconds
print("Status:", pl.LpStatus[model.status])
print("Optimal Solution will be: ",pl.value(model.objective))#See the optimal value
howmanyroute=0
chosen_route=[]
if (pl.LpStatus[model.status] == 'Optimal'):
    for v in model.variables():
        if v.varValue>0:
            chosen_route.append(v.name) #Obtain the assigned route so that we can plot them
            howmanyroute+=1#Count if every customer is assigned (should be 374 in total)
#print(chosen_route)#Show the assigned route for pulp
chosen_route=pd.Series(chosen_route)

#Data preperation for visualizing solution of optimization method:
d={}
for i in fdata.index:
    d[i]=chosen_route[chosen_route.str.contains(i)].reset_index(drop=True).str.split("_").str[1:-1].str.join(" ")
df={} 
for i in d.keys():
    df[i]=localauthorities[localauthorities.LAD22CD.isin(d[i])][['LONG','LAT','LAD22NM','LAD22CD']]
for i in df:
    df[i]["LONG_t"]=fdata.loc[i,"LONG"]
    df[i]["LAT_t"]=fdata.loc[i,"LAT"]
    df[i]["D"]=i
df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13 = [None]*13
for i, key in enumerate(df):
    ndf = df[key]
    new_df = ndf.copy()
    if i == 0:
        df1 = new_df
    elif i == 1:
        df2 = new_df
    elif i == 2:
        df3 = new_df
    elif i == 3:
        df4 = new_df
    elif i == 4:
        df5 = new_df
    elif i == 5:
        df6 = new_df
    elif i == 6:
        df7 = new_df
    elif i == 7:
        df8 = new_df
    elif i == 8:
        df9 = new_df
    elif i == 9:
        df10 = new_df
    elif i == 10:
        df11 = new_df
    elif i == 11:
        df12 = new_df
    elif i == 12:
        df13 = new_df
p=pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13])


def calculate_objective(demand_dict, assigned_demand):#put assigned_demand_constructive/assigned_demand_random into the second arguemnt
    obj_value = 0
    for demand, demand_value in demand_dict.items():
        assigned_facility = [facility for (demand_, facility) in assigned_demand if demand_ == demand][0]
        distance = distance_dict[(assigned_facility, demand)]
        obj_value += distance * demand_value
    return obj_value

# Implement the first improvement local search algorithm
capacity_dict = {(n): c for n, c in zip(cn,cc)}#Refill the capacity_dict so we know the improvement
def first_improvement_local_search(demand_dict, capacity_dict, distance_dict, assigned_demand, reallocation_cost=sum(demand_dict.values())/len(demand_dict)):
    # Calculate the initial objective value
    best_obj_value = calculate_objective(demand_dict, assigned_demand)
    # Initialize a flag to keep track of whether a better solution was found
    found_better_solution = True
    runtime=0
    while found_better_solution:
        found_better_solution = False
        # Loop over each unassigned demand and try to reassign it to a better facility
        for demand, demand_value in demand_dict.items():
            # Find the facility currently assigned to the demand
            assigned_facility = [facility for (demand_, facility) in assigned_demand if demand_ == demand][0]
            current_distance = distance_dict[(assigned_facility, demand)] * demand_value
            
            # Loop over each available facility and try to find a better one
            for facility, capacity in capacity_dict.items():
                
                # Check if the facility has enough capacity to accommodate the demand
                if demand_value <= capacity:
                    # Check if the facility is different from the current one
                    if facility != assigned_facility:
                        new_distance = distance_dict[(facility, demand)] * demand_value + reallocation_cost
                        
                        # Calculate the new objective value
                        #new_obj_value = best_obj_value - current_distance + new_distance
                        
                        # Check if the new objective value is better than the current one
                        if new_distance < current_distance:
                            runtime+=1
                            new_obj_value = best_obj_value - current_distance + new_distance
                            # Assign the demand to the new facility
                            assigned_demand.remove((demand, assigned_facility))
                            assigned_demand.append((demand, facility))
                            
                            # Subtract the demand from the capacity of the new facility and add it to the old one
                            capacity_dict[assigned_facility] += demand_value
                            capacity_dict[facility] -= demand_value
                            
                            # Update the best objective value
                            best_obj_value = new_obj_value
                            
                            # Set the flag to indicate that a better solution was found
                            found_better_solution = True
                            
                            # Break out of the inner loop and start over
                            break
                        
                # If a better facility was found, break out of the inner loop and start over
                if found_better_solution:
                    break
                
            # If a better facility was found, break out of the outer loop and start over
            if found_better_solution:
                break
                
    # Return the best solution
    return assigned_demand,runtime, best_obj_value

og_random=calculate_objective(demand_dict, assigned_demand_random)#Original value of random adaptive
og_construct=calculate_objective(demand_dict, assigned_demand_constructive)#Original value of constructive heuristic

capacity_dict = {(n): c*1.1 for n, c in zip(cn,cc)}#Relaxation of capacity (additional 10%)
first_random=first_improvement_local_search(demand_dict, capacity_dict, distance_dict, assigned_demand_random)#First improvement of random adaptive

capacity_dict = {(n): c for n, c in zip(cn,cc)}#Refill capacity
first_construct=first_improvement_local_search(demand_dict, capacity_dict, distance_dict, assigned_demand_constructive)#First improvement of constructive

capacity_dict = {(n): c*1.1 for n, c in zip(cn,cc)}#Refill capacity
first_construct_additional_capacity=first_improvement_local_search(demand_dict, capacity_dict, distance_dict, assigned_demand_constructive)

print()
print("The result of random adaptive algorithm is:\n", og_random,
      "\nThe first improvement (10% additional capacity) of random adaptive algorithm value is:\n",first_random[-1],
      "\nThe improvement value of random adaptive doing First Improvement is:\n", og_random-first_random[-1],
      "\nThe result of Heuristic algorithm is:\n",og_construct,
      "\nThe improvement value of Heuristic doing First Improvement(No additional capacity) is:\n",og_construct-first_construct[-1],
      "\nThe improvement value of Heuristic doing First Improvement(10% additional capacity) is:\n",og_construct-first_construct_additional_capacity[-1]) 

"""The following code is for visulization, red line is the route assigned by optimization model(pulp),
the blue line is the route assigned by Heuristic, and the yellow line is the route assigned by First improvement"""

test=first_random[0] #extract the new assigned routes to a new list named "test"
ff={}#Create an empty dictionary to store the dataframe of each route
for i in fdata.index:
    ff[i]=localauthorities[localauthorities.LAD22CD.isin([test[j][0] for j in range(374) if test[j][1]==i])][['LONG','LAT','LAD22NM']]
for i in f:
    ff[i]["LONG_t"]=fdata.loc[i,"LONG"]
    ff[i]["LAT_t"]=fdata.loc[i,"LAT"]
    ff[i]["D"]=i
ff1, ff2, ff3, ff4, ff5, ff6, ff7, ff8, ff9, ff10, ff11, ff12, ff13 = [None]*13
for i, key in enumerate(ff):
    nff = ff[key]
    new_df = nff.copy()
    if i == 0:
        ff1 = new_df
    elif i == 1:
        ff2 = new_df
    elif i == 2:
        ff3 = new_df
    elif i == 3:
        ff4 = new_df
    elif i == 4:
        ff5 = new_df
    elif i == 5:
        ff6 = new_df
    elif i == 6:
        ff7 = new_df
    elif i == 7:
        ff8 = new_df
    elif i == 8:
        ff9 = new_df
    elif i == 9:
        ff10 = new_df
    elif i == 10:
        ff11 = new_df
    elif i == 11:
        ff12 = new_df
    elif i == 12:
        ff13 = new_df
p_f=pd.concat([ff1, ff2, ff3, ff4, ff5, ff6, ff7, ff8, ff9, ff10, ff11, ff12, ff13])


source_to_dest=zip(p.LAT,p.LAT_t,p.LONG,p.LONG_t)
source_to_dest_h=zip(p_h.LAT,p_h.LAT_t,p_h.LONG,p_h.LONG_t)
source_to_dest_f=zip(p_f.LAT,p_f.LAT_t,p_f.LONG,p_f.LONG_t)

cities=list(dict.fromkeys(fdata["LAD22NM"]))
scatter_hover_data = [city for city in zip(cities)]
fig=go.Figure()
for slat,dlat, slon, dlon in source_to_dest:
    fig.add_trace(go.Scattergeo(
                        lat = [slat,dlat],
                        lon = [slon, dlon],
                        mode = 'lines',
                        line = dict(width = 0.5, color="red")
                        ))
for slat,dlat, slon, dlon in source_to_dest_h:
    fig.add_trace(go.Scattergeo(
                        lat = [slat,dlat],
                        lon = [slon, dlon],
                        mode = 'lines',
                        line = dict(width = 0.5, color="blue")
                        ))
for slat,dlat, slon, dlon in source_to_dest_f:
    fig.add_trace(go.Scattergeo(
                        lat = [slat,dlat],
                        lon = [slon, dlon],
                        mode = 'lines',
                        line = dict(width = 0.5, color="yellow")
                        ))

fig.add_trace(
    go.Scattergeo(
                lon = fdata["LONG"].values.tolist(),
                lat = fdata["LAT"].values.tolist(),
                hoverinfo = 'text',
                text = scatter_hover_data,
                mode = 'markers',
                marker = dict(size = 10, color = 'blue', opacity=0.1))
    )

fig.update_layout(title_text = 'The allocation of LAs to facilities',
                  height=700, width=900,
                  margin={"t":0,"b":0,"l":0, "r":0, "pad":0},
                  showlegend=False,
                  geo = dict(projection_type = 'natural earth',scope = 'europe'))
fig.show()



















