from graph import nodes ,elevation ,edges
node_list =[]

for node in nodes:
    node_list.append(list(nodes[node]))

if len(elevation) == len(node_list):
    for i in range(len(node_list)):
        node_list[i].append(elevation[i])
# print(node_list)
egde_dict ={}      
for i in edges:
    egde_dict[(i[0],i[1])] = [i[2],i[3]]
# print(egde_dict)