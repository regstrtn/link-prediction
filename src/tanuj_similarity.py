import numpy as np
import networkx

fl_epinions = open("soc-sign-epinions.txt")
fl_slashdot = open("soc-sign-Slashdot.txt")

count = 0
ls_epinions_train = []
ls_slashdot_train = []
ls_epinions_test = []
ls_slashdot_test = []

for lines in fl_epinions:
  if count < 4:
    count = count + 1
    continue
  else:
    ls_epinions_train.append(map(int,lines[:-1].split("\t")))


count = 0
for lines in fl_slashdot:
  if count < 4:
    count = count + 1
    continue
  else:
    ls_slashdot_train.append(map(int,lines[:-1].split("\t")))


ls_epinions_test = ls_epinions_train[int(len(ls_epinions_train)*0.8):]
ls_epinions_train = ls_epinions_train[:int(len(ls_epinions_train)*0.8)]

ls_slashdot_test = ls_slashdot_train[int(len(ls_slashdot_train)*0.8):]
ls_slashdot_train = ls_slashdot_train[:int(len(ls_slashdot_train)*0.8)]


#nodes_epinions = 131828
#nodes_slashdot = 82144

# matrix_epinions = np.zeros(shape=(nodes_epinions,nodes_epinions))
# matrix_epinions_positive = np.zeros(shape=(nodes_epinions,nodes_epinions))
# matrix_epinions_negative = np.zeros(shape=(nodes_epinions,nodes_epinions))
# matrix_slashdot = np.zeros(shape=(nodes_slashdot,nodes_slashdot))
# matrix_slashdot_positive = np.zeros(shape=(nodes_slashdot,nodes_slashdot))
# matrix_slashdot_negative = np.zeros(shape=(nodes_slashdot,nodes_slashdot))


matrix_slashdot = {}
matrix_epinions = {}


for items in ls_epinions_train:
  if matrix_epinions.get(items[0]) == None:
    matrix_epinions[items[0]] = set([items[1]])
  else:
    matrix_epinions[items[0]].add(items[1])

  if matrix_epinions.get(items[1]) == None:
    matrix_epinions[items[1]] = set([items[0]])
  else:
    matrix_epinions[items[1]].add(items[0])

for items in ls_slashdot_train:
  if matrix_slashdot.get(items[0]) == None:
    matrix_slashdot[items[0]] = set([items[1]])
  else:
    matrix_slashdot[items[0]].add(items[1])

  if matrix_slashdot.get(items[1]) == None:
    matrix_slashdot[items[1]] = set([items[0]])
  else:
    matrix_slashdot[items[1]].add(items[0])

"""
for items in ls_epinions_train:
  if items[2] == 1:
    if matrix_epinions_positive.get(items[0]) != None:
      matrix_epinions_positive[items[0]].append(items[1])
    else:
      matrix_epinions_positive[items[0]] = [items[1]]
  
  elif items[2] == -1:
    if matrix_epinions_negative.get(items[0]) != None:
      matrix_epinions_negative[items[0]].append(items[1])
    else:
      matrix_epinions_negative[items[0]] = [items[1]]

  
for items in ls_slashdot_train:
  if items[2] == 1:
    if matrix_slashdot_positive.get(items[0]) != None:
      matrix_slashdot_positive[items[0]].append(items[1])
    else:
      matrix_slashdot_positive[items[0]] = [items[1]]
  
  elif items[2] == -1:
    if matrix_slashdot_negative.get(items[0]) != None:
      matrix_slashdot_negative[items[0]].append(items[1])
    else:
      matrix_slashdot_negative[items[0]] = [items[1]]"""

common_neighbour_score = []

for k in matrix_epinions.keys():
  for k2 in matrix_epinions.keys():
    if (k2 not in matrix_epinions[k]) and (k != k2):
      print str(k)+","+str(k2)
      common_neighbour_score.append((k,k2,len(list(matrix_epinions[k] & matrix_epinions[k2]))))

common_neighbour_score = sorted(common_neighbour_score, key=lambda x: x[2], reverse = True)
common_neighbour_score = common_neighbour_score[:len(ls_epinions_test)]
fl = open("predicted_common_epinions.txt","w")
for items in common_neighbour_score:
  fl.write(str(items[0])+" "+str(items[1])+"\n")

fl.close()

common_neighbour_score = []

for k in matrix_slashdot.keys():
  for k2 in matrix_slashdot.keys():
    if (k2 not in matrix_slashdot[k]) and (k != k2):
      print str(k)+","+str(k2)
      common_neighbour_score.append((k,k2,len(list(matrix_slashdot[k] & matrix_slashdot[k2]))))

common_neighbour_score = sorted(common_neighbour_score, key=lambda x: x[2], reverse = True)
common_neighbour_score = common_neighbour_score[:len(ls_slashdot_test)]
fl = open("predicted_common_slashdot.txt","w")
for items in common_neighbour_score:
  fl.write(str(items[0])+" "+str(items[1])+"\n")

fl.close()