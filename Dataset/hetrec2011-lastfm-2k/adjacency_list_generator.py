fin = open('user_friends.dat.mapped','r')
adj = {}
maxid = 0
edge = 0
fin.readline()
for line in fin:
	edge += 1
	user1 = int(line.split()[0])
	user2 = int(line.split()[1])
	maxid = max(maxid, user1)
	if not user1 in adj:
		adj[user1] = []
	adj[user1].append(user2)
fin.close()
fout = open('user_relation_adjacency_list.dat','w')
fout.write(str(maxid)+' '+str(edge/2)+'\n')
for i in range(1, maxid+1):
	for j in adj[i]:
		fout.write(str(j)+' ') 
	fout.write('\n')

