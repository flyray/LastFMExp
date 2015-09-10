
user2idx = {}
num = 1
fin = open('user_taggedbookmarks-timestamps.dat','r')
fin.readline()
for line in fin:
	user = line.split()[0]
	if not user in user2idx:
		user2idx[user] = str(num)
		num += 1
fin.close()

fin = open('user_taggedbookmarks-timestamps.dat','r')
fout = open('user_taggedbookmarks-timestamps.dat.mapped','w')
fout.write(fin.readline())
for line in fin:
	user = line.split()[0]
	fout.write(user2idx[user]+line[len(user):])
fout.close()
fin = open('user_contacts.dat','r')
fout = open('user_contacts.dat.mapped','w')
fout.write(fin.readline())
for line in fin:
	user1 = line.split()[0]
	user2 = line.split()[1]
	fout.write(user2idx[user1]+'\t'+user2idx[user2]+line[len(user1)+len(user2)+1:])