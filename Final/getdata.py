import csv

maxnum = 500

with open('SBIC.v2.agg.tst.csv', newline='') as csvfile:
    
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    
    header = next(spamreader)

    postin = header.index('post')
    stereoin = header.index('targetStereotype')
    i = 0

    with open('data.csv', 'a', newline='') as newfile:
        spamwriter = csv.writer(newfile, delimiter=',', quotechar='"')

        for row in spamreader:
            if row[stereoin] != '[]':
                i += 1
                print(row[postin].strip('\n'))
                spamwriter.writerow([row[postin].strip('\n')] + [row[stereoin]])

                if i == maxnum:
                    exit()