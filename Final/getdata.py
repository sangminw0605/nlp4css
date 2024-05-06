import csv

with open('SBIC.v2.agg.tst.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    
    header = next(spamreader)

    postin = header.index('post')
    stereoin = header.index('targetStereotype')
    i = 1

    with open('eggs.csv', 'a', newline='') as newfile:
        spamwriter = csv.writer(newfile, delimiter=',', quotechar='"')

        for row in spamreader:
            if len(row) > stereoin and row[stereoin] != '[]':
                spamwriter.writerow([row[postin]] + [row[stereoin]])