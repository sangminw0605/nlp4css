import csv
from collections import defaultdict 

def percent_agreement(filename1, filename2, total, params):
    percentages = {}
    headers = {}

    with open(filename1) as file1:
        reader1 = csv.reader(file1, delimiter=',')

        with open(filename2) as file2:
            reader2 = csv.reader(file2, delimiter=',')

            next(reader1)
            first = next(reader2)

            for i, entry in enumerate(first[len(first) - 4 :]):
                headers[i] = entry
                percentages[entry] = 0

            for i in range(total):
                temp1, temp2 = next(reader1), next(reader2)
                l1 = temp1[len(temp1) - params:]
                l2 = temp2[len(temp2) - params:]

                for j in range(params):
                    if l1[j] == l2[j]:
                        percentages[headers[j]] += 1


    for key in percentages:
        percentages[key] /= total

    return percentages



def count(filename1, filename2):
    counts1 = defaultdict()
    counts2 = defaultdict()

    with open(filename1) as file1:
        reader = csv.reader(file1, delimiter=',')

        first = next(reader)
        header = {}

        for i, entry in enumerate(first[len(first) - 4 :]):
            header[i] = entry
            counts1[entry] = defaultdict()

        for row in reader:
            for i, entry in enumerate(row[len(row) - 4 :]):
                if entry not in counts1[header[i]]:
                    counts1[header[i]][entry] = 0

                counts1[header[i]][entry] += 1

    with open(filename2) as file2:
        reader = csv.reader(file2, delimiter=',')

        first = next(reader)
        header = {}

        for i, entry in enumerate(first[len(first) - 4 :]):
            header[i] = entry
            counts2[entry] = defaultdict()
    
        for row in reader:
            for i, entry in enumerate(row[len(row) - 4 :]):
                if entry not in counts2[header[i]]:
                    counts2[header[i]][entry] = 0

                counts2[header[i]][entry] += 1

    return counts1, counts2 


def fleiss_kappa(c1, c2, total, p_ag):
    percentages = {}

    for i in p_ag:
        percentages[i] = 0

    for key in c1:

        running = 0
        for entry in c1[key]:
            if entry in c2[key]:
                running += (c1[key][entry] * c2[key][entry])
        
        percentages[key] = running / (total * total)

    adjusted = {}

    for i in percentages:
        adjusted[i] = (p_ag[i] - percentages[i]) / (1 - percentages[i])
    
    return adjusted

        


if __name__ == "__main__":
    c1, c2 = count('Sangmin-2.csv', 'sabrina_2.csv')

    pa = percent_agreement('Sangmin-2.csv', 'sabrina_2.csv', 40, 4)

    print('Percent agreement: ', pa)
    print('Cohen\'s Kappa: ', fleiss_kappa(c1, c2, 40, pa))