import csv
import os

def makeMultiDataList(list_rootpaths, csv_name): #For Multi Directory
    data_list = []
    for list_rootpath in list_rootpaths:
        csv_path = os.path.join(list_rootpath, csv_name)
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                row[0] = os.path.join(list_rootpath, row[0])
                data_list.append(row)
    return data_list