import numpy as np

def parse(dataset_file_name):
    """
    Parse the txt file into wifi components
    """
    line = np.loadtxt(f"/Users/tianqi/Desktop/intro2ML/DecisionTrees/wifi_db/{dataset_file_name}.txt", delimiter='\t')
    return line
    
def presort(dataset):
    """
    Presort the data
    """
    sorted_data = [[sorted(dataset, key=lambda row: row[index])] for index in range(len(dataset[0])-1)]
    #print(sorted_data)
    return sorted_data

def start():
    data = parse("clean_dataset")
    #data = presort(data)
    print(type(data))
    return(data)