import numpy as np



def parse():
    """
    Parse the txt file into wifi components
    """
    line = np.loadtxt('wifi_db/clean_dataset.txt', delimiter=',')
    print(line)
    
if __name__ == "__main__":
    parse()