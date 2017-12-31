import pandas as pd
import numpy as np
stay_points = np.array([[1,1,0.1,0.1],[2,2,0.1,0.1]])
row_name = [str(p) for p in stay_points]
col_name = ['a','b','c','d']
df = pd.DataFrame(stay_points, index = row_name, columns = col_name)

