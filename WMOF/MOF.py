import numpy as np
from numba import jit, objmode
from scipy.spatial.distance import cdist

# Calculate number of points in radius
@jit(nopython=True)
def _point_in_radius(dm, sort_idx, win_size, n):
  idx = np.ones((win_size,n), dtype = np.int64)
  for i in range(win_size):
    pre_distance = -1
    point_count = n
    for j in range(n-1, -1, -1):
      obv_point = sort_idx[i][j]
      if dm[i][obv_point] != pre_distance:
        point_count = j
        pre_distance = dm[i][obv_point]
      idx[i][obv_point] = point_count
  return idx

# Calculate variance mass ratio
@jit(nopython=True)
def _Var_Massratio(Data,window):
  # Beware of large numbers, it might overflow python int
    n = len(Data)
    mass = np.zeros(n)
    mass2 = np.zeros(n)
    assert(window > 0)

    # slicing window through data
    for start_point in range(0,n,window):
      stop_point = min(start_point+ window, n)
      Current_Data = Data[start_point : stop_point]
      win_size = stop_point - start_point

      with objmode(window_dm = "i8[:, :]", sort_idx = "i8[:, :]"):
        window_dm = cdist(Current_Data, Data)
        sort_idx = np.argsort(window_dm)

      # when pairwise distance is same, the index is also same
      current_idx = _point_in_radius(window_dm, sort_idx, win_size, n)

      # calculate all current points
      for i in range(start_point, stop_point):
        for j in range(i+1, stop_point):
          m = (current_idx[j%window][i]*1.0 + 1)/ (current_idx[i%window][j] + 1)
          mass[i] += m
          mass2[i] += m**2
          mass[j] += 1/m
          mass2[j] += 1/m**2

      # calculate remaining points
      for i in range(stop_point,n):
        with objmode(dm = "i8[:, :]", sort_remain = "i8[:, :]"):
          dm = cdist([Data[i]], Data)
          sort_remain = np.argsort(dm)

        # when pairwise distance is same, the index is also same
        idx = _point_in_radius(dm, sort_remain, 1, n)

        for j in range(start_point, stop_point):
          m = (current_idx[j%window][i]*1.0 + 1 )/ (idx[0][j] + 1)
          mass[i] += m
          mass2[i] += m**2
          mass[j] += 1/m
          mass2[j] += 1/m**2

    var = mass2/(n-1)-(mass/(n-1))**2
    return var