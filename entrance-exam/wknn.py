# k_nn.py
# k nearest neighbors demo
# Anaconda3 5.2.0 (Python 3.6.5)

import numpy as np

def dist_func(item, data_point):
  sum = 0.0
  for i in range(2):
    diff = item[i] - data_point[i+1]
    sum += diff * diff
  return np.sqrt(sum) 

def make_weights(k, distances):
  result = np.zeros(k, dtype=np.float32)
  sum = 0.0
  for i in range(k):
    result[i] += 1.0 / distances[i]
    sum += result[i]
  result /= sum
  return result

def show(v):
  print("idx = %3d (%3.2f %3.2f) class = %2d  " \
    % (v[0], v[1], v[2], v[3]), end="")

def main():
  print("\nBegin weighted k-NN demo \n")
  print("Normalized income-education data looks like: ")
  print("[id =  0, 0.32, 0.43, class = 0]")
  print(" . . . ")
  print("[id = 29, 0.71, 0.22, class = 2]")

  data = np.array([
    [0, 0.32, 0.43, 0], [1, 0.26, 0.54, 0],
    [2, 0.27, 0.60, 0], [3, 0.37, 0.36, 0],
    [4, 0.37, 0.68, 0], [5, 0.49, 0.32, 0],
    [6, 0.46, 0.70, 0], [7, 0.55, 0.32, 0],
    [8, 0.57, 0.71, 0], [9, 0.61, 0.42, 0],
    [10, 0.63, 0.51, 0], [11, 0.62, 0.63, 0],
    [12, 0.39, 0.43, 1], [13, 0.35, 0.51, 1],
    [14, 0.39, 0.63, 1], [15, 0.47, 0.40, 1],
    [16, 0.48, 0.50, 1], [17, 0.45, 0.61, 1],
    [18, 0.55, 0.41, 1], [19, 0.57, 0.53, 1],
    [20, 0.56, 0.62, 1], [21, 0.28, 0.12, 1],
    [22, 0.31, 0.24, 1], [23, 0.22, 0.30, 1],
    [24, 0.38, 0.14, 1], [25, 0.58, 0.13, 2],
    [26, 0.57, 0.19, 2], [27, 0.66, 0.14, 2],
    [28, 0.64, 0.24, 2], [29, 0.71, 0.22, 2]],
    dtype=np.float32)

  item = np.array([0.62, 0.35], dtype=np.float32)
  print("\nNearest neighbors (k=6) to (0.62, 0.35): ")

  # 1. compute all distances to item
  N = len(data)
  k = 6
  c = 3
  distances = np.zeros(N)
  for i in range(N):
    distances[i] = dist_func(item, data[i])

  # 2. get ordering of distances
  ordering = distances.argsort()

  # 3. get and show info for k nearest
  k_near_dists = np.zeros(k, dtype=np.float32)
  for i in range(k):
    idx = ordering[i]
    show(data[idx])  # pretty formatting
    print("distance = %0.4f" % distances[idx])
    k_near_dists[i] = distances[idx]  # save dists

  # 4. vote
  votes = np.zeros(c, dtype=np.float32)
  wts = make_weights(k, k_near_dists)
  print("\nWeights (inverse distance technique): ")
  for i in range(len(wts)):
    print("%7.4f" % wts[i], end="")
  
  print("\n\nPredicted class: ")
  for i in range(k):
    idx = ordering[i]
    pred_class = np.int(data[idx][3])
    votes[pred_class] += wts[i] * 1.0
  for i in range(c):
    print("[%d]  %0.4f" % (i, votes[i]))

  print("\nEnd weighted k-NN demo ")

if __name__ == "__main__":
  main()