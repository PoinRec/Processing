import numpy as np

def compare_npz_files(file1, file2):
  data1 = np.load(file1)
  data2 = np.load(file2)

  keys1 = set(data1.files)
  keys2 = set(data2.files)

  if keys1 != keys2:
    print("Keys differ:")
    print("In first file only:", keys1 - keys2)
    print("In second file only:", keys2 - keys1)
    return False

  all_equal = True
  for key in keys1:
    arr1 = data1[key]
    arr2 = data2[key]
    if arr1.shape != arr2.shape:
      print(f"Shape mismatch for key '{key}': {arr1.shape} vs {arr2.shape}")
      all_equal = False
      continue
    if not np.array_equal(arr1, arr2):
      diff_indices = np.where(arr1 != arr2)
      print(f"Content mismatch for key '{key}', example diff indices: {diff_indices}")
      all_equal = False

  if all_equal:
    print("Both npz files are identical.")
  else:
    print("Differences found in npz files.")

  return all_equal

# Example usage:
file_path_1 = "/home/zhihao/new_split_list2.npz"
file_path_2 = "/home/zhihao/new_split_list.npz"
compare_npz_files(file_path_1, file_path_2)