import argparse
import math
import os
import random
import sys

import numpy as np
import scipy.optimize
import scipy.signal
import scipy.sparse
import scipy.sparse.csgraph
import cv2 as cv

print("^^^^^^^^^^^ it's safe to ignore any libdc1394 error.")

class ICMIS(object):
  def __init__(self, files, cols):
    assert len(files) % cols == 0, "Number of files ({:d}) not evenly divible by columns ({:d})".format(
      len(files), cols)
    files.sort()
    self.files = np.array(files).reshape((-1, cols))
    self.rows, self.cols = self.files.shape
    self.img_shape = np.array(cv.imread(self.files[0, 0]).shape)[0:2]

  def correlation(self, img1, img2):
    """Computes the correlation matrix between two images. The images must
    be of the same size. Note that because ndarrays have their first axis in the
    y direction, coordinates are always (y, x).

    Args:
        img1 (np.ndarray): The image data for the first image. This is
                           the result of cv.imread.
        img2 (np.ndarray): The image data for the second image. This is
                           the result of cv.imread.

    Returns:
        np.ndarray: The PCM, a real-valued 2D array.
    """
    assert img1.shape == img2.shape, "Images are not the same size: {:s} and {:s}".format(
      str(img1.shape), str(img2.shape))

    # Compute the Fourier Transforms of each image.
    fimg = np.fft.fft2(img1)
    fimg2 =np.fft.fft2(img2)

    # Element-wise multiply the first by the complex conjugate of the second.
    fc = fimg * fimg2.conj()

    # Element-wise divide the result by its own magnitude.
    # This sucks if one element is exactly zero. Hopefully that doesn't
    # happen. Maybe if it does, replace with 1+0j?
    fpcm = fc / abs(fc)

    # Compute the inverse Fourier Transform of the result.
    pcm = np.fft.ifft2(fpcm)

    # The result is real, with a small imaginary component due to numerical
    # imprecision. So we just take the real value.
    return pcm.real

  def match_template(self, source, template):
    result = cv.matchTemplate(source, template, method=cv.TM_CCORR_NORMED)
    return result

  def write_array_as_image(self, result, filename):
    min_result = result.min()
    max_result = result.max()
    print("min/max: {!r}/{!r}".format(min_result, max_result))
    result = (result - min_result) * 255.0 / (max_result - min_result)
    cv.imwrite(filename, result)

  def compute_maximum_spanning_tree(self, col_correlations, row_correlations):
    """Compute a maximum spanning tree over all the translations where the vertices
    are the images, and the weights of the connections between the images are the
    normalized cross-correlations between the images.

    Args:
      files (np.ndarray): The 2D array of filenames.
      col_correlations (np.ndarray): The 2D array of column correlations.
      row_correlations (np.ndarray): The 2D array of row correlations.

    Returns:
      scipy.sparse.csr_matrix: The len(files) x len(files) sparse matrix where a negative
                               number at graph(i, j) indicate that the image j is
                               to be positioned relative to the image i.
    """
    connections = scipy.sparse.lil_matrix((self.rows * self.cols, self.rows * self.cols))

    for y in range(self.rows):
      for x in range(self.cols - 1):
        connections[x + y * self.cols, x + 1 + y * self.cols] = -col_correlations[y, x]
        connections[x + 1 + y * self.cols, x + y * self.cols] = -col_correlations[y, x]
    for x in range(self.cols):
      for y in range(self.rows - 1):
        connections[x + y * self.cols, x + (y + 1) * self.cols] = -row_correlations[y, x]
        connections[x + (y + 1) * self.cols, x + y * self.cols] = -row_correlations[y, x]

    connections = scipy.sparse.csr_matrix(connections)
    print("Connection matrix has {!r} connections".format(connections.nnz))
    return scipy.sparse.csgraph.minimum_spanning_tree(connections)

  def compute_positions(self, spanning_tree, col_translations, row_translations):
    print("Grid is {!r} x {!r}".format(self.cols, self.rows))
    print("col_translations is {!r} x {!r}".format(col_translations.shape[1], col_translations.shape[0]))
    print("row_translations is {!r} x {!r}".format(row_translations.shape[1], row_translations.shape[0]))
    nodes, predecessors = scipy.sparse.csgraph.breadth_first_order(spanning_tree, 0, directed=False)
    nodes = nodes.tolist()
    predecessors = predecessors.tolist()

    positions = np.zeros((self.rows * self.cols, 2), dtype=int)
    for row in range(self.rows):
      for col in range(self.cols):
        node = col + row * self.cols
        self.compute_position(node, nodes, predecessors, positions, col_translations, row_translations)
        print("Position of {!r},{!r} is {!r},{!r}".format(col, row, positions[node][1], positions[node][0]))

    return positions

  def compute_position(self, node, nodes, predecessors, positions, col_translations, row_translations):
    stack = [node]
    while stack[-1] != 0 and (positions[stack[-1]] == 0).all():
      predecessor = predecessors[stack[-1]]
      stack.append(predecessor)

    print("To find position of node {!r} requires this sequence: {!r}".format(node, stack))

    while len(stack) > 1:
      predecessor = stack[-1] # position is known
      node = stack[-2] # position needs to be calculated
      node_row = node // self.cols
      node_col = node % self.cols
      pred_row = predecessor // self.cols
      pred_col = predecessor % self.cols

      if node_row != pred_row: # Use row_translations
        assert node_col == pred_col
        if node_row < pred_row:
          translation = row_translations[pred_row - 1, pred_col]
          positions[node] = positions[predecessor] - translation
        else:
          translation = row_translations[node_row - 1, node_col]
          positions[node] = positions[predecessor] + translation
      else:
        if node_col < pred_col:
          translation = col_translations[pred_row, pred_col - 1]
          positions[node] = positions[predecessor] - translation
        else:
          translation = col_translations[node_row, node_col - 1]
          positions[node] = positions[predecessor] + translation
      stack = stack[:-1]

    return positions[stack[0]]


def main():
  version = "0.1"
  print("ICMIS v" + version)

  parser = argparse.ArgumentParser(description="ICMIS, the Integrated Circuit Microscope Image Stitcher.")
  parser.add_argument("cols", metavar="<cols>", type=int, help="Number of columns of images")
  parser.add_argument("--left", metavar="<filename>", nargs=1, help="Left image filename")
  args = parser.parse_args()

  files = [f for f in os.listdir(".") if f.startswith("IMG_") and f.endswith(".JPG")]
  icmis = ICMIS(files, args.cols)

  # files.sort()
  # index = files.index(args.left[0])
  # left_img = cv.imread(files[index], flags=cv.IMREAD_GRAYSCALE)
  # right_img = cv.imread(files[index + 1], flags=cv.IMREAD_GRAYSCALE)
  # template = right_img[:, 0:10]
  # result = icmis.match_template(left_img, template)
  # result = np.log(result)

  fast_col_translations = np.zeros((icmis.rows, icmis.cols - 1, 2))
  max_minor_move = 5

  for y in range(icmis.rows):
    img1 = None
    img2 = cv.imread(icmis.files[y, 0], flags=cv.IMREAD_GRAYSCALE)
    for x in range(icmis.cols - 1):
      img1 = img2
      img2 = cv.imread(icmis.files[y, x + 1], flags=cv.IMREAD_GRAYSCALE)
      # Cropping the minor coordinate effectively shifts the offset up by that much.
      result = icmis.match_template(img1, img2[max_minor_move:-max_minor_move, 0:10])

      peak_pos = cv.minMaxLoc(result)[3]
      # Shift the minor coordinate offset back down
      peak_coords = np.array((peak_pos[1], peak_pos[0])) - (max_minor_move, 0)
      fast_col_translations[y, x] = peak_coords
      print("{!r}-{!r}: {!r}".format(icmis.files[y, x], icmis.files[y, x + 1], peak_coords))

  fast_row_translations = np.zeros((icmis.rows - 1, icmis.cols, 2))

  for x in range(icmis.cols):
    img1 = None
    img2 = cv.imread(icmis.files[0, x], flags=cv.IMREAD_GRAYSCALE)
    for y in range(icmis.rows - 1):
      img1 = img2
      img2 = cv.imread(icmis.files[y + 1, x], flags=cv.IMREAD_GRAYSCALE)
      result = icmis.match_template(img1, img2[0:10, max_minor_move:-max_minor_move])
      peak_pos = cv.minMaxLoc(result)[3]
      peak_coords = np.array((peak_pos[1], peak_pos[0])) - (0, max_minor_move)
      fast_row_translations[y, x] = peak_coords
      print("{!r}-{!r}: {!r}".format(icmis.files[y, x], icmis.files[y + 1, x], peak_coords))

  print("==================================")

  col_median = np.median(fast_col_translations, axis=(0,1)).astype(int)
  row_median = np.median(fast_row_translations, axis=(0,1)).astype(int)
  print("col median: {!r}".format(col_median))
  print("row median: {!r}".format(row_median))

  # Recompute the best correlations, but only within a +/- 2 pixel neighborhood
  # around the median.

  valid_row_translations = ((fast_row_translations[:, :, 0] >= row_median[0] - 2) &
    (fast_row_translations[:, :, 0] <= row_median[0] + 2) &
    (fast_row_translations[:, :, 1] >= row_median[1] - 2) &
    (fast_row_translations[:, :, 1] <= row_median[1] + 2))
  valid_col_translations = ((fast_col_translations[:, :, 0] >= col_median[0] - 2) &
    (fast_col_translations[:, :, 0] <= col_median[0] + 2) &
    (fast_col_translations[:, :, 1] >= col_median[1] - 2) &
    (fast_col_translations[:, :, 1] <= col_median[1] + 2))
  print("{!r}/{!r} valid col translations, {!r}/{!r} valid row translations".format(
    np.count_nonzero(valid_col_translations), fast_col_translations.shape[0] * fast_col_translations.shape[1],
    np.count_nonzero(valid_row_translations), fast_row_translations.shape[0] * fast_row_translations.shape[1]))

  max_neighborhood = 2

  col_median_min = col_median - (max_neighborhood, max_neighborhood)
  col_median_max = col_median + (max_neighborhood, max_neighborhood)

  col_median_min += (max_minor_move, 0)
  col_median_max += (max_minor_move, 0)
  print("col_median search bounds: {!r}, {!r}".format(col_median_min, col_median_max))

  assert col_median_min[0] >= 0, "col_median is too low"
  assert col_median_max[0] <= max_minor_move * 2, "col_median is too high"

  re_col_translations = np.zeros((icmis.rows, icmis.cols - 1, 2))
  col_correlations = np.zeros((icmis.rows, icmis.cols - 1))

  for y in range(icmis.rows):
    img1 = None
    img2 = cv.imread(icmis.files[y, 0], flags=cv.IMREAD_GRAYSCALE)
    for x in range(icmis.cols - 1):
      img1 = img2
      img2 = cv.imread(icmis.files[y, x + 1], flags=cv.IMREAD_GRAYSCALE)
      result = icmis.match_template(img1, img2[max_minor_move:-max_minor_move, 0:10])
      # result is (11, width - 9)
      # For a y point of 0, that is 5. For a y point of y, that is y + 5.
      # For an x point of x, that is x.
      # Therefore, if the y neighborhood we want to look at is y_median - 2 to y_median + 2,
      # then the y neighborhood is y_median - 2 + 5 to y_median + 2 + 5.
      # I can't be bothered to refactor out the result shape calculation.
      mask = np.zeros(result.shape, dtype=np.uint8)
      mask[col_median_min[0]:col_median_max[0] + 1, col_median_min[1]:col_median_max[1] + 1] = 1

      _, correlation, _, peak_pos = cv.minMaxLoc(result, mask)
      peak_coords = np.array((peak_pos[1], peak_pos[0])) - (max_minor_move, 0)
      re_col_translations[y, x] = peak_coords
      col_correlations[y, x] = correlation
      print("{!r}-{!r}: {!r} ({!r})".format(icmis.files[y, x], icmis.files[y, x + 1], peak_coords,
        correlation))

  row_median_min = row_median - (max_neighborhood, max_neighborhood)
  row_median_max = row_median + (max_neighborhood, max_neighborhood)

  row_median_min += (0, max_minor_move)
  row_median_max += (0, max_minor_move)
  print("row_median search bounds: {!r}, {!r}".format(row_median_min, row_median_max))

  assert col_median_min[0] >= 0, "col_median is too low"
  assert col_median_max[0] <= max_minor_move * 2, "col_median is too high"

  re_row_translations = np.zeros((icmis.rows - 1, icmis.cols, 2))
  row_correlations = np.zeros((icmis.rows - 1, icmis.cols))

  for x in range(icmis.cols):
    img1 = None
    img2 = cv.imread(icmis.files[0, x], flags=cv.IMREAD_GRAYSCALE)
    for y in range(icmis.rows - 1):
      img1 = img2
      img2 = cv.imread(icmis.files[y + 1, x], flags=cv.IMREAD_GRAYSCALE)
      result = icmis.match_template(img1, img2[0:10, max_minor_move:-max_minor_move])

      mask = np.zeros(result.shape, dtype=np.uint8)
      mask[row_median_min[0]:row_median_max[0] + 1, row_median_min[1]:row_median_max[1] + 1] = 1

      _, correlation, _, peak_pos = cv.minMaxLoc(result, mask)
      peak_coords = np.array((peak_pos[1], peak_pos[0])) - (0, max_minor_move)
      re_row_translations[y, x] = peak_coords
      row_correlations[y, x] = correlation
      print("{!r}-{!r}: {!r} ({!r})".format(icmis.files[y, x], icmis.files[y + 1, x], peak_coords,
        correlation))

  # col_correlations = np.where(valid_col_translations, 1.0, col_correlations)
  # row_correlations = np.where(valid_row_translations, 1.0, row_correlations)


  spanning_tree = icmis.compute_maximum_spanning_tree(col_correlations, row_correlations)
  positions = icmis.compute_positions(spanning_tree, re_col_translations, re_row_translations)

  min_position = np.amin(positions, axis=0)
  print("Min position {!r}".format(min_position))
  positions = positions - min_position
  max_position = np.amax(positions, axis=0)
  print("Max position {!r}".format(max_position))
  h = max_position[0] - min_position[0] + icmis.img_shape[0]
  w = max_position[1] - min_position[1] + icmis.img_shape[1]

  print("Final image will be {:d}x{:d}".format(w, h))
  stitch = np.zeros((h, w, 3), np.uint8)
  print("Created memory array")

  for row in range(icmis.rows):
    for col in range(icmis.cols):
      node = col + row * icmis.cols
      pos = positions[node] - min_position
      y = pos[0]
      x = pos[1]
      print("Copying image {!r} ({!r}) to {!r},{!r}".format(node, icmis.files[row, col], y, x))
      img = cv.imread(icmis.files[row, col])
      stitch[y:y + img.shape[0], x:x + img.shape[1]] = img
    print("Wrote row {:d} of {:d}".format(row, icmis.rows - 1))

  cv.imwrite("stitch.jpg", stitch)

  sys.exit(0)


  # col_translations = np.zeros((icmis.rows, icmis.cols - 1, 2))

  # for y in range(icmis.rows):
  #   left_img = None
  #   right_img = cv.imread(icmis.files[y, 0], flags=cv.IMREAD_GRAYSCALE)
  #   for x in range(icmis.cols - 1):
  #     left_img = right_img
  #     right_img = cv.imread(icmis.files[y, x + 1], flags=cv.IMREAD_GRAYSCALE)

  #     result = icmis.correlation(left_img, right_img)
  #     result = np.roll(result, 10, axis=0)
  #     result = result[:20, :]

  #     # Find the flattened indices of the n largest values (in random order).
  #     # This is O(N) where N is the number of pixels.
  #     peak_pos = np.argpartition(result, -1, axis=None)[-1]
  #     # Convert to (x, y) coordinates.
  #     peak_coords = np.array(np.unravel_index(peak_pos, result.shape))
  #     peak_coords -= (10, 0)
  #     col_translations[y, x] = peak_coords
  #     print("{!r}-{!r}: {!r}".format(icmis.files[y, x], icmis.files[y, x + 1], peak_coords))

  median_translation = np.median(col_translations, axis=(0, 1))
  print("{!r}".format(median_translation))

  cumsum = np.cumsum(col_translations[y], axis=0)
  max_cumsum = np.amax(cumsum, axis=0)
  min_cumsum = np.amin(cumsum, axis=0)
  print("min cumsum {!r}".format(min_cumsum))
  print("max cumsum {!r}".format(max_cumsum))

  img_shape = icmis.img_shape
  print("img_shape {!r}".format(img_shape))
  img_size = (max_cumsum - (min_cumsum[0], 0) + img_shape).astype(int)
  h = img_size[0]
  w = img_size[1]
  print("{!r}x{!r}".format(w, h))

  stitch = np.zeros((h, w, 3), np.uint8)

  y = 0
  xpos = 0
  for x in range(icmis.cols):
    if x == 0:
      ypos = int(-min_cumsum[0])
      xpos = 0
    else:
      ypos += int(col_translations[y, x - 1][0])
      xpos += int(col_translations[y, x - 1][1])
    print("Copying image {!r} to {!r},{!r}".format(icmis.files[y][x], xpos, ypos))
    img = cv.imread(icmis.files[y][x])
    stitch[ypos:ypos + img.shape[0], xpos:xpos + img.shape[1]] = img
  print("Wrote row {:d} of {:d}".format(y, icmis.rows - 1))

  cv.imwrite("stitch.jpg", stitch)

  # min_result = result.min()
  # max_result = result.max()
  # print("min/max: {!r}/{!r}".format(min_result, max_result))
  # result = (result - min_result) * 255.0 / (max_result - min_result)
  # cv.imwrite("corr.png", result)

if __name__ == "__main__":
  main()
