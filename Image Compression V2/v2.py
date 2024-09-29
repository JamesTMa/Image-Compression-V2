import numpy as np
import cv2 as cv
import random
import math

def getImage(index):
    return cv.imread('Images/img_' + str(index) + '.jpeg')

def showImage(img, waitKey):
    cv.imshow('Image', img)
    cv.waitKey(waitKey)
    cv.destroyAllWindows()

def getImageScore(original, old, new, mode):
    if mode == 'linear':
        return np.sum(abs(original - new)) - np.sum(abs(original - old))
    if mode == "quadratic":
        return np.sum(np.pow(original - new, 2)) - np.sum(np.pow(original - old, 2))

def getScoreWithBlotch(original, img, mode, cx, cy, height, width, color):
    samples = 5
    rows, cols, channels = img.shape
    resConstx = max(width / samples, 1)
    resConsty = max(height / samples, 1)
    #old = np.copy(img[max(cy - height, 0):min(cy + height, rows):resConsty, max(cx - width, 0):min(cx + width, cols):resConstx])
    #new = np.copy(img[max(cy - height, 0):min(cy + height, rows):resConsty, max(cx - width, 0):min(cx + width, cols):resConstx])
    coriginal = np.zeros((int(min(2 * samples, 2 * height)), int(min(2 * samples, 2 * width)), 3), dtype= np.float32)
    old = np.zeros((int(min(2 * samples, 2 * height)), int(min(2 * samples, 2 * width)), 3), dtype= np.float32)
    new = np.zeros((int(min(2 * samples, 2 * height)), int(min(2 * samples, 2 * width)), 3), dtype= np.float32)
    for y in range(int(min(2 * samples, 2 * height))):
        for x in range(int(min(2 * samples, 2 * width))):
            xx = int(resConstx * (x - min(samples, width)) + cx)
            yy = int(resConsty * (y - min(samples, height)) + cy)
            if xx >= 0 and xx < cols and yy >= 0 and yy < rows:
                coriginal[y, x] = original[yy, xx]
                old[y, x] = img[yy, xx]
                new[y, x] = img[yy, xx]
                distance = math.pow((xx - cx) / width, 2) + math.pow((yy - cy) / height, 2)
                if distance < 1:
                    new[y, x] += color * sigmoid((1 - distance) * 5)
    crows, ccols, cchannels = old.shape
    return getImageScore(coriginal, old, new, mode) * (width * height) / (crows * ccols)

'''if height < samples:
        for y in range(max(cy - height, 0), min(cy + height, rows)):
            if width < samples:
                for x in range(max(cx - width, 0), min(cx + width, cols)):
                    if math.pow(x / width, 2) + math.pow(y / height, 2) < 1:
                        new[int(y), int(x)] += color
            else:
                for x in range(max(-samples, int(-cx / resConstx) + 1), min(samples, int((cols - cx) / resConstx) + 1)):
                    xx = x * resConstx + cx
                    if math.pow(xx / width, 2) + math.pow(y / height, 2) < 1:
                        new[int(y), int(xx)] += color
    else:
        for y in range(max(-samples, int(-cy / resConsty) + 1), min(samples, int((rows - cy) / resConsty) + 1)):
            if width < samples:
                for x in range(max(cx - width, 0), min(cx + width, cols)):
                    yy = y * resConsty + cy
                    if math.pow(x / width, 2) + math.pow(yy / height, 2) < 1:
                        new[int(yy), int(x)] += color
            else:
                for x in range(max(-samples, int(-cx / resConstx) + 1), min(samples, int((cols - cx) / resConstx) + 1)):
                    xx = x * resConstx + cx
                    yy = y * resConsty + cy
                    if math.pow(xx / width, 2) + math.pow(yy / height, 2) < 1:
                        new[int(yy), int(xx)] += color
    return getImageScore(old)'''

def calculateColor(img, x, y, height, width):
    rows, cols, channels = img.shape
    #print(img[y, x])
    #print(x, y, scale)
    sector = img[int(max(0, y - height)):int(min(y + height, rows - 1)), int(max(0, x - width)):int(min(x + width, cols - 1))]
    #print(sector.shape)
    srows, scols, schannels = sector.shape
    if srows * scols == 0:
        return [0, 0, 0]
    return np.average(sector, (0, 1))

def generateCandidate(original, img):
    rows, cols, channels = img.shape
    x = random.randrange(cols)
    y = random.randrange(rows)
    height = random.randrange(1, rows)
    width = height#random.randrange(1, cols)
    color = calculateColor(original - img, x, y, height, width)
    return np.asarray([x, y, color[0], color[1], color[2], height, width]).astype(np.float32)

def generateNCandidates(original, img, n):
    candidates = np.zeros((n, 7), dtype= np.float32)
    for i in range(n):
        candidates[i] = generateCandidate(original, img)
    return candidates

def testCandidate(original, img, candidate):
    x = candidate[0]
    y = candidate[1]
    color = candidate[2:5]
    height = candidate[5]
    width = candidate[6]
    rows, cols, channels = original.shape
    return getScoreWithBlotch(original, img, 'linear', x, y, height, width, color)

def findBestNCandidates(original, img, candidates, n):
    numOfCandidates, candidateDimensions = candidates.shape
    scoredCandidates = np.zeros((numOfCandidates, candidateDimensions + 1), dtype= np.float32)
    for i in range(numOfCandidates):
        scoredCandidates[i] = np.asarray([candidates[i][0], candidates[i][1], candidates[i][2], candidates[i][3], candidates[i][4], candidates[i][5], candidates[i][6], testCandidate(original, img, candidates[i])]).astype(np.float32)
    bestNCandidates = np.zeros((n, candidateDimensions), dtype= np.float32)
    for i in range(n):
        bestIndex = np.argmin(scoredCandidates[:, candidateDimensions].flatten())
        bestNCandidates[i] = scoredCandidates[bestIndex][0:7]
        scoredCandidates[bestIndex][candidateDimensions] = 1000000000.0
    return bestNCandidates

def generateNChildren(diff, candidate, n):
    posvariance = 10
    variance = 0.3
    rng = np.random.default_rng()
    rows, cols, channels = diff.shape
    children = np.zeros((n, 7), dtype= np.float32)
    for i in range(n):
        x = min(max(candidate[0] + posvariance * rng.uniform(-1, 1), 0), cols)
        y = min(max(candidate[1] + posvariance * rng.uniform(-1, 1), 0), rows)
        height = candidate[5] * rng.uniform(1 - variance, 1 + variance)
        width = height#candidate[6] * rng.uniform(1 - variance, 1 + variance)
        color = calculateColor(diff, x, y, height, width)
        #print(color)
        children[i] = np.asarray([x, y, color[0], color[1], color[2], height, width]).astype(np.float32)
    return children

def evolveCandidates(original, img, repeats):
    n = 3
    diff = original - img
    candidates = generateNCandidates(original, img, n * n)
    bestCandidates = findBestNCandidates(original, img, candidates, n)
    #candidates = candidates.fill(0)
    for _ in range(repeats):
        for i in range(len(bestCandidates)):
            #print(i)
            candidates[i * n:(i + 1) * n] = generateNChildren(diff, bestCandidates[i], n)
        bestCandidates = findBestNCandidates(original, img, candidates, n)
    return findBestNCandidates(original, img, bestCandidates, 1)[0]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generateBlotchMat(img, cx, cy, color, height, width, sharpness):
    #print(cx, cy, color, scale, sharpness)
    mat = np.zeros_like(img, dtype= np.float32)
    rows, cols, channels = mat.shape
    for y in range(rows):
        for x in range(cols):
            distance = math.pow(math.pow((x - cx) / width, 2) + math.pow((y - cy) / height, 2), 0.5)
            mat[y, x] = color * sigmoid((1 - distance) * 5)
                #print(x, y)
    #for y in range(int(max(cy - scale, 0)), int(min(cy + scale, rows))):
        #for x in range(int(max(cx - scale, 0)), int(min(cx + scale, cols))):
            #distance = math.pow(math.pow(x - cx, 2) + math.pow(y - cy, 2), 0.5)
            #if distance < scale:
                #mat[y, x] = blotchFunction(x, y, cx, cy, color, scale, sharpness)
    return mat

original = getImage(0).astype(np.float32)[::10, ::10]
img = np.zeros_like(original, dtype= np.float32)
print(original.shape)
candidate = generateCandidate(original, img)
mat = generateBlotchMat(img, candidate[0], candidate[1], np.asarray([candidate[2], candidate[3], candidate[4]]).astype(np.float32), candidate[5], candidate[6], 10)
showImage(mat.astype(np.uint8), 0)
rows, cols, channels = img.shape
for i in range(1000):
    print(i)
    bestCandidate = evolveCandidates(original, img, 3)
    #print(bestCandidate)
    cx = bestCandidate[0]
    cy = bestCandidate[1]
    color = np.asarray([bestCandidate[2], bestCandidate[3], bestCandidate[4]]).astype(np.float32)
    height = bestCandidate[5]
    width = bestCandidate[6]
    for y in range(rows):
        for x in range(cols):
            xx = (cx - x) / width
            yy = (cy - y) / height
            if math.pow(xx, 2) + math.pow(yy, 2) < 1:
                img[y, x] += generateBlotchMat(img, cx, cy, np.asarray([bestCandidate[2], bestCandidate[3], bestCandidate[4]]).astype(np.float32), height, width, 10.1 - math.sqrt(i / 10))[y, x]
    print(bestCandidate)
    if i % 1 == 0:
        showImage(img / np.max(img), 1)
for y in range(rows):
    for x in range(cols):
        for c in range(channels):
            img[y, x, c] = min(max(img[y, x, c], 0), 255)
showImage(img.astype(np.uint8), 0)