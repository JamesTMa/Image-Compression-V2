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

def getImageScore(original, new, mode, res):
    if mode == 'linear':
        return np.sum(abs(original[::res, ::res] - new[::res, ::res]))

def sigmoid(x):
    #if x > 10:
        #return 1
    #if x < -10:
        #return 0
    return 1 / (1 + math.exp(-x))

def blotchFunction(x, y, cx, cy, color, scale, sharpness):
    distance = math.pow(math.pow(x - cx, 2) + math.pow(y - cy, 2), 0.5)
    #distance = max(abs(cx - x), abs(cy - y))
    return color * sigmoid((scale - distance) * sharpness)

def generateBlotchMat(img, cx, cy, color, scale, sharpness, res):
    #print(cx, cy, color, scale, sharpness)
    mat = np.zeros_like(img, dtype= np.float32)
    rows, cols, channels = mat.shape
    x = int(max(cx - scale, 0))
    y = int(max(cy - scale, 0))
    while y < int(min(cy + scale, rows)):
        while x < int(min(cx + scale, cols)):
            distance = math.pow(math.pow(x - cx, 2) + math.pow(y - cy, 2), 0.5)
            if distance < scale:
                mat[y, x] = blotchFunction(x, y, cx, cy, color, scale, sharpness)
                #print(x, y)
            x += res
        y += res
        x = int(max(cx - scale, 0))
    #for y in range(int(max(cy - scale, 0)), int(min(cy + scale, rows))):
        #for x in range(int(max(cx - scale, 0)), int(min(cx + scale, cols))):
            #distance = math.pow(math.pow(x - cx, 2) + math.pow(y - cy, 2), 0.5)
            #if distance < scale:
                #mat[y, x] = blotchFunction(x, y, cx, cy, color, scale, sharpness)
    return mat

def calculateColor(img, x, y, scale):
    rows, cols, channels = img.shape
    #print(img[y, x])
    #print(x, y, scale)
    sector = img[int(max(0, y - scale)):int(min(y + scale, rows - 1)), int(max(0, x - scale)):int(min(x + scale, cols - 1))]
    #print(sector.shape)
    srows, scols, schannels = sector.shape
    if srows * scols == 0:
        return [0, 0, 0]
    return np.average(sector, (0, 1))

def generateCandidate(original, img):
    rows, cols, channels = img.shape
    x = random.randrange(cols)
    y = random.randrange(rows)
    scale = random.randrange(1, max(rows, cols))
    sharpness = random.random()
    color = calculateColor(original - img, x, y, scale)
    return np.asarray([x, y, color[0], color[1], color[2], scale, sharpness]).astype(np.float32)

def generateNCandidates(original, img, n):
    candidates = np.zeros((n, 7), dtype= np.float32)
    for i in range(n):
        candidates[i] = generateCandidate(original, img)
    return candidates

def testCandidateWithRes(original, img, diff, candidate, res):
    x = candidate[0]
    y = candidate[1]
    color = candidate[2:5]
    scale = candidate[5]
    sharpness = candidate[6]
    rows, cols, channels = original.shape
    croppedOriginal = original[int(max(y - scale, 0)):int(min(y + scale, rows)), int(max(x - scale, 0)):int(min(x + scale, cols))]
    croppedImg = img[int(max(y - scale, 0)):int(min(y + scale, rows)), int(max(x - scale, 0)):int(min(x + scale, cols))]
    diff[int(max(y - scale, 0)):int(min(y + scale, rows)), int(max(x - scale, 0)):int(min(x + scale, cols))] = diff[int(max(y - scale, 0)):int(min(y + scale, rows)), int(max(x - scale, 0)):int(min(x + scale, cols))] - generateBlotchMat(img, x, y, color, scale, sharpness, res)[int(max(y - scale, 0)):int(min(y + scale, rows)), int(max(x - scale, 0)):int(min(x + scale, cols))]
    return getImageScore(diff, np.zeros_like(diff), 'linear', res) * res * res

def findBestNCandidates(original, img, diff, candidates, n):
    resRatio = 5
    numOfCandidates, candidateDimensions = candidates.shape
    scoredCandidates = np.zeros((numOfCandidates, candidateDimensions + 1), dtype= np.float32)
    for i in range(numOfCandidates):
        scoredCandidates[i] = np.asarray([candidates[i][0], candidates[i][1], candidates[i][2], candidates[i][3], candidates[i][4], candidates[i][5], candidates[i][6], testCandidateWithRes(original, img, diff, candidates[i], int(candidates[i][5] / resRatio) + 1)]).astype(np.float32)
    bestNCandidates = np.zeros((n, candidateDimensions), dtype= np.float32)
    for i in range(n):
        bestIndex = np.argmin(scoredCandidates[:, candidateDimensions].flatten())
        bestNCandidates[i] = scoredCandidates[bestIndex][0:7]
        scoredCandidates[bestIndex][candidateDimensions] = 1000000000.0
    return bestNCandidates

def generateNChildren(img, candidate, n):
    posvariance = 10
    variance = 0.3
    rng = np.random.default_rng()
    rows, cols, channels = img.shape
    children = np.zeros((n, 7), dtype= np.float32)
    for i in range(n):
        x = min(max(candidate[0] + posvariance * rng.uniform(-1, 1), 0), cols)
        y = min(max(candidate[1] + posvariance * rng.uniform(-1, 1), 0), rows)
        scale = candidate[5] * rng.uniform(1 - variance, 1 + variance)
        sharpness = candidate[6] * rng.uniform(1 - variance, 1 + variance)
        color = calculateColor(img, x, y, scale)
        #print(color)
        children[i] = np.asarray([x, y, color[0], color[1], color[2], scale, sharpness]).astype(np.float32)
    return children

def evolveCandidates(original, img, repeats):
    n = 3
    diff = original - img
    candidates = generateNCandidates(original, img, n * n)
    bestCandidates = findBestNCandidates(original, img, diff, candidates, n)
    #candidates = candidates.fill(0)
    for _ in range(repeats):
        for i in range(len(bestCandidates)):
            #print(i)
            candidates[i * n:(i + 1) * n] = generateNChildren(diff, bestCandidates[i], n)
        bestCandidates = findBestNCandidates(original, img, diff, candidates, n)
    return findBestNCandidates(original, img, diff, bestCandidates, 1)[0]

original = getImage(0).astype(np.float32)[::1, ::1]
img = np.zeros_like(original, dtype= np.float32)
print(original.shape)
candidate = generateCandidate(original, img)
mat = generateBlotchMat(img, candidate[0], candidate[1], np.asarray([candidate[2], candidate[3], candidate[4]]).astype(np.float32), candidate[5], candidate[6], 1)
#showImage(mat.astype(np.uint8))
for i in range(100):
    print(i)
    bestCandidate = evolveCandidates(original, img, 3)
    #print(bestCandidate)
    img += generateBlotchMat(img, bestCandidate[0], bestCandidate[1], np.asarray([bestCandidate[2], bestCandidate[3], bestCandidate[4]]).astype(np.float32), bestCandidate[5], bestCandidate[6], 1)
    if i % 10 == 0:
        showImage(img / np.max(img), 1)
rows, cols, channels = img.shape
for y in range(rows):
    for x in range(cols):
        for c in range(channels):
            img[y, x, c] = min(max(img[y, x, c], 0), 255)
showImage(img.astype(np.uint8), 0)
#print(bestCandidate)

#showImage(mat)
#print(mat[300, 300])
