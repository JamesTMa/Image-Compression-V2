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

def sigmoid(x):
    if x > 10:
        return 1.0
    if x < -10:
        return 0.0
    return 1 / (1 + np.exp(-x))

def getScoreWithBlotch(original, img, mode, cx, cy, height, width, tilt, sharpness):
    samples = 5
    rows, cols, channels = original.shape
    resConstx = width / samples
    resConsty = height / samples
    #old = np.copy(img[max(cy - height, 0):min(cy + height, rows):resConsty, max(cx - width, 0):min(cx + width, cols):resConstx])
    #new = np.copy(img[max(cy - height, 0):min(cy + height, rows):resConsty, max(cx - width, 0):min(cx + width, cols):resConstx])
    coriginal = np.zeros((2 * samples, 2 * samples, 3), dtype= np.float32)
    cimg = np.zeros((2 * samples, 2 * samples, 3), dtype= np.float32)
    old = np.zeros((2 * samples, 2 * samples, 3), dtype= np.float32)
    new = np.zeros((2 * samples, 2 * samples, 3), dtype= np.float32)
    sigtotal = 0
    for y in range(int(2 * samples)):
        for x in range(int(2 * samples)):
            xx = int(resConstx * (x - samples) + cx)
            yy = int(resConsty * (y - samples) + cy)
            if xx >= 0 and xx < cols and yy >= 0 and yy < rows:
                tiltedx = (xx - cx) * math.cos(-tilt) - (yy - cy) * math.sin(-tilt)
                tiltedy = (yy - cy) * math.cos(-tilt) + (xx - cx) * math.sin(-tilt)
                old[y, x] = img[yy, xx]
                distance = math.pow((tiltedx) / width, 2) + math.pow((tiltedx) / height, 2)
                if distance < 1:
                    sigtotal += sigmoid((1 - distance))
                    coriginal[y, x] += original[yy, xx] * sigmoid((1 - distance) * sharpness)
                    cimg[y, x] += img[yy, xx] * sigmoid((1 - distance) * sharpness)
    #showImage(coriginal / np.max(coriginal), 0)
    color = np.sum(coriginal - cimg, (0, 1)) / sigtotal
    #print(color)
    for y in range(2 * samples):
        for x in range(2 * samples):
            xx = int(resConstx * (x - samples) + cx)
            yy = int(resConsty * (y - samples) + cy)
            tiltedx = (xx - cx) * math.cos(-tilt) - (yy - cy) * math.sin(-tilt)
            tiltedy = (yy - cy) * math.cos(-tilt) + (xx - cx) * math.sin(-tilt)
            if xx >= 0 and xx < cols and yy >= 0 and yy < rows:
                coriginal[y, x] = original[yy, xx]
                new[y, x] = img[yy, xx]
                distance = math.pow((tiltedx) / width, 2) + math.pow((tiltedy) / height, 2)
                if distance < 1:
                    new[y, x] += color * sigmoid((1 - distance) * sharpness)
    crows, ccols, cchannels = coriginal.shape
    #showImage(new / np.max(new), 0)
    #print(getImageScore(coriginal, old, new, mode) * (width * height) / (crows * ccols))
    return getImageScore(coriginal, old, new, mode) * (width * height) / (crows * ccols) * np.sum(np.abs(color)) / abs(math.log10(height / width))

def generateCandidate(original, img):
    rng = np.random.default_rng()
    rows, cols, channels = img.shape
    x = rng.uniform(0, cols)
    y = rng.uniform(0, rows)
    height = int(math.pow(rows, random.randrange(0, 100) / 100))
    width = random.randrange(1, cols)
    tilt = rng.uniform(0, math.pi / 2)
    sharpness = np.exp(rng.uniform(0, 2))
    return np.asarray([x, y, height, width, tilt, sharpness]).astype(np.float32)

def generateNCandidates(original, img, n):
    candidates = np.zeros((n, 6), dtype= np.float32)
    for i in range(n):
        candidates[i] = generateCandidate(original, img)
    return candidates

def generateImgWithBlotch(original, img, cx, cy, height, width, tilt, sharpness):
    rows, cols, channels = original.shape
    diff = np.zeros_like(original)
    sigtotal = 0
    for y in range(rows):
        for x in range(cols):
            tiltedx = (x - cx) * math.cos(-tilt) - (y - cy) * math.sin(-tilt)
            tiltedy = (y - cy) * math.cos(-tilt) + (x - cx) * math.sin(-tilt)
            distance = math.pow((tiltedx) / width, 2) + math.pow((tiltedy) / height, 2)
            diff[y, x] = (original[y, x] - img[y, x]) * sigmoid((1 - distance) * sharpness)
            sigtotal += sigmoid((1 - distance) * sharpness)
    color = np.sum(diff, (0, 1)) / sigtotal
    #print(color)
    #showImage(diff / np.max(diff), 0)
    diff = np.zeros_like(original)
    for y in range(rows):
        for x in range(cols):
            tiltedx = (x - cx) * math.cos(-tilt) - (y - cy) * math.sin(-tilt)
            tiltedy = (y - cy) * math.cos(-tilt) + (x - cx) * math.sin(-tilt)
            distance = math.pow((tiltedx) / width, 2) + math.pow((tiltedy) / height, 2)
            diff[y, x] += color * sigmoid((1 - distance) * sharpness)
    #showImage(diff / np.max(diff), 0)
    return img + diff, color

def generateChild(original, candidate):
    posvariance = 20
    sizevariance = 0.3
    tiltvariance = 0.2
    sharpnessvariance = 0.1
    rng = np.random.default_rng()
    rows, cols, channels = original.shape
    x = min(max(candidate[0] + posvariance * rng.normal(0, 1), 0), cols)
    y = min(max(candidate[1] + posvariance * rng.normal(0, 1), 0), rows)
    height = candidate[2] * max(rng.normal(1, sizevariance), 1 - sizevariance)
    width = candidate[3] * max(rng.normal(1, sizevariance), 1 - sizevariance)
    tilt = candidate[4] + math.pi * tiltvariance * rng.normal(0, 1)
    sharpness = candidate[5] * rng.normal(1, sharpnessvariance)
    '''x = min(max(candidate[0] + posvariance * rng.uniform(-1, 1), 0), cols)
    y = min(max(candidate[1] + posvariance * rng.uniform(-1, 1), 0), rows)
    height = candidate[2] * rng.uniform(1 - sizevariance, 1 + sizevariance)
    width = candidate[3] * rng.uniform(1 - sizevariance, 1 + sizevariance)
    tilt = candidate[4] + math.pi * rng.uniform(-tiltvariance, tiltvariance)
    sharpness = candidate[5] * rng.uniform(1 - sharpnessvariance, 1 + sharpnessvariance)'''

    return np.asarray([x, y, height, width, tilt, sharpness]).astype(np.float32)

original = getImage(0).astype(np.float32)[0::5, 0::5]
img = np.zeros_like(original, dtype= np.float32)
img[:, :, 0].fill(np.average(original, (0, 1))[0])
img[:, :, 1].fill(np.average(original, (0, 1))[1])
img[:, :, 2].fill(np.average(original, (0, 1))[2])
print(img.shape)
mode = 'quadratic'
for i in range(1000):
    print(i)
    gray = (np.zeros_like(original))
    gray.fill(-np.min(original - img))
    showImage(img / np.max(img), 1)
    #showImage((original - img + gray) / np.max(original - img + gray), 0)
    n = 5 #trim size
    m = 10 #children - technically, the candidate survives and produces two children
    candidates = generateNCandidates(np.copy(original), np.copy(img), n * m)
    for g in range(5):
        scores = np.zeros((n * m), dtype= np.float32)
        orderedCandidates = np.zeros_like(candidates)
        for c in range(n * m):
            scores[c] = getScoreWithBlotch(np.copy(original), np.copy(img), mode, candidates[c, 0], candidates[c, 1], candidates[c, 2], candidates[c, 3], candidates[c, 4], candidates[c, 5])
        for c in range(n):
            bestIndex = np.argmin(scores)
            orderedCandidates[c] = candidates[bestIndex]
            #print(candidates[bestIndex])
            #print(scores[bestIndex])
            scores[bestIndex] = 1000000000.0
            for child in range((c * (m - 1)) + n, ((c + 1) * (m - 1)) + n):
                orderedCandidates[child] = generateChild(np.copy(original), candidates[bestIndex])
        candidates = np.copy(orderedCandidates)
    if candidates[0, 2] * candidates[0, 3] > 100:
        newImg, color = generateImgWithBlotch(original, img, candidates[0, 0], candidates[0, 1], candidates[0, 2], candidates[0, 3], candidates[0, 4], candidates[0, 5])
        if np.sum(np.abs(color)) > 15.0:
            img = newImg

showImage(img / np.max(img), 0)

'''candidates = generateNCandidates(np.copy(original), np.copy(img), 10)
        bestScore = 0
        bestCandidate = candidates[0]
        for c in candidates:
            print(c)
            currScore = getScoreWithBlotch(np.copy(original), np.copy(img), mode, c[0], c[1], c[2], c[3], c[4], c[5])
            if currScore < bestScore:
                bestScore = currScore
                bestCandidate = c
        print(bestCandidate)'''