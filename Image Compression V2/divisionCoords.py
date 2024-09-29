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

def generateLine(coords, low, high):
    rng = np.random.default_rng()
    slope = math.tan(rng.uniform(-math.pi / 2, math.pi / 2))
    coord = coords[random.randrange(low, high)]
    return [slope, coord[0] - coord[1] * slope]

def generateCoords(img):
    rows, cols, channels = img.shape
    coords = np.zeros((rows * cols, 2), dtype= np.int32)
    for y in range(rows):
        for x in range(cols):
            coords[y * cols + x] = [y, x]
    return coords

def testCoord(coord, line):
    return coord[0] < coord[1] * line[0] + line[1]

def orderCoords(coords, bounds, line, low, high):
    inCoords = np.zeros((high - low, 2), dtype= np.int32)
    inLen = 0
    outCoords = np.zeros((high - low, 2), dtype= np.int32)
    outLen = 0
    print(outCoords.shape)
    for i in range(low, high):
        if testCoord(coords[i], line):
            inCoords[inLen] = coords[i]
            inLen += 1
        else:
            outCoords[outLen] = coords[i]
            outLen += 1
    coords[low:low + inLen] = inCoords[:inLen]
    coords[low + inLen:high] = outCoords[:outLen]
    #set lower half's upper bound lower, set upper half's lower bound higher
    bounds[low:low + inLen, 1].fill(low + inLen)
    bounds[low + inLen:high, 0].fill(low + inLen)
    return coords, bounds

def findRandomBounds(bounds):
    length, two = bounds.shape
    bound = bounds[random.randrange(0, length)]
    return bound[0], bound[1]

def generateNLines(coords, low, high, n):
    lines = np.zeros((n, 2))
    for i in range(n):
        lines[i] = generateLine(coords, low, high)
    return lines

def getLineScore(line, coords, original, low, high, res):
    #res = 5
    lowColors = np.zeros((int((high - low) / res) + 1, 3), dtype= np.int32)
    lowPointer = 0
    highColors = np.zeros((int((high - low) / res) + 1, 3), dtype= np.int32)
    highPointer = 0
    for i in coords[low:high:res]:
        if testCoord(i, line):
            lowColors[lowPointer] = original[i[0], i[1]]
            lowPointer += 1
        else:
            highColors[highPointer] = original[i[0], i[1]]
            highPointer += 1
    if lowPointer == 0 or highPointer == 0:
        return 0
    stdev = max(math.pow(np.std(lowColors[:lowPointer]) * np.std(highColors[:highPointer]), 0.5), 1)
    return np.sum(np.abs(np.average(lowColors[:lowPointer], 0) - np.average(highColors[:highPointer], 0))) * math.pow(min(lowPointer, highPointer) / max(lowPointer, highPointer), 0.5) / stdev

def findBestNCandidates(candidates, coords, original, low, high, trimSize, res):
    length, two = candidates.shape
    scores = np.zeros((length), np.int32)
    for c in range(length):
        scores[c] = getLineScore(candidates[c], coords, original, low, high, res)
    bestCandidates = np.zeros((trimSize, 2), dtype= np.float32)
    for i in range(trimSize):
        bestIndex = np.argmax(scores)
        bestCandidates[i] = candidates[bestIndex]
        scores[bestIndex] = 0
    return bestCandidates

def closestPointOnLine(line, point):
    if line[0] == 0:
        return point
    invSlope = -1 / line[0]
    #y = invSlope * (x - point[1]) + point[0]
    #y = line[0] * x + line[1]
    #invSlope * (x - point[1]) + point[0] = line[0] * x + line[1]
    #invSlope * x - invSlope * point[1] + point[0] = line[0] * x + line[1]
    #(invSlope - line[0]) * x = line[1] + invSlope * point[1] - point[0]
    if invSlope - line[0] != 0:
        x = (line[1] + invSlope * point[1] - point[0]) / (invSlope - line[0])
    else:
        x = point[1]
    y = line[0] * x + line[1]
    return [y, x]

def generateNChildren(line, n, averageCoord, stdevCoord):
    angleVariance = 0.1 #in half-rotations
    translationVariance = 0.1 #in standard deviations
    rotationPoint = closestPointOnLine(line, averageCoord)
    children = np.zeros((n, 2), dtype= np.float32)
    children[0] = line
    rng = np.random.default_rng()
    for i in range(1, n):
        slope = math.tan(math.atan(line[0]) + rng.uniform(-math.pi / 2, math.pi / 2) * angleVariance)
        translation = [rng.uniform(-translationVariance, translationVariance) * stdevCoord[0], rng.uniform(-translationVariance, translationVariance) * stdevCoord[0]]
        x = rotationPoint[1] + translation[1]
        y = rotationPoint[0] + translation[0]
        yIntercept = y - x * slope
        children[i] = [yIntercept, slope]
    return children

def generateImageWithCoords(original, coords, bounds):
    low, high = bounds[0]
    length, two = coords.shape
    img = np.zeros_like(original)
    stdevs = np.zeros((length), dtype= np.int32)
    while high <= length:
        colors = np.zeros((high - low, 3), dtype=np.int32)
        for i in range(low, high):
            colors[i - low] = original[coords[i, 0], coords[i, 1]]
        averageColor = np.average(colors, 0)
        stdevColor = np.sum(np.std(colors, 0))
        for i in range(low, high):
            img[coords[i, 0], coords[i, 1]] = averageColor
            stdevs[i] = stdevColor
        if high < length:
            low, high = bounds[high] #beautiful traversing!!!
        else:
            high += 1 #weird but probably works
    return img, stdevs

def testCoordVal(coord, line):
    return coord[1] * line[0] + line[1] - coord[0]

def addLineToImage(original, coords, line, low, high):
    lowColors = np.zeros((high - low, 3), dtype=np.int32)
    lowPointer = 0
    highColors = np.zeros((high - low, 3), dtype=np.int32)
    highPointer = 0
    for i in range(low, high):
        if not testCoord(coords[i], line):
            lowColors[lowPointer] = original[coords[i, 0], coords[i, 1]]
            lowPointer += 1
        else:
            highColors[highPointer] = original[coords[i, 0], coords[i, 1]]
            highPointer += 1
    lowAverageColor = np.average(lowColors[:lowPointer], 0)
    highAverageColor = np.average(highColors[:highPointer], 0)
    lowStdevColor = np.sum(np.std(lowColors[:lowPointer], 0))
    highStdevColor = np.sum(np.std(highColors[:highPointer], 0))
    blur = 0
    for i in range(low, high):
        val = testCoordVal(coords[i], line)
        if val <= -blur:
            img[coords[i, 0], coords[i, 1]] = lowAverageColor
            stdevs[i] = lowStdevColor
        elif val > blur:
            img[coords[i, 0], coords[i, 1]] = highAverageColor
            stdevs[i] = highStdevColor
        else:
            img[coords[i, 0], coords[i, 1]] = lowAverageColor * (0.5 - val / (2 * blur)) + highAverageColor * (0.5 + val / (2 * blur))
    return img, stdevs

original = getImage(0).astype(np.float32)[::1, ::1]
showImage(original / 256, 0)
img = np.zeros_like(original)
rows, cols, channels = original.shape
numOfDivisions = 10000
lines = np.zeros((numOfDivisions, 2), dtype= np.float32)
coords = generateCoords(original)
bounds = np.zeros_like(coords, dtype= np.int32)
#bounds[:, 0].fill(0) - unnecessary, but notable
bounds[:, 1].fill(rows * cols)
stdevs = np.zeros((rows * cols), dtype= np.int32)
trimSize = 5
numOfChildren = 5
generations = 20
for i in range(numOfDivisions):
    print(i)
    low, high = bounds[np.argmax(stdevs * stdevs * np.pow(bounds[:, 1] - bounds[:, 0], 0.5)), (0, 1)]#findRandomBounds(bounds)
    averageCoord = np.average(coords[low:high], 0)
    stdevCoord = np.std(coords[low:high], 0)
    candidates = generateNLines(coords, low, high, trimSize * numOfChildren)
    bestCandidates = findBestNCandidates(candidates, coords, original, low, high, trimSize, 10)
    for g in range(generations):
        #for 20 generations
        #n^20 = 20
        #n^0 = 1
        #n = 20throot20
        res = int(math.pow(math.pow(generations, 1 / generations), generations - g)) * 4
        print(g)
        for j in range(trimSize):
            candidates[j * numOfChildren:(j + 1) * numOfChildren] = generateNChildren(bestCandidates[j], numOfChildren, averageCoord, stdevCoord)
        bestCandidates = findBestNCandidates(candidates, coords, original, low, high, trimSize, res)
    lines[i] = bestCandidates[0] #bestCandidates is already sorted by score
    coords, bounds = orderCoords(coords, bounds, lines[i], low, high)
    #img, stdevs = generateImageWithCoords(original, coords, bounds)
    img, stdevs = addLineToImage(original, coords, lines[i], low, high)
    if i % 10 == 0:
        showImage(img / 256, 1)

showImage(img / 256, 1)

#coordsIndex = generateCoords(original)
#coords, bounds = orderCoords(coords, bounds, [1, 290], 0, rows * cols)
#print(coords, bounds)