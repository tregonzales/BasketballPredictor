from collections import *

import csv
import random
import os
import os.path
import shutil

def lastTenAvgsToCSV(allGameRows, filename):
    rows = generateAllLastTenAvgs(allGameRows)
    write_to_csv(rows, filename)

def generateAllLastTenAvgs(gameRows):
    avgRows = []
    headerRow = gameRows[0]
    avgRows.append(headerRow)
    for i in range(11,len(gameRows)):
        #just get the points from the current game
        points = gameRows[i][5]
        newRow = getLastTenAvg(gameRows[i-10:i],points)
        avgRows.append(newRow)
    return avgRows

def getLastTenAvg(lastTenRows,points):
    avgSumRow = list(zip(*lastTenRows))
    summedRows = list(map(sum,avgSumRow))
    avgRow = list(map(lambda x: x/10, summedRows))
    avgRow.append(points)
    return avgRow

def write_to_csv( list_of_rows, filename ):
    try:
        csvfile = open( filename, "w", newline='' )
        filewriter = csv.writer( csvfile, delimiter=",")
        for row in list_of_rows:
            filewriter.writerow( row )
        csvfile.close()

    except:
        print("File", filename, "could not be opened for writing...")
