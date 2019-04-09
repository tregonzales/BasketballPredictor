from collections import *
#

#
# starter file for hw1pr2, cs35 spring 2017...
#

import csv
import random

def lastTenAvgsToCSV(allGameRows, filename):
    rows = generateAllLastTenAvgs(allGameRows)
    write_to_csv(rows, filename)

def generateAllLastTenAvgs(gameRows):
    avgRows = []
    for i in range(10,len(gameRows)):
        #just get the points from the current game
        points = gameRows[i][len(gameRows[i])-1]
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

def main():
    allgames = [None]*82
    for g in range(0,len(allgames)):
        newGameRow = [None]*5
        for i in range(0,len(newGameRow)):
            stat = random.randint(0,100)
            newGameRow[i] = stat
        allgames[g] = newGameRow

    lastTenAvgsToCSV(allgames, "bigBuckets.csv")


if __name__ == '__main__':
    main()
