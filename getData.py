from bs4 import BeautifulSoup
import requests
import time
from nba_py.constants import TEAMS
from selenium import webdriver
from defEffenciencies import allDefEff
import averageGames
import json

def get_defensive_efficiency(team, season):
    """
    Takes team string ex. BOS, CHI, LAC and season year
    Returns the defensive efficency of a team for the last ten games for a season
    """
    team_id = TEAMS[team]['id']
    prev_year = str(season-1)
    current_year_short = str(season)[-2:]
    stats_url = "https://stats.nba.com/team/"+team_id+"/advanced/?Season="+ \
    prev_year+"-"+current_year_short+"&SeasonType=Regular%20Season&Split=lastn&sort=DEF_RATING&dir=1"
    # run firefox webdriver
    driver = webdriver.Chrome(executable_path='/Users/user/develop/BasketballPredictor/chromedriver')
    # get web page
    driver.get(stats_url)
    # # sleep for 10s so javascript loads
    time.sleep(10)
    overall_stats_table = driver.find_elements_by_class_name("nba-stat-table__overflow")[0]
    team_defensive_eff_rating = overall_stats_table.find_elements_by_class_name("sorted")[1].text
    driver.quit()
    print(team + " has a defensive efficency rating of "+ \
    team_defensive_eff_rating + " for the " + prev_year+"-"+current_year_short + \
    " season")
    return team_defensive_eff_rating

def get_team_stats(team, season):
    """
    Takes team string ex. BOS, CHI, LAC and season year
    returns gamelog of team stats for
    """
    stats_url = "https://www.basketball-reference.com/teams/"+team+"/"+str(season)+"/gamelog/"
    response = requests.get(stats_url)
    if response.status_code == 404:
        print("There was a problem with getting the page:")
        print(stats_url)

    data_from_url = response.text
    soup = BeautifulSoup(data_from_url,"lxml")
    table = soup.find('table', {'id':"tgl_basic"})
    rows = table.find_all('tr')
    header = ['Rk', 'Date', 'Home or Away', 'Opp', 'W/L', 'Points', 'Opp',
     'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'TRB',
     'AST', 'STL', 'BLK', 'TOV', 'PF', 'Opponent FG', 'Opponent FGA',
     'Opponent FG%', 'Opponent 3P', 'Opponent 3PA', 'Opponent 3P%',
     'Opponent FT', 'Opponent FTA', 'Opponent FT%', 'Opponent ORB',
     'Opponent TRB', 'Opponent AST', 'Opponent STL', 'Opponent BLK',
     'Opponent TOV', 'Opponent PF', 'Opponent DER']

    output = [header]
    count = 0
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        if len(cols) > 0:
            # set away (@) to 1 and home ('') to 0
            if(cols[2] == '@'):
                cols[2] = '1'
            else:
                cols[2] = '0'
            if(cols[4] == 'W'):
                cols[4] = '1'
            else:
                cols[4] = '0'
            # get rid of blank col
            #set date and team to zero for easy csv
            cols[1] = '0'
            del cols[23]
            opponent = cols[3]
            cols[3] = '0'
            cols.append(allDefEff[opponent+str(season)])
            for i in range(len(cols)):
                if cols[i].isdigit() or "." in cols[i]:
                    cols[i] = float(cols[i])
            output.append(cols)
    return output

def makeCSV(team, season):
    csvName = team+str(season)+".csv"
    seasonLog = get_team_stats(team, season)
    averageGames.lastTenAvgsToCSV(seasonLog, csvName)
    
def makeAllCSVs():
    allTeams = TEAMS
    for x in range(2017,2020):
        for tm in allTeams:
            if tm=="BKN":
                tm="BRK"
            elif tm=="PHX":
                tm="PHO"
            elif tm=="CHA":
                tm="CHO"
            print("making csv for"+(tm+str(x)))
            makeCSV(tm,x)

#this got all the defeffs and put it into the txt file, and then we made it a py file to use the array
def collectDEF():
    allDefs = {}
    allTeams = TEAMS
    
    for x in range(2017,2020):
        for tm in allTeams:
            dfe = get_defensive_efficiency(tm, x)
            curKey = tm+str(x)
            allDefs[curKey] = dfe
                
    try:
        file = open( "defEffenciencies.txt", "w")
        file.write(json.dumps(allDefs))
        file.close()
    except:
        print("could not make text file...")
    print(allDefs)

# makeAllCSVs()
# get_team_stats('ATL', 2017)