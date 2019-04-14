from bs4 import BeautifulSoup
import requests

def get_team_stats(team, season):
    """
    Takes team string ex. BOS, CHI, LAC and season year
    returns gamelog of team stats for
    """
    stats_url = "https://www.basketball-reference.com/teams/+"team/season"+/gamelog/"
    response = requests.get(stats_url)
    if response.status_code == 404:
        print("There was a problem with getting the page:")
        print(stats_url)

    data_from_url = response.text
    soup = BeautifulSoup(data_from_url,"lxml")
    table = soup.find('table', {'id':"tgl_basic"})
    rows = table.find_all('tr')
    headers = [ele.text.strip() for ele in rows[1].find_all('th')]
    print(headers)
    output = []
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        if len(cols) > 0:
            output.append(cols)
    print(len(output))

get_team_stats('ATL', 2017)
