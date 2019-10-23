import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

#
#def get_team_extras(player, teams, thestat):
#      
#    thegameidjoin = teams.loc[player.GameID == teams.GameID]          
#    
#    if thestat == "team_minutes":
#        theteammin = thegameidjoin.loc[thegameidjoin.Team == player.Team].get('Min').iloc[0]       
#        return theteammin
#    
#    if thestat == "team_TRB":
#        theteamTRB = thegameidjoin.loc[thegameidjoin.Team == player.Team].get('Reb').iloc[0]          
#        return theteamTRB
#    
#    if thestat == "Tm_DRB":
#        theteamDRB = thegameidjoin.loc[thegameidjoin.Team == player.Team].get('Dreb').iloc[0]          
#        return theteamDRB    
#    
#    if thestat == "Tm_FTA":
#        theteam_FTA = thegameidjoin.loc[thegameidjoin.Team == player.Team].get('FTA').iloc[0]            
#        return theteam_FTA   
#        
#    if thestat == "Tm_FGP":
#        theteam_FGP = thegameidjoin.loc[thegameidjoin.Team == player.Team].get('FGP').iloc[0]            
#        return theteam_FGP   
#    
#    if thestat == "Tm_FGA":
#        theteam_FGA = thegameidjoin.loc[thegameidjoin.Team == player.Team].get('FGA').iloc[0]            
#        return theteam_FGA   
#    
#    if thestat == "Tm_TOV":
#        theteam_TOV = thegameidjoin.loc[thegameidjoin.Team == player.Team].get('TO').iloc[0]            
#        return theteam_TOV   
#    
#    if thestat == "Tm_ORB":
#        theteam_Orb = thegameidjoin.loc[thegameidjoin.Team == player.Team].get('Oreb').iloc[0]            
#        return theteam_Orb  
#
#    if thestat == "Opp_TRB":
#        theOpp_TRB = thegameidjoin.loc[thegameidjoin.Team != player.Team].get('Reb').iloc[0]            
#        return theOpp_TRB   
#    
#    if thestat == "Opp_DRB":
#        theOpp_DRB  = thegameidjoin.loc[thegameidjoin.Team != player.Team].get('Dreb').iloc[0]            
#        return theOpp_DRB  
#    
#    if thestat == "Opp_ORB":
#        theOpp_ORB  = thegameidjoin.loc[thegameidjoin.Team != player.Team].get('Oreb').iloc[0]            
#        return theOpp_ORB  
#    
#    if thestat == "Opp_FGA":
#        theOpp_FGA  = thegameidjoin.loc[thegameidjoin.Team != player.Team].get('FGA').iloc[0]            
#        return theOpp_FGA  
#    
#    if thestat == "Opp_TOV":
#        theOpp_TOV  = thegameidjoin.loc[thegameidjoin.Team != player.Team].get('TO').iloc[0]            
#        return theOpp_TOV  
#    
#    if thestat == "Opp_FTA":
#        theOpp_FTA  = thegameidjoin.loc[thegameidjoin.Team != player.Team].get('FTA').iloc[0]            
#        return theOpp_FTA  
#    
#    if thestat == "Opp_FGP":
#        theOpp_FGP  = thegameidjoin.loc[thegameidjoin.Team != player.Team].get('FGP').iloc[0]            
#        return theOpp_FGP  
#    
#    return 


def main():
    # read data
    player_box_score = pd.read_csv('Data/player_box_score_cleaned.csv')       
    team_box_score = pd.read_csv('Data/team_box_score_cleaned.csv')
    
    player_box_score =  player_box_score.drop(['Location'], axis=1)  #Location name contains commas for french location names. comma created new columns.dont need location for this 
    team_box_score =  team_box_score.drop(['Location'], axis=1)
    
    team_box_score_test = team_box_score[['GameID', 'Team', 'HomeTeam', 'AwayTeam', 'PlayerID', 'Min', 'Reb', 'Dreb', 'FTA', 'FGP', 'FGA', 'TO', 'Oreb']]
    
    
    player_box_score_home = player_box_score[player_box_score['Team'] == player_box_score['HomeTeam']]
    player_box_score_away = player_box_score[player_box_score['Team'] != player_box_score['HomeTeam']]
    
    player_box_score_home = player_box_score_home.merge(team_box_score_test, left_on = ['GameID', 'Team'], right_on = ['GameID', 'Team'], suffixes = ['', '_TM'])
    player_box_score_home = player_box_score_home.merge(team_box_score_test, left_on = ['GameID', 'AwayTeam'], right_on = ['GameID', 'Team'], suffixes = ['', '_Opp'])
    
    player_box_score_away = player_box_score_away.merge(team_box_score_test, left_on = ['GameID', 'Team'], right_on = ['GameID', 'Team'], suffixes = ['', '_TM'])
    player_box_score_away = player_box_score_away.merge(team_box_score_test, left_on = ['GameID', 'HomeTeam'], right_on = ['GameID', 'Team'], suffixes = ['', '_Opp'])
    
    player_box_score = pd.concat([player_box_score_home, player_box_score_away])
    
    
    player_box_score["TOV_percent"] = 100 * player_box_score.TO / (player_box_score.FGA + 0.44 * player_box_score.FTA + player_box_score.TO)   
    player_box_score["TRB_percent"] = 100 * ((player_box_score.Reb_TM * (player_box_score.Min_TM / 5)) / (player_box_score.Min * (player_box_score.Reb_TM + player_box_score.Reb_Opp)))
    player_box_score["USG_percent"] = 100 * ((player_box_score.FGA + 0.44 * player_box_score.FTA + player_box_score.TO) * (player_box_score.Min_TM / 5)) / (player_box_score.Min * (player_box_score.FGA_TM + 0.44 * player_box_score.FTA_TM + player_box_score.TO_TM))
    player_box_score["AST_percent"] = 100 * player_box_score.Ast / (((player_box_score.Min / (player_box_score.Min_TM / 5)) * player_box_score.FGA_TM) - player_box_score.FGA)
    
    player_box_score["Opp_Poss"] = 0.5 * ((player_box_score.FGA_TM + 0.4 * player_box_score.FTA_TM - 1.07 * (player_box_score.Oreb_TM / 
                    (player_box_score.Oreb_TM + player_box_score.Dreb_Opp)) * (player_box_score.FGA_TM - player_box_score.FGP_TM) + player_box_score.TO_TM) + 
                    (player_box_score.FGA_Opp + 0.4 * player_box_score.FTA_Opp - 1.07 * (player_box_score.Oreb_Opp / 
                    (player_box_score.Oreb_Opp + player_box_score.Dreb_TM)) * (player_box_score.FGA_Opp - player_box_score.FGP_Opp) + player_box_score.TO_Opp))
    
    player_box_score["STL_percent"] = 100 * (player_box_score.STL * (player_box_score.Min_TM / 5)) / (player_box_score.Min * player_box_score.Opp_Poss)     
    
    
    player_box_score = player_box_score.dropna(axis=0, how='any')
    
    player_box_score = player_box_score.loc[player_box_score.TRB_percent <= 100] # collumns with minutes played of zero creates inf numbers. drop these rows
    

    player_box_score.to_csv('Data/advanced_stats.csv')
    
    
    #added these lines for further dev       
#    TS_percent, #True Shooting Percentage; the formula is PTS / (2 * TSA). True shooting percentage is a measure of shooting     
#    TRB_percent, #Total Rebound Percentage (available since the 1970-71 season in the NBA); the formula is 100 * (TRB * (Tm MP / 5)) / (MP * (Tm TRB + Opp TRB)). Total rebound percentage is an estimate of the percentage of available rebounds a player grabbed while he was on the floor.
#    TOV_percent, #Turnover Percentage (available since the 1977-78 season in the NBA); the formula is 100 * TOV / (FGA + 0.44 * FTA + TOV). Turnover percentage is an estimate of turnovers per 100 plays.
#    STL_percent, #Steal Percentage (available since the 1973-74 season in the NBA); the formula is 100 * (STL * (Tm MP / 5)) / (MP * Opp Poss). Steal Percentage is an estimate of the percentage of opponent possessions that end with a steal by the player while he was on the floor.
#    PER, #Player Efficiency Rating (available since the 1951-52 season); PER is a rating developed by ESPN.com columnist John Hollinger. In John's words, "The PER sums up all a player's positive accomplishments, subtracts the negative accomplishments, and returns a per-minute rating of a player's performance." Please see the article Calculating PER for more information. Also see VAA and VAR.
#    AST_percent, #Assist Percentage (available since the 1964-65 season in the NBA); the formula is 100 * AST / (((MP / (Tm MP / 5)) * Tm FG) - FG). Assist percentage is an estimate of the percentage of teammate field goals a player assisted while he was on the floor.
#    USG_percent #Usage Percentage (available since the 1977-78 season in the NBA); the formula is 100 * ((FGA + 0.44 * FTA + TOV) * (Tm MP / 5)) / (MP * (Tm FGA + 0.44 * Tm FTA + Tm TOV)). Usage percentage is an estimate of the percentage of team plays used by a player while he was on the floor.
    
    
#    Poss
#   Possessions (available since the 1973-74 season in the NBA); the formula for teams is 0.5 * ((Tm FGA + 0.4 * Tm FTA - 1.07 * (Tm ORB / (Tm ORB + Opp DRB)) * (Tm FGA - Tm FG) + Tm TOV) + (Opp FGA + 0.4 * Opp FTA - 1.07 * (Opp ORB / (Opp ORB + Tm DRB)) * (Opp FGA - Opp FG) + Opp TOV)). 
#   This formula estimates possessions based on both the team's statistics and their opponent's statistics, then averages them to provide a more stable estimate. Please see the article Calculating Individual Offensive and Defensive Ratings for more information.
#   
#    player_box_score["Tm_MP"] = player_box_score.apply(lambda x: (get_team_extras(x , team_box_score, "team_minutes")), axis=1)    
#    player_box_score["Tm_TRB"] = player_box_score.apply(lambda x: (get_team_extras(x , team_box_score, "team_TRB")), axis=1)    
#    
#    player_box_score["Tm_FTA"] = player_box_score.apply(lambda x: (get_team_extras(x , team_box_score, "Tm_FTA")), axis=1)
#    player_box_score["Tm_FGA"] = player_box_score.apply(lambda x: (get_team_extras(x , team_box_score, "Tm_FGA")), axis=1)
#    player_box_score["Tm_TOV"] = player_box_score.apply(lambda x: (get_team_extras(x , team_box_score, "Tm_TOV")), axis=1)    
#    player_box_score["Tm_ORB"] = player_box_score.apply(lambda x: (get_team_extras(x , team_box_score, "Tm_ORB")), axis=1)
#    player_box_score["Tm_DRB"] = player_box_score.apply(lambda x: (get_team_extras(x , team_box_score, "Tm_DRB")), axis=1)
#    player_box_score["Tm_FGP"] = player_box_score.apply(lambda x: (get_team_extras(x , team_box_score, "Tm_FGP")), axis=1)
#    
#    player_box_score["Opp_TRB"] = player_box_score.apply(lambda x: (get_team_extras(x , team_box_score, "Opp_TRB")), axis=1)
#    player_box_score["Opp_DRB"] = player_box_score.apply(lambda x: (get_team_extras(x , team_box_score, "Opp_DRB")), axis=1)
#    player_box_score["Opp_ORB"] = player_box_score.apply(lambda x: (get_team_extras(x , team_box_score, "Opp_ORB")), axis=1)
#    player_box_score["Opp_FGA"] = player_box_score.apply(lambda x: (get_team_extras(x , team_box_score, "Opp_FGA")), axis=1)
#    player_box_score["Opp_TOV"] = player_box_score.apply(lambda x: (get_team_extras(x , team_box_score, "Opp_TOV")), axis=1)
#    player_box_score["Opp_FTA"] = player_box_score.apply(lambda x: (get_team_extras(x , team_box_score, "Opp_FTA")), axis=1)
#    player_box_score["Opp_FGP"] = player_box_score.apply(lambda x: (get_team_extras(x , team_box_score, "Opp_FGP")), axis=1)
#    
#    
#    player_box_score["TOV_percent"] = 100 * player_box_score.TO / (player_box_score.FGA + 0.44 * player_box_score.FTA + player_box_score.TO)   
#    player_box_score["TRB_percent"] = 100 * ((player_box_score.Tm_TRB * (player_box_score.Tm_MP / 5)) / (player_box_score.Min * (player_box_score.Tm_TRB + player_box_score.Opp_TRB)))
#    player_box_score["USG_percent"] = 100 * ((player_box_score.FGA + 0.44 * player_box_score.FTA + player_box_score.TO) * (player_box_score.Tm_MP / 5)) / (player_box_score.Min * (player_box_score.Tm_FGA + 0.44 * player_box_score.Tm_FTA + player_box_score.Tm_TOV))
#    player_box_score["AST_percent"] = 100 * player_box_score.Ast / (((player_box_score.Min / (player_box_score.Tm_MP / 5)) * player_box_score.Tm_FGA) - player_box_score.FGA)
#    
#    player_box_score["Opp_Poss"] = 0.5 * ((player_box_score.Tm_FGA + 0.4 * player_box_score.Tm_FTA - 1.07 * (player_box_score.Tm_ORB / 
#                    (player_box_score.Tm_ORB + player_box_score.Opp_DRB)) * (player_box_score.Tm_FGA - player_box_score.Tm_FGP) + player_box_score.Tm_TOV) + 
#                    (player_box_score.Opp_FGA + 0.4 * player_box_score.Opp_FTA - 1.07 * (player_box_score.Opp_ORB / 
#                    (player_box_score.Opp_ORB + player_box_score.Tm_DRB)) * (player_box_score.Opp_FGA - player_box_score.Opp_FGP) + player_box_score.Opp_TOV))
#    
#    player_box_score["STL_percent"] = 100 * (player_box_score.STL * (player_box_score.Tm_MP / 5)) / (player_box_score.Min * player_box_score.Opp_Poss)     
#    
#    
#    player_box_score = player_box_score.dropna(axis=0, how='any')
#    
#    player_box_score = player_box_score.loc[player_box_score.TRB_percent <= 100] # collumns with minutes played of zero creates inf numbers. drop these rows
#    
#
#    player_box_score.to_csv('Data/advanced_stats.csv')



if __name__ == '__main__':
    main()