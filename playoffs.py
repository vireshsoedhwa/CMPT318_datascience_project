import pandas as pd
import matplotlib.pyplot as plt
import string, re
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.weightstats import ttest_ind

"""
Non-playoff games are any games in the group phase, group round, 
or classification games
Playoff games are all other games and are games where the losing team loses out
on qualification contention for the Olympics from that tournament
    (ie TournamentID: 480, 525, 526, 527)

Offensive possessions (Opos)
0.5 * (
    (FGA + 0.4*FTA - 1.07*(OReb/(OReb + OpponentDReb))*(FGA - FGM) + TO) +
    ((OpponentFGA + 0.4*(OpponentFTA) - 1.07*(OpponentOReb)/(Opponent OReb + DReb)) *
        (OpponentFGA - OpponentFGM) + OpponentTO)
    )
    
Use to derive PPP (points per possession) to determine answer to problem 2
headers:
Min_home	FGM_home	FGA_home	FGP_home	FGM2_home	FGA2_home	FGP2_home	
FGM3_home	FGA3_home	FGP3_home	FTM_home	FTA_home	FTP_home	Oreb_home	
Dreb_home	Reb_home	Ast_home	PF_home	TO_home	STL_home	BLK_home	
PTS_home	ID_away	Team_away	PlayerID_away	Number_away	Name_away	Min_away
FGM_away	FGA_away	FGP_away	FGM2_away	FGA2_away	FGP2_away	FGM3_away
FGA3_away	FGP3_away	FTM_away	FTA_away	FTP_away	Oreb_away	Dreb_away
Reb_away	Ast_away	PF_away	TO_away	STL_away	BLK_away	PTS_away
"""


def main():
    
    # read data and clean extra index column
    team_box_scores = pd.read_csv('Data/games_cleaned.csv')
    team_box_scores = team_box_scores.drop('Unnamed: 0', axis=1)
    
    # distinguish game types - nonPlayoff vs playoff (aka all other games)
    # for 'Classification' use regex to find value
#    nonPlayoffGames = ['Group Phase', 'Group Round', 'Classification', 'Preliminary']
    team_box_scores['nonPlayoff'] = team_box_scores['Round of 16'].str.contains('Group') \
                                | team_box_scores['Round of 16'].str.contains('Classification') \
                                | team_box_scores['Round of 16'].str.contains('Preliminary')

    # create OPos column for _home and _away (roughly the same for both)
    team_box_scores['Opos_home'] = 0.5 * (
            (team_box_scores['FGA_home'] + 0.4*team_box_scores['FTA_home'] - 
             1.07*(team_box_scores['Oreb_home']/(team_box_scores['Oreb_home'] + 
                   team_box_scores['Dreb_away']))*
                   (team_box_scores['FGA_home'] - team_box_scores['FGM_home']) 
                   + team_box_scores['TO_home']) +
            (team_box_scores['FGA_away'] + 0.4*(team_box_scores['FTA_away']) - 
              1.07*(team_box_scores['Oreb_away']/(team_box_scores['Oreb_away'] +
                   team_box_scores['Dreb_home'])) *
             (team_box_scores['FGA_away'] - team_box_scores['FGM_away']) + 
             team_box_scores['TO_away'])
            )
    team_box_scores['Opos_away'] = 0.5 * (
            (team_box_scores['FGA_away'] + 0.4*team_box_scores['FTA_away'] - 
             1.07*(team_box_scores['Oreb_away']/(team_box_scores['Oreb_away'] + 
                   team_box_scores['Dreb_home']))*
                   (team_box_scores['FGA_away'] - team_box_scores['FGM_away']) 
                   + team_box_scores['TO_away']) +
            (team_box_scores['FGA_home'] + 0.4*(team_box_scores['FTA_home']) - 
              1.07*(team_box_scores['Oreb_home']/(team_box_scores['Oreb_home'] +
                   team_box_scores['Dreb_away'])) *
             (team_box_scores['FGA_home'] - team_box_scores['FGM_home']) + 
             team_box_scores['TO_home'])
            )

    # separate into two DFs
    # we create copies to deal with SettingWithCopyWarning
    non_playoffs = team_box_scores.copy()
    non_playoffs = non_playoffs.loc[non_playoffs['nonPlayoff'] == True]
    playoffs = team_box_scores.copy()
    playoffs = playoffs.loc[playoffs['nonPlayoff'] == False]
    
    # view distribution of pace for both types of games using sqrt rule for 
    # number of bins and get Opos means
    non_playoffs.hist(column='Opos_home', bins=17)
    plt.title("Non-Playoffs, offensive possessions")
    plt.xlabel('offensive possessions')
    plt.savefig('non_playoff_pace_distribution.png')
    playoffs.hist(column='Opos_home', bins=14)
    plt.title("Playoffs, offensive possessions")
    plt.xlabel('offensive possessions')
    plt.savefig('playoff_pace_distribution.png')
    playoffs_pace_mean = np.mean(playoffs['Opos_home'])
    non_playoffs_pace_mean = np.mean(non_playoffs['Opos_home'])

    # one sided t-test looking for playoff aka H1 pace slower than non-playoff
    # pace, where we do NOT assume the standard deviation of the samples to be
    # the same
    # the Opos for both teams are similar so we can default to one team: Opos_home
    pace_tstat, pace_p_val, pace_df = ttest_ind(
            non_playoffs['Opos_home'],
            playoffs['Opos_home'],
            alternative='larger', # default
            usevar='unequal')
    
    print("Do playoff games have a slower pace than non-playoff games?")
    print("The pace p-value is: {}".format(pace_p_val))
    print("Possesions in playoff games:", playoffs_pace_mean)
    print("Possesions in non-playoff games:", non_playoffs_pace_mean)
    
    # Observe total points per possession (tPPP) of the game, ie of both teams
    non_playoffs['tPPP'] = (non_playoffs['PTS_home'] + non_playoffs['PTS_away']) /\
                           (non_playoffs['Opos_home'] + non_playoffs['Opos_away'])
    playoffs['tPPP'] = (playoffs['PTS_home'] + playoffs['PTS_away']) /\
                           (playoffs['Opos_home'] + playoffs['Opos_away'])

    # view distribution of tPPP for both types of games using sqrt rule for 
    # number of bins
    non_playoffs.hist(column='tPPP', bins=17)
    plt.title("Non-Playoffs, total points per possesion")
    plt.xlabel('offensive efficiency')
    plt.savefig('non_playoff_tPPP_distribution.png')
    playoffs.hist(column='tPPP', bins=17)
    plt.title("Playoffs, total points per possesion")
    plt.xlabel('offensive efficiency')
    plt.savefig('playoff_tPPP_distribution.png')
    playoffs_tPPP_mean = np.mean(playoffs['tPPP'])
    non_playoffs_tPPP_mean = np.mean(non_playoffs['tPPP'])
    
    # one sided t-test looking for playoff aka H1 pace slower than non-playoff
    # pace, where we do NOT assume the standard deviation of the samples to be
    # the same
    tPPP_tstat, tPPP_p_val, tPPP_df = ttest_ind(
            non_playoffs['tPPP'],
            playoffs['tPPP'],
            alternative='larger', # default
            usevar='unequal')
    
    print("Do playoff games have a worse efficiency, ie lower tPPP, than non-playoff games?")
    print("The tPPP p-value is: {}".format(tPPP_p_val))
    print("Effeciency in playoff games:", playoffs_tPPP_mean)
    print("Efficiency in non-playoff games:", non_playoffs_tPPP_mean)
   
    """
    Deals with _home vs _away PPP, not needed for now
    # Observe points per possession (PPP) metric 
    non_playoffs['PPP_home'] = non_playoffs['PTS_home']/non_playoffs['Opos_home']
    non_playoffs['PPP_away'] = non_playoffs['PTS_away']/non_playoffs['Opos_away']
    playoffs['PPP_home'] = playoffs['PTS_home']/playoffs['Opos_home']
    playoffs['PPP_away'] = playoffs['PTS_away']/playoffs['Opos_away']
    
    # view distribution of PPP for both types of games using sqrt rule for 
    # number of bins
    # we compare home and away due to PTS varying for the two
    non_playoffs.hist(column='PPP_home', bins=17)
    plt.savefig('non_playoff_PPP_home_distribution.png')
    non_playoffs.hist(column='PPP_away', bins=17)
    plt.savefig('non_playoff_PPP_away_distribution.png')
    
    playoffs.hist(column='PPP_home', bins=14)
    plt.savefig('playoff_PPP_home_distribution.png')
    playoffs.hist(column='PPP_away', bins=14)
    plt.savefig('playoff_PPP_away_distribution.png')
    
    PPP_home_tstat, PPP_home_p_val, PPP_home_df = ttest_ind(
            non_playoffs['PPP_home'],
            playoffs['PPP_home'],
            alternative='larger', # default
            usevar='unequal')
    PPP_away_tstat, PPP_away_p_val, PPP_away_df = ttest_ind(
            non_playoffs['PPP_away'],
            playoffs['PPP_away'],
            alternative='larger', # default
            usevar='unequal')
    
    print("Do playoff games have a worse efficiency (ie lower PPP) than \
          non-playoff games?")
    print("The home and away, respectively, PPP p-values are: {}, {}"\
          .format(PPP_home_p_val, PPP_away_p_val))
    """
    
    
    
if __name__ == '__main__':
    main()    