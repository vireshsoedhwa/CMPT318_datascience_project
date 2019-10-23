import pandas as pd



def main():
    # read data
    df_boxscore = pd.read_csv('Data/cleanedboxscorestats.csv')
    df_game = pd.read_csv('Data/gameinformationUTF-8.csv')
    df_player = pd.read_csv('Data/PlayerIDs.csv')
    df_tournament = pd.read_csv('Data/tournamentlist.csv')
    
    
    df_game_merged = df_game.merge(df_boxscore, left_on = ['GameID', 'HomeTeam'], right_on = ['GameID', 'Name'], suffixes = ['_base', '_home'])
    df_game_merged = df_game_merged.merge(df_boxscore, left_on = ['GameID', 'AwayTeam'], right_on = ['GameID', 'Name'], suffixes = ['_home', '_away'])
    
    
    df_boxscore = df_boxscore.merge(df_game, on = 'GameID')
    df_boxscore = df_boxscore.merge(df_tournament, on = 'TournamentID')
    
    df_team_boxscore = df_boxscore.loc[df_boxscore['PlayerID'] == 0, ]
    df_player_boxscore = df_boxscore.loc[df_boxscore['PlayerID'] != 0, ]
    
    df_player_boxscore = df_player_boxscore.merge(df_player, on = 'PlayerID')
    
    df_team_boxscore.to_csv('Data/team_box_score_cleaned.csv', encoding='utf-8')
    df_player_boxscore.to_csv('Data/player_box_score_cleaned.csv', encoding='utf-8')
    df_game_merged.to_csv('Data/games_cleaned.csv', encoding='utf-8')
    
if __name__ == '__main__':
    main()