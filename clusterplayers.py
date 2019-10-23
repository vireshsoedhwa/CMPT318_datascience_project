# Code adapted from clustering exercise on weather data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

def get_clusters(X, k):
    model = make_pipeline(
         KMeans(k, random_state = 42)
    )
    model.fit(X)
    return model.predict(X)

def get_pca(X, p):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(p, random_state = 1)
    X2 = pca.fit_transform(X_scaled)
    
    print("Variance Explained by PCA")
    print(pca.fit(X).explained_variance_ratio_)
    
    assert X2.shape == (X.shape[0], p)
    return X2
    
def main():
    # read data
    player_box_score = pd.read_csv('Data/player_box_score_cleaned.csv')
    advanced = pd.read_csv('Data/advanced_stats.csv')
    
    print("Clusters for Individual Games")
    # Cluster with PCA on offensive style
    off_style = player_box_score.loc[:, ['FGA2', 'FGA3', 'FTA', 'Oreb', 'Ast', 'TO']]
    off_style_pca = get_pca(off_style, 2)
    clusters = get_clusters(off_style_pca, 6)
    
    # For tie breaking purposes
    clusters[clusters == 0] = 6
    clusters[clusters == 2] = 0
    clusters[clusters == 1] = 2
    clusters[clusters == 4] = 1
    clusters[clusters == 5] = 4
    clusters[clusters == 3] = 5
    labels = ["Bigs", "Ball Handlers", "Wings", "", "LU Bigs/Wings", "LU BH/Wings", "Bench"]
    
    # attach clusters to dataframes to summarise
    player_box_score['clusters'] = clusters
    off_style['clusters'] = clusters
    
    # summarise, offense, defense
    print("\n")
    print("Summary of Offensive Style (clustered features)")
    print(off_style.groupby(['clusters']).agg(['mean']))
    print("\n")
    print("Summary of Defensive Style (features not clustered)")
    print(player_box_score.groupby(['clusters'])['Min', 'Dreb', 'STL', 'BLK', 'PTS'].agg(['mean']))
    print("\n")
    # shooting percentages
    shooting = player_box_score.groupby(['clusters'])['FGM2', 'FGA2', 'FGM3', 'FGA3', 'FTM', 'FTA'].agg(['sum'])
    # ravel code adapted
    #from https://stackoverflow.com/questions/19078325/naming-returned-columns-in-pandas-aggregate-function
    shooting.columns = [x[0]for x in shooting.columns.ravel()]
    shooting['FGP2'] = shooting['FGM2'].div(shooting['FGA2'], axis = 0)
    shooting['FGP3'] = shooting['FGM3'].div(shooting['FGA3'], axis = 0)
    shooting['FTP'] = shooting['FTM'].div(shooting['FTA'], axis = 0)
    # shooting summary
    
    print("Shooting Summary")
    print(shooting) 

    # merge advanced stats
    advanced = advanced[['GameID', 'PlayerID', 'TOV_percent', 'TRB_percent', 'USG_percent', 'AST_percent', 'STL_percent']]
    player_box_score_adv = player_box_score.merge(advanced, on=['GameID', 'PlayerID'])
    
    print("Summary of Advanced Stats")
    print(player_box_score_adv.groupby(['clusters'])['USG_percent', 'AST_percent'].agg(['mean']))
    print("\n")
    
    
    
    
    # plot figure
    print("Figure of PCA Components for individual game cluster")
    plt.scatter(off_style_pca[:, 0], off_style_pca[:, 1], c = clusters, cmap='tab10', edgecolor='k', s=20)
    plt.xlabel("Principle Component 1")
    plt.ylabel("Principle Component 2")
    plt.title("Individual Game Clustering")
    plt.show()
    plt.savefig('clusters_individual_game.png')
    
    # count
    player_clust = pd.DataFrame({
        'cluster': clusters,
        'PlayerID': player_box_score['PlayerID'],
    })
    counts = pd.crosstab(player_clust['PlayerID'], player_clust['cluster'])
    
    counts_logic = counts > 0
    
    
    # most common cluster is there position
    games = counts.sum(1)
    print("Average proportion of games that each player played at their classified position")
    print(np.mean(counts.max(1)/games))
    print("\n")
    counts['position'] = counts.idxmax(1)
    
    
    # for loops because there are only 6 clusters
    print("Size of Clusters")
    # Sizes for clusters
    for i in [0,1,2,4,5,6]:
        print("Size of Cluster", labels[i])
        print(len(counts[counts['position'] == i]))
    print("\n")
    
    print("Other positions that Bigs played")
    # Big
    for i in [0,1,2,4,5,6]:
        print("Number of Bigs who played a game at position", labels[i])
        print(sum(counts_logic[counts['position'] == 0][i]))
    print("\n")
    
    
    print("Other positions that Ball Handlers played")
    # Ball Handler
    for i in [0,1,2,4,5,6]:
        print("Number of Ball Handlers who played a game at position", labels[i])
        print(sum(counts_logic[counts['position'] == 1][i]))
    print("\n")

    print("Other positions that Wings played")
    # Wings
    for i in [0,1,2,4,5,6]:
        print("Number of Wings who played a game at position", labels[i])
        print(sum(counts_logic[counts['position'] == 2][i]))
    print("\n")
    
    print("Other positions that Bench players played")
    # No Time
    for i in [0,1,2,4,5,6]:
        print("Number of Bench players who played a game at position", labels[i])
        print(sum(counts_logic[counts['position'] == 6][i]))
    print("\n")
  
    
    games_played = player_box_score.groupby(['PlayerID'], as_index = False)['PlayerID'].agg(['count'])
    
    # mode code from
    #https://stackoverflow.com/questions/15222754/group-by-pandas-dataframe-and-select-most-common-string-factor
    # a players name is not always reported the same,
    # sometimes jj barea, others Jose juan barea for example
    # get most commonly used name, ties don't matter
    dfnames = player_box_score.groupby(['PlayerID'])['Name_x'].agg(lambda x:x.value_counts().index[0])
    dfclusters = player_box_score.groupby(['PlayerID'])['clusters'].agg(lambda x:x.value_counts().index[0])
    
    # aggregate over all games for a player and sum
    per_game_averages = player_box_score.groupby(['PlayerID'])['Min', 'FGM2', 'FGA2', 'FGM3', 'FGA3', 'FTM', 'FTA', 'Oreb', 'Dreb', 'Ast', 'PF', 'TO', 'STL', 'BLK', 'PTS'].agg(['sum'])
    
    # rename columns
    per_game_averages.columns = [x[0]for x in per_game_averages.columns.ravel()]
    # add the number of games played calculated above
    per_game_averages['games'] = games_played
    # calculate percentages
    per_game_averages['FGP2'] = per_game_averages['FGM2'].div(per_game_averages['FGA2'], axis = 0)
    per_game_averages['FGP3'] = per_game_averages['FGM3'].div(per_game_averages['FGA3'], axis = 0)
    per_game_averages['FTP'] = per_game_averages['FTM'].div(per_game_averages['FTA'], axis = 0)
    # calculate per game averages
    per_game_averages[['Min', 'FGM2', 'FGA2', 'FGM3', 'FGA3', 'FTM', 'FTA', 'Oreb', 'Dreb', 'Ast', 'PF', 'TO', 'STL', 'BLK', 'PTS']] = per_game_averages[['Min', 'FGM2', 'FGA2', 'FGM3', 'FGA3', 'FTM', 'FTA', 'Oreb', 'Dreb', 'Ast', 'PF', 'TO', 'STL', 'BLK', 'PTS']].div(per_game_averages['games'], axis = 0)
    # add the most common name and clusters
    per_game_averages['Name'] = dfnames
    per_game_averages['clusters'] = dfclusters
    # Check out names of top 10 scorers and their labels
    print("validation by name recognition")
    print(per_game_averages.sort_values(['PTS'],ascending = False)[['Name', 'clusters']].head(15))
    print("\n")
    
    print("Per Game Averages Clusters")
    
    # cluster on offensive style
    off_style_per_game = per_game_averages.loc[:, ['FGA2', 'FGA3', 'FTA', 'Oreb', 'Ast', 'TO']]
    off_style_pg_pca = get_pca(off_style_per_game, 2)
    clusters_pg = get_clusters(off_style_pg_pca, 5)
    
    # attach clusters to dataframes to summarise
    per_game_averages['clusters'] = clusters_pg
    off_style_per_game['clusters'] = clusters_pg
    
    # summarise Per game Clusters
    
    print("\n")
    print("Summary of Offensive Style (clustered features)")
    print(off_style_per_game.groupby(['clusters']).agg(['mean']))
    print("\n")
    print("Summary of Defensive Style and Shooting (features not clustered)")
    print(per_game_averages.groupby(['clusters'])['Min', 'FGP2', 'FGP3', 'FTP', 'Dreb', 'STL', 'BLK', 'PTS'].agg(['mean']))
    print("\n")
   
    print("Figure of PCA Components for per game averages cluster")
    plt.scatter(off_style_pg_pca[:, 0], off_style_pg_pca[:, 1], c = clusters_pg, cmap='tab10', edgecolor='k', s=20)
    plt.xlabel("Principle Component 1")
    plt.ylabel("Principle Component 2")
    plt.title("Per Game Average Clustering")
    # plt.savefig('clusters.png')
    
    
    
    
if __name__ == '__main__':
    main()