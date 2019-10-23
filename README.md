# How to run our code
### Libraries
  - pandas
  - sklearn
  - numpy
  - matplotlib

### Files expected in Data folder:
  - [cleanedboxscorestats.csv](Data/cleanedboxscorestats.csv)
  - [gameinformationUTF-8.csv](Data/gameinformationUTF-8.csv)
  - [PlayerIDs.csv](Data/PlayerIDs.csv)
  - [tournamentlist.csv](Data/tournamentlist.csv)


##### 1. Run [cleandata.py](cleandata.py)
Expected Output in Data folder:
  - team_box_score_cleaned.csv
  - player_box_score_cleaned.csv
  - games_cleaned.csv

##### 2. Run [make_advanced_stats.py](make_advanced_stats.py)
Expected Output in Data folder:
  - advanced_stats.csv

##### 3. Run [clusterplayers.py](clusterplayers.py)
Expected Output in console:
Summary stats detailed in report and graphs

##### 4. Run [playoffs.py](playoffs.py)
Expected Output in console:
Summary stats detailed in report and graphs