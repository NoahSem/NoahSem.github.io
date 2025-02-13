---
layout: post
title: Analysis of Gaming Trends
date: 2025-02-12
author: Noah Seminoff
categories: Data-Science
published: true
excerpt: If I were to create my own video game, what sort of things are important? What aspects are important in popular games? To this end, I wanted to analyze trends in the gaming ecosystem, to see what I could find out.
---
One of my favourite things to do is to make something. Whether this is music, woodworking, writing code; the feeling of bringing something to life that you made yourself never gets old. The feeling of "*I* made that". Lately, i've been trying my hand at combining this with one of my other favourite things to do, playing video games. There has never been a better time than the present to make video games. Numerous tutorials, documentation, free tools and game engines have made it incredibly easy to start. Additionally, the last decade has seen an increase in the popularity and number of releases of indie games (*indie* or *independent* games are made without the support of a large publisher, usually by a solo developer or a small team). If I were to create my own video game, what sort of things are important? What aspects are important in popular games? To this end, I wanted to analyze trends in the gaming ecosystem, to see what I could find out. A couple questions to start off:

- What are the most popular genres?
- Are popular games also newer, or are some older games still popular?
- Do popular games tend to stay popular?
- Do you need a large publisher to be successful?
- Can we predict player counts with machine learning?

## The Dataset

To analyze these trends, we will use Python to look into a dataset consisting of data following 2000 applications on the distribution platform Steam, that had the most concurrent users on December 11, 2017. This dataset (available [here](https://data.mendeley.com/datasets/ycy3sy3vj2/1)), was collected by a group of researchers (Wannigamage et al. 2020) using the Steam API (and Steam Spy API for tags) over a course of 3 years, between Dec 14, 2017 and August 12, 2020. This dataset is available under the Creative Commons 4.0 Open Source License (learn more [here](https://creativecommons.org/licenses/by/4.0/)).

The dataset consists of 7 files containing cross-sectional data about each application:

Filename | Features | notes
---------|----------|---------
`applicationDevelopers.csv` | appid, developers | list of developers for each app
`applicationGenres.csv` | appid, genres | list of genres provided by the developer
`applicationInformation.csv` | appid, type, name, releasedate, freetoplay | type is game, demo, mod, dlc, or advertising. freetoplay is 1.0 if app is free, 0.0 if paid
`applicationPackages.csv` | appid, packages | list of package ID's each application is included in
`applicationPublishers.csv` | appid, publishers | list of publisher names for each app
`applicationSupportedlanguages.csv` | appid, languages | list of languages each app is available in
`applicationTags.csv` | appid, tags | list of user-added tags for each app

For each of the files except for applicationInformation, the format is the same; each row contains an appid to index the applications, followed by a variable length list of comma separated values. These variable length lists may have no values, or it may have a large number of values. We will have to process this carefully later.

In addition to the static data, we also have a multitude of files containing time-series data for each application, on the topics of:
- `PriceHistory`:
    - 1512 csv files, one per paid application.
    - Files are named after the appid
    - starting on April 7, 2019 and ending on August 12, 2020
    - Features:
        - `Date`, formatted YYYY-MM-DD, one row per day
        - `Initialprice`, price of the application in USD, before any sales are applied
        - `Finalprice`, price in USD, after any sales are applied
        - `Discount`, percent discount from 0 to 100, to the nearest whole number
- `PlayerCountHistory`:
    - 2000 CSV files, one per application
    - Files are named after the appid
    - Features are `Time`, in YYYY-MM-DD HH:MM format, and `Playercount`, the current number of users who are using that application.
    - Data are divided into two parts, based on popularity and frequency of data collection.
    - Part 1 contains the first 1000 csv's, for the apps with the most active users. Data was collected in 5 minute increments.
    - Part 2 contains the next 1000 csv's, for apps with the 1001 to 2000th most active users. Data was collected in 1 hour increments.

### Preprocessing our Data

To work with this data, we'll first take a few steps to make it nicer to handle. Firstly, we will use the python package `pandas` to read in each of the cross-sectional data files into dataframes, and combine them into one larger dataframe. These files are all indexed by appid already, so this will make it nice and easy to put everything together. However, there is one problem; most of these files contain variable length lists. Ideally the data would be normalized, such that at each row and column combination there is only a single value, and not a list of values. There are a few options to proceed here.

Let's take the genres as an example. One option would be to find the application with the most genres, and use it's length as the number of columns (eg genre1, genre2, genre3, etc). This could end up being quite bulky if there is application with many genres, and there may be a lot of null values for applications with less genres. Another method would be to have a single genre column and repeat each row a number of times equal to its number of genres. This could work, but we would lose the property of each row being a single application, and if this is done for each of genres, developers, tags, packages, and languages the number of rows needed per application would skyrocket. Finally, we could just keep the list as a string, and extract it later. This has the downside of needing additional processing down the line, but this would the keep one-row-per-application property, and there would just be a single column for each list. Let's go with this one.

While we're at it, we can also calculate the length of each variable width list and store it (eg, number of developers, number of languages, etc). The next file to look at is Information. The only slight difficulty here is processing dates, but pandas provides a nice easy `parse_dates` and `date_format` kwargs to solve this. 

Next, we can look at the time-series data. We come to another problem; how do we store this data? We would like it all to be in one place, ideally the same place as our other information. To put all this data in the same dataframe one-application-per-row, we would need a column for each time value, which would give us an extra 23,353 columns. No thanks. The best solution for our purposes here is to reduce the dimensionality of this data. Let's simply calculate a few summary statistics for each application, and store those in a few columns in the central dataframe. For player count, the mean number of active players in each year is a good measure, and for price history, we'll grab a handful of things that might be handy later; the initial non-sale price, the median nonsale price, the median mean and minimum sale price, as well as the largest discount percentage.

Now we can simply concatenate these dataframes to stack them horizontally into one central dataframe. It may also be handy to have a few other measures, so here we also calculate the "main genre" as the first listed genre, the "release year" as the year part of the release date, and "player dropoff" as the difference between the mean active players per year in 2020 and 2017 (both in raw difference, and as a percentage of the 2017 player count). 

We can also impute some missing values. Namely, the `freetoplay` column has a handful of null values. However, we also know that we have price history only for paid games, so we can check whether the summary price statistics are null (which should only occur for games that are free), and use that to infer whether the game is free to play.

Finally, we want to analyze games and not dlc or tools or anything else, so we select only the rows where `type` is "game". This leaves us with 1851 games and 29 columns (not including `appid` or `type`) to investigate.

### What are the most popular genres?
First, lets take a look at the most common genres or user applied tags in popular games. I imagine this may differ slightly for free games versus paid games, so we can take that into account. Firstly, we can look at Genres. These are applied by the developers when they submit their game on Steam. If we recall, these are variable length lists stored as a single string. To analyze the most popular tags, we need to count the frequency distribution. Since we also want freetoplay information, I opted to manually loop through the dataframe and count occurances of each genre, and whether the game was free.


    
![png](/assets/blog/gamingtrends/2025-02-12-Gaming-Trends_7_0.png)
    


The overall length of each bar here is the total number of games for which each genre appears in. The bar is split into green for free games (347 games total), and blue for paid games (1504). A game may contain multiple genres. Here, we only show the top 12 genres by total count, because there are a lot of genres with only a couple of games, like "Education" or "Violent". 

A few observations:
- By far, the most common genres in the top 2000 popular games are Action and Indie.
- A few games are listed with the genre "Free to Play" in error, because they cost money. This may be an error in the dataset, or this may be games that were once free and now cost money.
- In free games, the most popular genre is "Free to Play" (obviously), followed by Action, then Massively Multiplayer (MMO), RPG, and then Indie. MMO style games are generally free up-front and then require a monthly subscription, so this is likely why they are more common within free games. Indie games being further down in free games is likely due to the cost associated with making games. A small developer putting their own time and money into making a game would most likely want to recoup some of those costs by charging for their game, even if it is a small amount. Although, there may be hobbyists who just want their game to be played, or free games that make money through in-app purchases that are still free.

We can also look at user applied tags to get a sense of what genres are popular. These are applied by steam users to a game, so there may be a slight difference in what a developer thinks their game is, and what users think their game is. Tags are also more open-ended, and can contain things other than genres. Again, this may differ in freetoplay, so we can color based on that factor. The analysis was done in the same way as the genres, where I looped through each game and counted the occurances of each tag.


    
![png](/assets/blog/gamingtrends/2025-02-12-Gaming-Trends_10_0.png)
    


Again, the overall length of the bar counts the number of games which contain that tag, and the bar is split between green for free games, and blue for paid games. We again only show the top 12.

Some obervations here:
- Tags can contain some non-genre things, such as features (singleplayer, multiplayer, open world), as well as feelings (atmospheric, great soundtrack)
- In terms of genres, we remain consistent with Action, Adventure, and Indie being the top 3, but here Adventure is second rank, while in Genres it was third. This may be due to the effect described early, whereby a developer may not think their game is an "Adventure" game, but Steam users do think so.
- The Singleplayer tag is more common than Multiplayer. However, a randomly chosen multiplayer game has a 26.16% chance to be free, while a singleplayer game has just a 13.58% chance (That is, a free game is 1.93x more likely to be multiplayer)

### How old are popular games?
Next, we can take a look at the distribution of the release dates for these games. I care more about how old the game is than what month or day it came out on, so i've binned the data based on the year it came out. Again, I have a suspicion that this may change with freetoplay games, so I've coloured the data based on that.


    
![png](/assets/blog/gamingtrends/2025-02-12-Gaming-Trends_12_0.png)
    


Of note here is that the data was collected for the games which were most popular in December 2017, but the data was collected until August 2020, so there are a handful of games which were popular in 2017, but didn't come out until later. This is because these games were in Early Access, which is a style of game publication where players have access to a game that is still in development and may change or not be entirely finished, but they can support the developers monetarily.

Interestingly:
- There doesn't appear to be any sort of pattern between regarding freetoplay games, they seem to follow the same distribution as the rest.
- The most common release year is 2017, which if we recall, is when the most popular games were initially chosen. This means that the most popular games are generally those that came out in the "current" year. The games which came out the previous year are also fairly popular, but they start to dropoff prior to this year.
- I also chose to model these values based on their difference between the "current" year and the year the game came out. That is, "how many years old is this game". The histogram looks roughly exponential, so my initial curve choice was of the form `$a/x$`, for constant `$a$` and game age `$x$`. It turns out that this fits quite well, and the number of popular games from a given year is inversely proportional to how old they are. Specifically, the best fit model was:
`$$
f(year) = \frac{403}{2017 - year}
$$`
Which has a correlation coefficient of  `$R^2 = 0.939$`. Another way to look at this would be, when you compare games released in a given year to those `$N$` years prior, we see `$1/N$` times as many games. (eg, there are 1/5th as many popular games in 2012 as 2017, a 5 year difference).

### Do popular games stay popular?
If a game is popular now, will it tend to stay popular, or do games tend to lose popularity over time? To find this, we can look at the mean active player count at the beginning of data collection, and compare it to the mean active player count at the end of the data collection. We will do this as a percentage of the player count at the beginning, so we get a percent difference between the two. Again, we will also split this by freetoplay.


    
![png](/assets/blog/gamingtrends/2025-02-12-Gaming-Trends_14_0.png)
    


We have similar bars here as before, total, split by free and paid. Additionally, since this is expressed as a percentage of the starting playercount, a value of zero on the x-axis means that a game retains exactly the same number of players between 2017 and 2020. Values that are negative have lost players, while values that are positive have gained players.

We can see that:
- Most games (70%) lose players, which matches what we saw in the release date analysis; older games tend to be played less than newer games, so it makes sense that a game would transition to a lower playercount as it ages.
- only 5.85% of games have doubled their users or more
- Free games peak at -100%, while paid games peak at -50%. A likely interpretation for this effect could be that users who did not pay for a product are more likely to stop using it much sooner because they have no attachment, while those who put hard earned money into a purchase feel more obligated to continue to play the game.

### Do you need a large publisher to be successful?
What does the distribution of publishers look like? Are there many successful games without publishers, or is it a landscape dominated by large publishers? To answer this, we can look at the list of publishers for each game, and count the occurances of each unique publisher name. This required a bit more effort, because a lot of publisher names have commas in them (eg, Noah's Cool Example Game Company, LLC). After cleaning the data to account for these, we get the below plot.


    
![png](/assets/blog/gamingtrends/2025-02-12-Gaming-Trends_18_0.png)
    


Here, we have a count of the number of games for which the publisher appears in their list of publishers. If none listed, they are categorized as "no publisher". We also limit the plot to the top 10 publishers, because there are a lot of publishers who are simply the developers listed as their own publisher (eg, self-publishing), and they only publish the one game.

Observations:
- The largest publishers are big game companies, like ubisoft and sega. These companies are responsible for publishing a large number of the most popular games. However, these are still only responsible for ~2% of popular games each.
- No Publisher comes in at rank 8.
- These top 10 only account for 322 of our 1851 games. So where does the rest come in?

We can analyze whether a game is self-published by checking if the developers and the publishers are the same:


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Count</th>
      <th>Percent</th>
    </tr>
    <tr>
      <th>Publisher</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Self Published/None</th>
      <td>921</td>
      <td>49.76</td>
    </tr>
    <tr>
      <th>Outside Publisher</th>
      <td>956</td>
      <td>51.65</td>
    </tr>
  </tbody>
</table>
</div>


A big surprise here! As it turns out, despite only 26 games having no publisher, an astounding 921 games (almost exactly half) are self published. This matches what we saw earlier, where nearly half of all games had Indie as a genre. Good news for small developers, your game can still be successful even if you dont have a big publisher behind you.

### Can we predict player counts with Machine Learning?
Last but not least, we can do a simple clustering analysis to try and group data points together. With this, we can simulate a new theoretical game, and predict which cluster it is in. We can then estimate some statistics about our theoretical game, by assuming it will have a similar distribution of its parameters to those in its same cluster.

Here, we are using a K-prototypes clustering algorithm (from the `kmodes` library). The algorithm groups data together based on a combination of Euclidean and Hamming distances, to classify data points into clusters of points that are close together in feature-space. This algorithm is designed to handle categorical as well as numerical data, which will be helpful in tackling our genre category. Since there can be a number of genres listed, the combination of possible genres, and possible orders of genres is too large. Instead, we will simply take the first listed genre, as that is usually the main or most applicable one. In addition to the main genre, we will also use the release year, freetoplay flag, number of genres, developers, packages, languages and publishers, as well as the mean sale price. We will skip over the player dropoff and playercount data, because we want to predict the player counts, and we will skip the name and list of specific publishers, languages, and developers, because they are generally quite specific.

The algorithm needs a number of clusters `$k$`, to use. We can estimate the best number using the "Elbow Method", where we plot the cost function versus the number of clusters, and see where additional clusters give us diminishing returns.


    
![png](/assets/blog/gamingtrends/2025-02-12-Gaming-Trends_26_0.png)
    


From the above plot, we can see that at k=3 clusters (the "elbow" of the graph), we start getting diminishing returns. Thus, we can choose k=3 clusters to proceed with analysis.

Let's choose a theoretical game to classify into a cluster. Let's say i'm going solo develop and self-publish a casual+puzzle game and release it this year (2017, the "current" year in our data). I'll charge $15 USD, make it only available in english, and it won't be available in any packages. What can I expect the typical player count to be? By taking summary statistics from games in the same cluster, we get:

    Active players prediction (mean): 2861
    Active players prediction (median): 164
    

From the games in our dataset that ended up in the same cluster as our theoretical game, the mean active players is 2861, but the median is only 164. Likely, there is a single game that ended up in our cluster, that skews the results. Originally, we did not include player counts in the clustering, so it is only being trained on other factors, and player count does not contribute to the distance calculation. However, we can still use the median, which will be more reflective of the distribution without the outliers. Here, we can expect a median player count of 164. Not bad.

What kind of games are in this cluster? We can take some summary statistics from the games in our cluster, but this time using the original data so we have all the columns:


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>releaseyear</th>
      <th>num_genres</th>
      <th>num_devs</th>
      <th>num_pkgs</th>
      <th>num_langs</th>
      <th>num_pubs</th>
      <th>num_tags</th>
      <th>2017</th>
      <th>median_sale_price</th>
      <th>largest_discount_pct</th>
      <th>playerdropoff_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>782.000000</td>
      <td>782.000000</td>
      <td>782.000000</td>
      <td>782.000000</td>
      <td>782.000000</td>
      <td>782.000000</td>
      <td>782.000000</td>
      <td>7.820000e+02</td>
      <td>782.000000</td>
      <td>782.000000</td>
      <td>782.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2014.419437</td>
      <td>2.402813</td>
      <td>1.232737</td>
      <td>1.625320</td>
      <td>7.549872</td>
      <td>1.170077</td>
      <td>17.079284</td>
      <td>2.861024e+03</td>
      <td>20.708696</td>
      <td>65.843990</td>
      <td>-5.382098</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.829881</td>
      <td>1.281261</td>
      <td>0.606877</td>
      <td>1.078968</td>
      <td>4.890550</td>
      <td>0.491125</td>
      <td>4.706370</td>
      <td>5.181986e+04</td>
      <td>5.791311</td>
      <td>20.366959</td>
      <td>98.746624</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2001.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.207486e+01</td>
      <td>9.990000</td>
      <td>0.000000</td>
      <td>-99.633746</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2013.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>15.250000</td>
      <td>7.394932e+01</td>
      <td>14.990000</td>
      <td>60.000000</td>
      <td>-55.098746</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2015.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>1.640730e+02</td>
      <td>19.990000</td>
      <td>75.000000</td>
      <td>-23.936534</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2017.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>4.475596e+02</td>
      <td>24.990000</td>
      <td>80.000000</td>
      <td>16.345505</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2019.000000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>14.000000</td>
      <td>27.000000</td>
      <td>4.000000</td>
      <td>20.000000</td>
      <td>1.443587e+06</td>
      <td>39.990000</td>
      <td>100.000000</td>
      <td>1323.876404</td>
    </tr>
  </tbody>
</table>
</div>


Here, the 2017 column is the number of active players averaged over 2017. It looks like the median same-cluster game is a few years old, with the same number of genres, devs, publishers, although with more languages and packages. The median game also has an active player count of 164 (as we saw above), sells for $20, occasionally goes on sale for 75% off, and loses 24% of its active playerbase over the course of a few years. All-in-all, it seems like we chose parameters well, and our theoretical game is decently successful.

## Future Steps
There are a couple of things in this dataset that I didn't get a chance to analyze, and may be interesting to add in the future;
- Analysis of applications with type other than "Game". While this is only a small fraction of the dataset, it may be interesting to see what trends exist.
- Analysis of languages. Do popular games tend to have more languages available? Is there a difference in the number of languages between self-published and outside published games?
- Price history analysis. What does the distribution of prices look like for popular games? Is there any correlation between active players and the price of a game? The price by genre?
- Are there common aspects between the games with the top 10 or top 50 active player counts? While the dataset contains the top 2000, are there different trends within the best of the best?
- Time-series analysis on the full price and player history data, instead of just the summary statistics.

## Conclusion
We set out to investigate trends in the video game ecosystem. To this end, we looked at a dataset of the top 2000 most active games on the platform Steam (Wannigamage et al. 2020). We investigated genres and tags, release dates for these games, active player counts each year, and publishers, as well as a K-prototypes machine learning clustering model. We found that the most common genres for popular games are Action and Indie. We also found that singleplayer games are more popular than multiplayer overall, although free to play games are more likely to be multiplayer. Looking at release dates, the current year is the most common while the number of older games in the top 2000 follow `$\sim 1/age$`. In terms of active players, 70% of games lose players after 3 years. Free to play games are most likely to completely lose their playerbase after this time, while paid games are more likely to hold on to a few players. We also saw that almost exactly half of all games are self-published, while the other half are supported by an outside publisher. We also applied the K-prototypes algorithm to cluster the games into 3 clusters, and used this to predict the number of active players we might expect for a new theoretical game.

## References
Wannigamage, Dulakshi; Barlow, Michael; Lakshika, Erandi; Kasmarik, Kathryn (2020), “Steam Games Dataset: Player count history, Price history and data about games”, Mendeley Data, V1, doi: 10.17632/ycy3sy3vj2.1
