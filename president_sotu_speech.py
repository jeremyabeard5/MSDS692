# president_sotu_speech.py

###############################################
# This project is meant to be a submission for the MSDS 692 initial Data Science Practicum
# The goal of this project is to analyze the State of the Union speeches given by the presidents of the United States
# It analyzes each speech and produces various visualizations and data output which can be used to show how each 
# president's speech compares to each other.
###############################################
# Thank you,
# Jeremy Beard
###############################################

# 0-step to any project: time it
import timeit
start_time = timeit.default_timer()

# First step to any project: hello world :)
print("Hello World!")

# Next step, import the libraries we'll need
import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from collections import defaultdict
#from collections import Counter
#import re
nltk.download('vader_lexicon')

# Next steps, define paths, directories
data_dir = "data"
speeches_dir = "sotu_firstname" # the '_firstname' directory has been modified so all filenames include first and last name (this took some time :) )
file_path = os.path.join(data_dir, speeches_dir)

# Just for fun, I wanted to test if I could see all the files
files = os.listdir(file_path)

# Create empty speech list
speeches = []#[{'prez': [], 'year': [], 'text': []}]

# Set figure dpi
dpi = 300

# Create test dataframe for use later
testdata = [
    {'president': 'George Washington', 'text': 'First President of the United States, he was great. Just great.'},
    {'president': 'Abraham Lincoln', 'text': '16th President of the United States, he loved slavery.'},
    {'president': 'Franklin D. Roosevelt', 'text': '32nd President of the United States'}
]
df_test = pd.DataFrame(testdata)

# Create a function to read in the speeches and filter the stopwords
def filter_stops(tex):
    stops = stopwords.words('english')
    words = tex.lower().split()
    filtered_words = [w for w in words if w not in set(stops)]
    return filtered_words

# Create a function to count the words of a string of text
def count_words(tex):
    if type(tex) == list:
        return len(tex)
    else:
        words = tex.lower().split()
        return len(words)
    
# Create a function that finds the amount of speeches per president
def speeches_per_president(tex):
    return df_speeches['prez'].value_counts()[tex]

def sentiment_analysis(tex):
    this_speechs_sentiments = []
    for word in tex:
        if word in sentiment_dict.keys():
            this_speechs_sentiments.append(sentiment_dict[word])
        else:
            this_speechs_sentiments.append(0)
    return (sum(this_speechs_sentiments) / len(this_speechs_sentiments))

# Main function
if __name__ == "__main__":
    
    # Iterate over files in directory with format "president_year.txt"
    for file in files:
        print(file)
        name, extension = file.split(".")
        president, year = name.split("_")
        print(president, year)
        print()
        
        # Add in the data from the text file to the 'speeches' list
        # 'speeches' will have all of the raw speeches, per year. 
        # later we'll consolidate them by president
        speeches.append({'prez': president, 'year': int(year), 'text': open(os.path.join(file_path, file)).read()})        
        
    print(f'Done reading in speeches, TIME: {timeit.default_timer() - start_time}')
    
    # 'speeches' looks like this:
    # speeches[0]
    # {'prez': 'Adams', 'year': 1797, 'text': 'Gentlemen of the Sen...ncurrence.'}
    # speeches[1]
    # {'prez': 'Adams', 'year': 1798, 'text': 'Gentlemen of the Sen... entitled.'}

    # Now that we have all the speeches, let's consolidate them by president
    # First step is to sort the 'speeches' list by year. This took a handful of syntactical tries :)
    sorted_list = sorted(speeches, key=lambda x: x['year'])
    # print(sorted_list)

    # 'sorted_list' looks like:
    # sorted_list[0]
    # {'prez': 'Washington', 'year': 1790, 'text': ''}
    # sorted_list[1]
    # {'prez': 'Washington', 'year': 1791, 'text': 'Fellow-Citizens of t...WASHINGTON'}

    # The following line of code is a bit of a doozy. 
    # It's a dictionary comprehension that takes the sorted list of speeches
    # and consolidates them by president.
    # so 'president_text['Washington']' will have all of Washington's speeches
    print(f'Sorting presidents, TIME: {timeit.default_timer() - start_time}')
    president_text = defaultdict(str)
    for speech in sorted_list:
        president_text[speech['prez']] += speech['text']
    
    # Now we use president_text to bring the consolidated speeches back in a list of dictionaries
    new_data = [{'prez': k, 'text': v} for k, v in president_text.items()]
    #print(new_data)
    
    # NOW let's create THREE dataframes: 
    #   One for the speeches as-is, alphabetized
    #   One for the speeches as-is, ordered chronologically
    #   One for the speeches joined per president, ordered chronologically
    df_speeches = pd.DataFrame(speeches)
    #               prez  year                                               text
    #0    AbrahamLincoln  1861  Fellow-Citizens of the Senate and House of Rep...
    #1    AbrahamLincoln  1862  Fellow-Citizens of the Senate and House of Rep...
    #2    AbrahamLincoln  1863  Fellow-Citizens of the Senate and House of Rep...
    #3    AbrahamLincoln  1864  Fellow-Citizens of the Senate and House of Rep...
    #4     AndrewJackson  1829  Fellow Citizens of the Senate and of the House...
    #..              ...   ...                                                ...
    #223   WoodrowWilson  1917  GENTLEMEN OF THE CONGRESS:\n\nEight months hav...
    #224   WoodrowWilson  1918  GENTLEMEN OF THE CONGRESS:\n\nThe year that ha...
    #225   WoodrowWilson  1919  TO THE SENATE AND HOUSE OF REPRESENTATIVES:\n\...
    #226   WoodrowWilson  1920  GENTLEMEN OF THE CONGRESS:\n\nWhen I addressed...
    #227   ZacharyTaylor  1849  Fellow-Citizens of the Senate and House of Rep...
    
    df_speeches_sorted = pd.DataFrame(sorted_list)
    #                 prez  year                                               text
    #0    GeorgeWashington  1790                                                   
    #1    GeorgeWashington  1791  Fellow-Citizens of the Senate and House of Rep...
    #2    GeorgeWashington  1792  Fellow-Citizens of the Senate and House of Rep...
    #    GeorgeWashington  1793  Fellow-Citizens of the Senate and House of Rep...
    #4    GeorgeWashington  1794  Fellow-Citizens of the Senate and House of Rep...
    #..                ...   ...                                                ...
    #223       BarackObama  2014  Mr. Speaker, Mr. Vice President, Members of Co...
    #224       BarackObama  2015  Mr. Speaker, Mr. Vice President, Members of Co...
    #225       BarackObama  2016  Mr. Speaker, Mr. Vice President, Members of Co...
    #226       DonaldTrump  2017  Thank you very much. Mr. Speaker, Mr. Vice Pre...
    #227       DonaldTrump  2018  Mr. Speaker, Mr. Vice President, Members of Co...

    
    df_president_speeches = pd.DataFrame(new_data)
    #                 prez                                               text
    #    GeorgeWashington  Fellow-Citizens of the Senate and House of Rep...
    #1           JohnAdams  Gentlemen of the Senate and Gentlemen of the H...
    #2     ThomasJefferson  Fellow Citizens of the Senate and House of Rep...
    #3        JamesMadison  Fellow-Citizens of the Senate and House of Rep...
    #4         JamesMonroe  Fellow-Citizens of the Senate and House of Rep...
    #5     JohnQuincyAdams  Fellow Citizens of the Senate and of the House...
    #6       AndrewJackson  Fellow Citizens of the Senate and of the House...
    #7      MartinVanBuren  Fellow-Citizens of the Senate and House of Rep...
    
    # After creating the df_president_speeches dataframe, I want to add in the number of speeches per president
    df_president_speeches['num_spchs'] = df_president_speeches['prez'].apply(speeches_per_president)    
    
    #print(f'Done, TIME: {timeit.default_timer() - start_time}')
    
        
    # now we're done reading in the speedches
    # I want to answer the following questions:
    # 1. Which presidents gave the longest speeches?
    # 2. Which presidents gave the shortest speeches?
    # 3. Which presidents used the most unique words?
    # 4. Which presidents used the fewest unique words?
    # 5. What were the most common words used in each speech?
    # 6. Given sentiment analysis, which presidents were the most positive? Negative?

    # But first, we have to clean the text! So we can actually analyze.
    # Let's CLEAN
    fd = nltk.FreqDist(' '.join(df_speeches['text']).split())
    print("")
    print(f"Uncleaned Common Words, TIME: {timeit.default_timer() - start_time}")
    print(fd.most_common(30))
    stops = stopwords.words('english')
    words = ' '.join(df_speeches['text']).lower().split()
    cleaned_words = [w for w in words if w not in set(stops)]
    cleaned_fd = nltk.FreqDist(cleaned_words)
    print()
    print(f"Cleaned Common Words, TIME: {timeit.default_timer() - start_time}")
    print(cleaned_fd.most_common(30))


    # Let's start with the first question: Which presidents gave the longest speeches?
    # Since this question is based on each speech individually, we'll use the 'speeches' list, the df_speeches dataframe
    fig, ax = plt.subplots()
    df_speeches['prez'].value_counts()[:10].plot.barh(title='# Speeches per President', ylabel='# Speeches')
    x_ticks = [0, 4, 8, 12, 16, 20]
    x_labels = ['0', '4', '8', '12', '16', '20']
    plt.xticks(ticks=x_ticks, labels=x_labels)
    #plt.show()
    plt.savefig('output/speeches_per_president_total.png')
    
    
    # That was a bit interesting to perform the overall frequency analysis of all presidents' speeches
    # But now let's look at the individual speeches and filter the stopwords
    print()
    print(f'Creating cleaned_text column in df_speeches, TIME: {timeit.default_timer() - start_time}')
    df_speeches['cleaned_text'] = df_speeches['text'].apply(filter_stops)
    print(df_speeches.head())
    #             prez  year                                               text                                       cleaned_text
    #0  AbrahamLincoln  1861  Fellow-Citizens of the Senate and House of Rep...  [fellow-citizens, senate, house, representativ...
    #1  AbrahamLincoln  1862  Fellow-Citizens of the Senate and House of Rep...  [fellow-citizens, senate, house, representativ...
    #2  AbrahamLincoln  1863  Fellow-Citizens of the Senate and House of Rep...  [fellow-citizens, senate, house, representativ...
    #3  AbrahamLincoln  1864  Fellow-Citizens of the Senate and House of Rep...  [fellow-citizens, senate, house, representativ...
    #4   AndrewJackson  1829  Fellow Citizens of the Senate and of the House...  [fellow, citizens, senate, house, representati...

    # Now let's do the same with the df_president_speeches dataframe
    print()
    print(f'Creating cleaned_text column in df_president_speeches, TIME: {timeit.default_timer() - start_time}')
    df_president_speeches['cleaned_text'] = df_president_speeches['text'].apply(filter_stops)
    print(df_president_speeches.head())
    df_speeches_sorted['cleaned_text'] = df_speeches_sorted['text'].apply(filter_stops)
    
    
    print()
    print('Testing word count function with test dataframe')
    print('BEFORE:')
    print(df_test)
    df_test['word_count'] = df_test['text'].apply(count_words)
    print('AFTER:')
    print(df_test)
    df_test['cleaned_text'] = df_test['text'].apply(filter_stops)
    print('Filtered stops...')
    df_test['cleaned_word_count'] = df_test['cleaned_text'].apply(count_words)
    print('AFTER CLEANING:')
    print(df_test)
    print('Done with test dataframe\n')
    
    # Now let's do the same with the real dataframes!
    print()
    print(f'Creating word_count and cleaned_word_count columns in df_speeches and df_president_speeches, TIME: {timeit.default_timer() - start_time}')
    df_speeches['word_count'] = df_speeches['text'].apply(count_words)
    df_speeches['cleaned_word_count'] = df_speeches['cleaned_text'].apply(count_words)
    df_president_speeches['word_count'] = df_president_speeches['text'].apply(count_words)
    df_president_speeches['cleaned_word_count'] = df_president_speeches['cleaned_text'].apply(count_words)
    df_president_speeches['avg_word_per_speech'] = (df_president_speeches['word_count'] / df_president_speeches['num_spchs']).astype(int)
    df_president_speeches['avg_cleaned_word_per_speech'] = (df_president_speeches['cleaned_word_count'] / df_president_speeches['num_spchs']).astype(int)
    df_president_speeches['word_substance'] = (df_president_speeches['avg_cleaned_word_per_speech'] / df_president_speeches['avg_word_per_speech']).astype(float)
    
    print(df_president_speeches.loc[:, ['prez', 'num_spchs', 'word_count', 'avg_word_per_speech', 'cleaned_word_count', 'avg_cleaned_word_per_speech', 'word_substance']])
    
    # find the top 15 most common words and put them in a new 'topwords' feature
    df_president_speeches['top_words'] = df_president_speeches['cleaned_text'].apply(lambda x: nltk.FreqDist(x).most_common(15))
    print()
    print('Top Words by President')
    print(df_president_speeches.loc[:, ['prez', 'top_words']])
    
    
    #df_sorted_speech_substance = df_president_speeches.sort_values(by='word_substance', ascending=False)
    
    # We definitely have enough information now to answer questions about word substance and word frequency among presidents
    # It seems right now that we only need the df_president_speeches dataframe for our analysis. We'll see if that changes.
    # I'll first answer the question of the longest speeches and shortest speeches, and most substantive and least substantive speeches
    fig0, ax0 = plt.subplots(figsize=(6,6))
    ax0.barh(df_president_speeches['prez'], df_president_speeches['avg_word_per_speech'], color='tab:blue')
    ax0.set(xlabel='Avg. Words per Speech', ylabel='President', title='Avg. Words per Speech by President')
    plt.tight_layout()
    plot0_filename = 'output/Avg-Words-per-Speech-by-President-CHRONO.png'
    fig0.savefig(plot0_filename, dpi=dpi)
    
    fig1, ax1 = plt.subplots(figsize=(6,6))
    df_sorted_word_per_speech = df_president_speeches.sort_values(by='avg_word_per_speech', ascending=False)
    ax1.barh(df_sorted_word_per_speech['prez'], df_sorted_word_per_speech['avg_word_per_speech'], color='tab:blue')
    ax1.set(xlabel='Avg. Words per Speech', ylabel='President', title='Avg. Words per Speech by President')
    plt.tight_layout()
    plot1_filename = 'output/Avg-Words-per-Speech-by-President-SORTED.png'
    fig1.savefig(plot1_filename, dpi=dpi)
    
    # We're done charting the word count per speech, now let's chart the word substance per speech
    fig2, ax2 = plt.subplots(figsize=(6,6))
    df_sorted_word_substance = df_president_speeches.sort_values(by='word_substance', ascending=False)
    ax2.barh(df_sorted_word_substance['prez'], df_sorted_word_substance['word_substance'], color='tab:blue')
    ax2.set(xlabel='Cleaned Word/Total Word Ratio', ylabel='President', title='Speech Substance by President')
    plt.tight_layout()
    plot2_filename = 'output/Speech-Substance-by-President-SORTED.png'
    fig2.savefig(plot2_filename, dpi=dpi)
    
    fig3, ax3 = plt.subplots(figsize=(6,6))
    ax3.barh(df_president_speeches['prez'], df_president_speeches['word_substance'], color='tab:blue')
    ax3.set(xlabel='Cleaned Word/Total Word Ratio', ylabel='President', title='Speech Substance by President')
    plt.tight_layout()
    plot3_filename = 'output/Speech-Substance-by-President-CHRONO.png'
    fig3.savefig(plot3_filename, dpi=dpi)
    
    print()
    print(f'Created initial word count / word frequency figures, TIME: {timeit.default_timer() - start_time}')
    
    # NOW Let's start delving into sentiment analysis
    #analyzer = SentimentIntensityAnalyzer()
    #df_president_speeches['sentiment_score'] = df_president_speeches['cleaned_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])    
    #df_speeches['sentiment_score'] = df_speeches['cleaned_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    #print(df_president_speeches.loc[:, ['prez', 'num_spchs', 'word_count', 'avg_word_per_speech', 'cleaned_word_count', 'avg_cleaned_word_per_speech', 'word_substance', 'sentiment_score']])
    sentiment_df = pd.read_csv('data/AFINN-en-165.txt', sep='\t', names=['word', 'score'], index_col='word')
    sentiment_dict = sentiment_df.to_dict()['score']
    
    df_president_speeches['sentiment_score'] = df_president_speeches['cleaned_text'].apply(sentiment_analysis)
    df_speeches_sorted['sentiment_score'] = df_speeches_sorted['cleaned_text'].apply(sentiment_analysis)
    
    print()
    print(df_president_speeches.loc[:, ['prez', 'num_spchs', 'word_count', 'avg_word_per_speech', 'cleaned_word_count', 'avg_cleaned_word_per_speech', 'word_substance', 'sentiment_score']])
    df_sorted_sentiment_01 = df_president_speeches.sort_values(by='sentiment_score', ascending=False)
    print()
    print(f'Printing sorted sentiment scores, TIME: {timeit.default_timer() - start_time}')
    print(df_sorted_sentiment_01.loc[:, ['prez', 'num_spchs', 'word_count', 'avg_word_per_speech', 'cleaned_word_count', 'avg_cleaned_word_per_speech', 'word_substance', 'sentiment_score']])
    df_sorted_sentiment_02 = df_president_speeches.sort_values(by='sentiment_score', ascending=True)
    print()
    print(f'Printing sorted sentiment scores, TIME: {timeit.default_timer() - start_time}')
    print(df_sorted_sentiment_02.loc[:, ['prez', 'num_spchs', 'word_count', 'avg_word_per_speech', 'cleaned_word_count', 'avg_cleaned_word_per_speech', 'word_substance', 'sentiment_score']])
    
    fig4, ax4 = plt.subplots(figsize=(6,6))
    ax4.barh(df_sorted_sentiment_02['prez'], df_sorted_sentiment_02['sentiment_score'], color='tab:blue')
    ax4.set(xlabel='Sentiment Score by President', ylabel='President', title='Sentiment by President')
    plt.tight_layout()
    plot4_filename = 'output/Sentiment-by-President-Sorted.png'
    fig4.savefig(plot4_filename, dpi=dpi)
    
    fig5, ax5 = plt.subplots(figsize=(6,6))
    ax5.barh(df_speeches_sorted['prez'], df_speeches_sorted['sentiment_score'], color='tab:blue')
    ax5.set(xlabel='Sentiment Score by President', ylabel='President', title='Sentiment by President')
    plt.tight_layout()
    plot5_filename = 'output/Sentiment-by-President.png'
    fig5.savefig(plot5_filename, dpi=dpi)
    
    fig6, ax6 = plt.subplots(figsize=(6,6))
    ax6.barh(df_speeches_sorted['year'], df_speeches_sorted['sentiment_score'], color='tab:blue')
    ax6.set(xlabel='Sentiment Score by Year', ylabel='President', title='Sentiment by President')
    plt.tight_layout()
    plot6_filename = 'output/Sentiment-by-Year.png'
    fig6.savefig(plot6_filename, dpi=dpi)
    
    # Create a bar chart of the top 15 most common words from GeorgeWashington, the first row of the dataframe
    
    fig7, ax7 = plt.subplots(figsize=(6,6))
    words_freq = df_president_speeches['top_words'][0]
    words = [word for word, freq in words_freq]
    freqs = [freq for word, freq in words_freq]
    
    ax7.barh(words, freqs, color='tab:blue')
    ax7.set(xlabel='Top Words by President Washington', ylabel='President', title='Top Words by President')
    plt.tight_layout()
    plot7_filename = 'output/Top-Words-by-President-01-GeorgeWashington.png'
    fig7.savefig(plot7_filename, dpi=dpi)
    
    # Now create the same chart but iterate through all presidents
    fig8, ax8 = plt.subplots(figsize=(6,6))
    for i in range(len(df_president_speeches)):
        print(f"Creating chart for president {i}: {df_president_speeches['prez'][i]}...")
        words_freq = df_president_speeches['top_words'][i]
        words = [word for word, freq in words_freq]
        freqs = [freq for word, freq in words_freq]
        ax8.barh(words[0:15], freqs[0:15], color='tab:blue')
        ylab=f"Top Words by President {df_president_speeches['prez'][i]}"
        ax8.set(xlabel='Mentions', ylabel=ylab, title=ylab)
        plt.tight_layout()
        plot8_filename = f"output/Top-Words-by-President-{i}-{df_president_speeches['prez'][i]}.png"
        fig8.savefig(plot8_filename, dpi=dpi)
    
    
    print()
    print(f'DONE, TOTAL TIME: {timeit.default_timer() - start_time} seconds')
    