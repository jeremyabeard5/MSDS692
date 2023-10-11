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
# ASSUMPTIONS
# 1. Each speech is one single text file, contained in a 'sotu_firstname' directory, within the 'data' directory
# 2. Each text file is formatted as '[FirstName][LastName]_[Year].txt' for example 'AbrahamLincoln_1861.txt'
###############################################

# 0-step to any project: time it
import timeit
start_time = timeit.default_timer()

# First step to any project: hello world :)
print("Hello World!")



# Next step, import the libraries we'll need
import os
import subprocess
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
#subprocess.run(["pip", "install", "-U", "pip"])
#subprocess.run(["pip", "install", "-U", "pillow"])
#subprocess.run(["pip", "install", "-U", "Pillow"])
#subprocess.run(["pip", "install", "-U", "wordcloud"])
from wordcloud import WordCloud

# I had a LOT of trouble getting textblob to import correctly
# I ended up having to install it within the actual script, below
#subprocess.run(["pip", "install", "-U", "textblob"])
#subprocess.run(["python", "-m", "textblob.download_corpora"])
from textblob import TextBlob
#import re
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Next steps, define paths, directories
data_dir = "data"
speeches_dir = "sotu_firstname" # the '_firstname' directory has been modified so all filenames include first and last name (this took some time :) )

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

def read_files():
    file_path = os.path.join(data_dir, speeches_dir)
    files = os.listdir(file_path)
    # Iterate over files in directory with format "president_year.txt"
    # last line of this for-loop reads in the files and puts them into the 'speeches' list
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
        

# Create a function to read in the speeches and filter the stopwords
# This function also removes punctuation and numerical characters
def clean_words(tex):
    stops = stopwords.words('english')
    
    # remove all symbols and punctuation
    chars_to_replace = '^&!@#$%*()_+=-`~[]{}|\\;:\'",.<>/?'
    replacement_char = ' '
    tex = tex.translate(str.maketrans(chars_to_replace, replacement_char * len(chars_to_replace)))
    tex = tex.replace('\n', ' ')
    
    # remove all numerical characters from the text
    tex = ''.join([i for i in tex if not i.isdigit()])    
    
    # make all words lowercase and split them into a list without stopwords
    words = tex.lower().split()
    filtered_words = [w for w in words if w not in set(stops)]
    s = pd.Series(filtered_words)
    # not really sure if I need to include the next line (probably not), but I kept it :P
    s_cleaned = s.str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)
    
    # remove all words that are less than 3 characters
    s_cleaned = s_cleaned[s_cleaned.str.len() > 2]
    
    # remove all extraneous spaces
    s_cleaned = s_cleaned.str.strip()
    
    # lemmatize!
    lemmatizer = WordNetLemmatizer()
    s_cleaned = s_cleaned.apply(lambda x: lemmatizer.lemmatize(x))
    
    # convert the series back to a list after lemmatizing
    filtered_words = s_cleaned.tolist()
    
    return filtered_words

# Create a function to count the words of a string of text
def count_words(tex):
    if type(tex) == list:
        return len(tex)
    else:
        words = tex.lower().split()
        return len(words)
    
# Create a function that finds the amount of speeches per president, based upon the number of counts of his/her name in the 'prez' column
def speeches_per_president(tex):
    return df_speeches['prez'].value_counts()[tex]

# define sentiment_analysis function to take in a text and return the sentiment score
sentiment_df = pd.read_csv('data/AFINN-en-165.txt', sep='\t', names=['word', 'score'], index_col='word')
sentiment_dict = sentiment_df.to_dict()['score']
def sentiment_analysis(tex):
    this_speechs_sentiments = []
    for word in tex:
        if word in sentiment_dict.keys():
            this_speechs_sentiments.append(sentiment_dict[word])
        else:
            this_speechs_sentiments.append(0)
    return (sum(this_speechs_sentiments) / len(this_speechs_sentiments))

# find sentiment analysis another method
def get_sentiment(tex):
    return TextBlob(tex).sentiment

# define find_grams function to take in a number and a text and return the top 15 most common n-grams
def find_grams(gram_num, tex):
    top_grams_count = 15
    bigrams = []
    trigrams = []
    if gram_num == 2:
        return nltk.FreqDist(nltk.bigrams(tex)).most_common(top_grams_count)
    elif gram_num == 3:
        return nltk.FreqDist(nltk.trigrams(tex)).most_common(top_grams_count)
    else:
        return None        

# Main function
if __name__ == "__main__":
    ################################ READ IN FILES, CREATE DATAFRAME ################################
    read_files()
    print(f'Done reading in speeches, TIME: {timeit.default_timer() - start_time}')
    
    # Create dataframe for original speech dataset
    df_speeches = pd.DataFrame(speeches)
    #print(df_speeches.head())
    # Now we can effectively forget about the original 'speeches' list, because it's more useful to have them sorted by year
    
    # Now that we have all the speeches, let's consolidate them by president
    # First step is to sort the 'speeches' list by year. This took a handful of syntactical tries :)
    sorted_list = sorted(speeches, key=lambda x: x['year'])
    # print(sorted_list)  
    df_speeches_sorted = pd.DataFrame(sorted_list)  

    # Now we want to create the dataframe that has the speeches consolidated by president
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
    df_president_speeches = pd.DataFrame(new_data)
    
    # NOW we have created THREE dataframes: 
    #   One for the speeches as-is, alphabetized (we won't really use this one)
    #   One for the speeches as-is, ordered chronologically
    #   One for the speeches joined per president, ordered chronologically

    # After creating the df_president_speeches dataframe, I want to add in the number of speeches per president
    df_president_speeches['num_spchs'] = df_president_speeches['prez'].apply(speeches_per_president)    
    
    # I also want to add in the Year of Presidency, so we can track the sentiment throughout a president's presidency
    # I'll do this by creating a new column in the df_president_speeches dataframe
    df_speeches['year_of_prez'] = df_speeches.groupby('prez').cumcount() + 1
    df_speeches_sorted['year_of_prez'] = df_speeches_sorted.groupby('prez').cumcount() + 1
    
    print()
    print('DF_SPEECHES COLUMNS AND HEAD')
    print(df_speeches.columns)
    print(df_speeches.head())
    
    print()
    print('DF_SPEECHES_SORTED COLUMNS AND HEAD')
    print(df_speeches_sorted.columns)
    print(df_speeches_sorted.head())
    
    print()
    print('DF_PRESIDENT_SPEECHES COLUMNS AND HEAD')
    print(df_president_speeches.columns)
    print(df_president_speeches.head())
    
    ################################ DONE READ IN FILES, CREATE DATAFRAME ################################
        
    # now we're done reading in the speedches
    # I want to answer the following questions:
    # 1. Which presidents gave the longest speeches?
    # 2. Which presidents gave the shortest speeches?
    # 3. Which presidents used the most unique words?
    # 4. Which presidents used the fewest unique words?
    # 5. What were the most common words used in each speech?
    # 6. Given sentiment analysis, which presidents were the most positive? Negative?

    ################################ CLEAN DATA ################################

    # But first, we have to clean the text! So we can actually analyze.
    # Let's CLEAN
    
    # The following is just a test demonstration of the nltk FreqDist function on the entire speeches as a whole
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
    
    # Create an example wordcloud for my title page :)
    fig, ax = plt.subplots(figsize=(6,6))
    cloudtex = ' '.join(cleaned_words)
    wordcloud = WordCloud(width=600, height=400, font_path='C:\\Windows\\WinSxS\\amd64_microsoft-windows-font-truetype-verdana_31bf3856ad364e35_10.0.22621.1_none_200eeed3f2ec147f\\verdana.ttf').generate(cloudtex)
    plt.imshow(wordcloud, interpolation='bilinear') 
    plt.axis("off")
    plt.tight_layout()
    plt.savefig('output/example_wordcloud.png', dpi=dpi)
    

    # This is another test chart showing the amount of speeches each president gave
    #fig, ax = plt.subplots()
    #df_speeches['prez'].value_counts()[:10].plot.barh(title='# Speeches per President', ylabel='# Speeches')
    #x_ticks = [0, 4, 8, 12, 16, 20]
    #x_labels = ['0', '4', '8', '12', '16', '20']
    #plt.xticks(ticks=x_ticks, labels=x_labels)
    ##plt.show()
    #plt.savefig('output/speeches_per_president_total.png')
    
    
    # That was a bit interesting to perform the overall frequency analysis of all presidents' speeches
    # But now let's look at the individual speeches and filter the stopwords
    print()
    print(f'Creating cleaned_text column in df_speeches, TIME: {timeit.default_timer() - start_time}')
    df_speeches['cleaned_text'] = df_speeches['text'].apply(clean_words)
    print(df_speeches.head())
    
    # Now let's do same with df_speeches_sorted
    print()
    print(f'df_speeches_sorted Cleaned Text, TIME: {timeit.default_timer() - start_time}')
    df_speeches_sorted['cleaned_text'] = df_speeches_sorted['text'].apply(clean_words)
    print(df_speeches_sorted.head())

    # Now let's do the same with the df_president_speeches dataframe
    print()
    print(f'df_president_speeches Cleaned Text, TIME: {timeit.default_timer() - start_time}')
    df_president_speeches['cleaned_text'] = df_president_speeches['text'].apply(clean_words)
    print(df_president_speeches.head())
    
    fig, ax = plt.subplots(figsize=(6,6))
    filtered_df = df_president_speeches[df_president_speeches['prez'] == 'DonaldTrump']
    all_words = ' '.join([' '.join(words) for words in filtered_df['cleaned_text']])
    wordcloud = WordCloud(width=600, height=400, font_path='C:\\Windows\\WinSxS\\amd64_microsoft-windows-font-truetype-verdana_31bf3856ad364e35_10.0.22621.1_none_200eeed3f2ec147f\\verdana.ttf').generate(all_words)
    plt.imshow(wordcloud, interpolation='bilinear') 
    plt.axis("off")
    plt.tight_layout()
    plt.savefig('output/trump_wordcloud.png', dpi=dpi)
    
    fig, ax = plt.subplots(figsize=(6,6))
    filtered_df = df_president_speeches[df_president_speeches['prez'] == 'GeorgeWashington']
    all_words = ' '.join([' '.join(words) for words in filtered_df['cleaned_text']])
    wordcloud = WordCloud(width=600, height=400, font_path='C:\\Windows\\WinSxS\\amd64_microsoft-windows-font-truetype-verdana_31bf3856ad364e35_10.0.22621.1_none_200eeed3f2ec147f\\verdana.ttf').generate(all_words)
    plt.imshow(wordcloud, interpolation='bilinear') 
    plt.axis("off")
    plt.tight_layout()
    plt.savefig('output/washington_wordcloud.png', dpi=dpi)
    
    ################################ WORD COUNT, WORD SUBSTANCE ANALYSIS ################################
    
    # Below is some legacy code that tests the cleaning function we created earlier
    #print()
    #print('Testing word count function with test dataframe')
    #print('BEFORE:')
    #print(df_test)
    #df_test['word_count'] = df_test['text'].apply(count_words)
    #print('AFTER:')
    #print(df_test)
    #df_test['cleaned_text'] = df_test['text'].apply(clean_words)
    #print('Filtered stops...')
    #df_test['cleaned_word_count'] = df_test['cleaned_text'].apply(count_words)
    #print('AFTER CLEANING:')
    #print(df_test)
    #print('Done with test dataframe\n')
    
    # Now let's do the same with the real dataframes!
    print()
    print(f'Creating word_count and cleaned_word_count columns in df_speeches and df_president_speeches, TIME: {timeit.default_timer() - start_time}')
    df_speeches['word_count'] = df_speeches['text'].apply(count_words)
    df_speeches['cleaned_word_count'] = df_speeches['cleaned_text'].apply(count_words)
    
    df_speeches_sorted['word_count'] = df_speeches_sorted['text'].apply(count_words)
    df_speeches_sorted['cleaned_word_count'] = df_speeches_sorted['cleaned_text'].apply(count_words)
    df_speeches_sorted['word_substance'] = (df_speeches_sorted['cleaned_word_count'] / df_speeches_sorted['word_count']).astype(float)
    
    df_president_speeches['word_count'] = df_president_speeches['text'].apply(count_words)
    df_president_speeches['cleaned_word_count'] = df_president_speeches['cleaned_text'].apply(count_words)
    df_president_speeches['avg_word_per_speech'] = (df_president_speeches['word_count'] / df_president_speeches['num_spchs']).astype(int)
    df_president_speeches['avg_cleaned_word_per_speech'] = (df_president_speeches['cleaned_word_count'] / df_president_speeches['num_spchs']).astype(int)
    df_president_speeches['word_substance'] = (df_president_speeches['avg_cleaned_word_per_speech'] / df_president_speeches['avg_word_per_speech']).astype(float)
    
    print(df_president_speeches.loc[:, ['prez', 'num_spchs', 'word_count', 'avg_word_per_speech', 'cleaned_word_count', 'avg_cleaned_word_per_speech', 'word_substance']])
    
    ################################ COMMON 1-GRAM, 2-GRAM, 3-GRAM ANALYSIS (COMMON WORDS) ################################
    
    # find the top 15 most common words and put them in a new 'topwords' feature
    top_words_count = 20 # how many you want to find
    df_president_speeches['top_words'] = df_president_speeches['cleaned_text'].apply(lambda x: nltk.FreqDist(x).most_common(top_words_count))
    print()
    print('Top Words by President')
    print(df_president_speeches.loc[:, ['prez', 'top_words']])
    
    # Do the same with the per-year dataset
    df_speeches_sorted['top_words'] = df_speeches_sorted['cleaned_text'].apply(lambda x: nltk.FreqDist(x).most_common(top_words_count))
    print()
    print('Top Words by President')
    print(df_speeches_sorted.loc[:, ['prez', 'top_words']])
    
    # Find most common 2-grams and 3-grams
    df_president_speeches['top_2grams'] = df_president_speeches['cleaned_text'].apply(lambda x: find_grams(2, x))
    df_president_speeches['top_3grams'] = df_president_speeches['cleaned_text'].apply(lambda x: find_grams(3, x))
    df_speeches_sorted['top_2grams'] = df_speeches_sorted['cleaned_text'].apply(lambda x: find_grams(2, x))
    df_speeches_sorted['top_3grams'] = df_speeches_sorted['cleaned_text'].apply(lambda x: find_grams(3, x))
    
    ################################ VOCABULARY / UNIQUE WORD COUNT ANALYSIS ################################
    
    # count the number of unique words per president (a set automatically has only unique elements)
    df_president_speeches['unique_words_total'] = df_president_speeches['cleaned_text'].apply(lambda x: len(set(x)))
    df_speeches_sorted['unique_words'] = df_speeches_sorted['cleaned_text'].apply(lambda x: len(set(x)))
    
    # now create unique_words_per_speech
    df_president_speeches['unique_words_per_speech'] = df_president_speeches['unique_words_total'] / df_president_speeches['num_spchs']
    print()
    print('Unique Words by President')
    print(df_president_speeches.loc[:, ['prez', 'unique_words_total', 'unique_words_per_speech']])
    print()
    print('Unique Words by Year')
    print(df_speeches_sorted.loc[:, ['prez', 'unique_words']])
    
    print()
    print(f'Done creating top words, top 2-grams, top 3-grams, and unique words, TIME: {timeit.default_timer() - start_time}')
    
    ################################ INITIAL PLOT CREATION ################################
    
    # We definitely have enough information now to answer questions about word substance and word frequency among presidents
    # It seems right now that we only need the df_president_speeches dataframe for our analysis. We'll see if that changes.
    # I'll first answer the question of the longest speeches and shortest speeches, and most substantive and least substantive speeches
    fig0, ax0 = plt.subplots(figsize=(6,6))
    ax0.barh(df_president_speeches['prez'], df_president_speeches['avg_word_per_speech'], color='tab:blue')
    ax0.set(xlabel='Avg. Words per Speech', ylabel='President', title='Avg. Words per Speech by President')
    plt.tight_layout()
    plot0_filename = 'output/Word-Count-by-President-CHRONO.png'
    fig0.savefig(plot0_filename, dpi=dpi)
    
    fig1, ax1 = plt.subplots(figsize=(6,6))
    df_sorted_word_per_speech = df_president_speeches.sort_values(by='avg_word_per_speech', ascending=False)
    ax1.barh(df_sorted_word_per_speech['prez'], df_sorted_word_per_speech['avg_word_per_speech'], color='tab:blue')
    ax1.set(xlabel='Avg. Words per Speech', ylabel='President', title='Avg. Words per Speech by President')
    ax1.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    plot1_filename = 'output/Word-Count-by-President-SORTED.png'
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
    
    fig4, ax4 = plt.subplots(figsize=(6,6))
    ax4.barh(df_speeches_sorted['year'], df_speeches_sorted['cleaned_word_count'], color='tab:blue')
    ax4.set(xlabel='Word Count', ylabel='Year', title='Word Count by Year')
    plt.tight_layout()
    plot4_filename = 'output/Word-Count-by-Year-CHRONO.png'
    fig4.savefig(plot4_filename, dpi=dpi)
    
    fig5, ax5 = plt.subplots(figsize=(6,6))
    df_speeches_sorted_word_count = df_speeches_sorted.sort_values(by='cleaned_word_count', ascending=False)
    ax5.barh(df_speeches_sorted_word_count['year'], df_speeches_sorted_word_count['cleaned_word_count'], color='tab:blue')
    ax5.set(xlabel='Word Count', ylabel='Year', title='Word Count by Year')
    plt.tight_layout()
    plot5_filename = 'output/Word-Count-by-Year-SORTED.png'
    fig5.savefig(plot5_filename, dpi=dpi)
    
    fig6, ax6 = plt.subplots(figsize=(6,6))
    ax6.barh(df_speeches_sorted['year'], df_speeches_sorted['word_substance'], color='tab:blue')
    ax6.set(xlabel='Cleaned Word/Total Word Ratio', ylabel='Year', title='Speech Substance by Year')
    plt.tight_layout()
    plot6_filename = 'output/Speech-Substance-by-Year-CHRONO.png'
    fig6.savefig(plot6_filename, dpi=dpi)
    
    fig7, ax7 = plt.subplots(figsize=(6,6))
    df_speeches_sorted_word_substance = df_speeches_sorted.sort_values(by='word_substance', ascending=False)
    ax7.barh(df_speeches_sorted_word_substance['year'], df_speeches_sorted_word_substance['word_substance'], color='tab:blue')
    ax7.set(xlabel='Cleaned Word/Total Word Ratio', ylabel='Year', title='Speech Substance by Year')
    plt.tight_layout()
    plot7_filename = 'output/Speech-Substance-by-Year-SORTED.png'
    fig7.savefig(plot7_filename, dpi=dpi)
    
    print()
    print(f'Created initial word count / word frequency figures, TIME: {timeit.default_timer() - start_time}')
    
    ################################ SENTIMENT ANALYSIS ################################
    
    # NOW Let's start delving into sentiment analysis, using the sentiment_analysis function we created earlier
    df_president_speeches['sentiment_score'] = df_president_speeches['cleaned_text'].apply(sentiment_analysis)
    df_speeches_sorted['sentiment_score'] = df_speeches_sorted['cleaned_text'].apply(sentiment_analysis)
    
    # Now we'll plot the sentiment score
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
    
    fig8, ax8 = plt.subplots(figsize=(6,6))
    ax8.barh(df_sorted_sentiment_02['prez'], df_sorted_sentiment_02['sentiment_score'], color='tab:blue')
    ax8.set(xlabel='Sentiment Score by President', ylabel='President', title='Sentiment by President')
    plt.tight_layout()
    plot8_filename = 'output/Sentiment-by-President-SORTED.png'
    fig8.savefig(plot8_filename, dpi=dpi)
    
    fig8, ax8 = plt.subplots(figsize=(6,6))
    ax8.plot(df_sorted_sentiment_02['prez'], df_sorted_sentiment_02['sentiment_score'], color='tab:blue')
    ax8.set(xlabel='Sentiment Score by President', ylabel='President', title='Sentiment by President')
    plt.tight_layout()
    plot8_filename = 'output/Line-Sentiment-by-President-SORTED.png'
    fig8.savefig(plot8_filename, dpi=dpi)
    
    fig9, ax9 = plt.subplots(figsize=(6,6))
    ax9.barh(df_president_speeches['prez'], df_president_speeches['sentiment_score'], color='tab:blue')
    ax9.set(xlabel='Sentiment Score by President', ylabel='President', title='Sentiment by President')
    plt.tight_layout()
    plot9_filename = 'output/Sentiment-by-President-CHRONO.png'
    fig9.savefig(plot9_filename, dpi=dpi)
    
    fig9, ax9 = plt.subplots(figsize=(6,6))
    ax9.plot(df_president_speeches['prez'], df_president_speeches['sentiment_score'], color='tab:blue')
    ax9.set(xlabel='Sentiment Score by President', ylabel='President', title='Sentiment by President')
    plt.tight_layout()
    plot9_filename = 'output/Line-Sentiment-by-President-CHRONO.png'
    fig9.savefig(plot9_filename, dpi=dpi)
    
    fig10, ax10 = plt.subplots(figsize=(6,6))
    df_speeches_sorted_sentiment = df_speeches_sorted.sort_values(by='sentiment_score', ascending=False)
    ax10.barh(df_speeches_sorted_sentiment['year'], df_speeches_sorted_sentiment['sentiment_score'], color='tab:blue')
    ax10.set(xlabel='Sentiment Score by Year', ylabel='Year', title='Sentiment by Year')
    plt.tight_layout()
    plot10_filename = 'output/Sentiment-by-Year-SORTED.png'
    fig10.savefig(plot10_filename, dpi=dpi)
    
    fig11, ax11 = plt.subplots(figsize=(6,6))
    ax11.barh(df_speeches_sorted['year'], df_speeches_sorted['sentiment_score'], color='tab:blue')
    ax11.set(xlabel='Sentiment Score by Year', ylabel='Year', title='Sentiment by Year')
    plt.tight_layout()
    plot11_filename = 'output/Sentiment-by-Year-CHRONO.png'
    fig11.savefig(plot11_filename, dpi=dpi)
    
    ################################ N-GRAM PLOTTING ################################
    
    # Now I iterate through the list of all presidents and plot the most common 1-grams, 2-grams, and 3-grams
    for i in range(len(df_president_speeches)):
        print(f"Creating 1-gram chart for president {i+1}: {df_president_speeches['prez'][i]}...")
        fig12, ax12 = plt.subplots(figsize=(6,6))
        words_freq = df_president_speeches['top_words'][i]
        words = [word for word, freq in words_freq]
        freqs = [freq for word, freq in words_freq]
        print(f"words are {words}")
        print(f"freqs are {freqs}")
        ax12.barh(words, freqs, color='tab:blue')
        ylab=f"Top Words by President {df_president_speeches['prez'][i]}"
        ax12.set(xlabel='Mentions', ylabel=ylab, title=ylab)
        plt.tight_layout()
        plot12_filename = f"output/Top-Words-by-President-{i+1}-{df_president_speeches['prez'][i]}.png"
        fig12.savefig(plot12_filename, dpi=dpi)
        print()
    
    # let's plot the 2-grams by president!
    for i in range(len(df_president_speeches)):
        print(f"Creating 2-gram chart PER PRESIDENT, {i+1}: {df_president_speeches['prez'][i]}...")
        fig13, ax13 = plt.subplots(figsize=(6,6))
        grams2_freq = df_president_speeches['top_2grams'][i]
        grams2 = [gram for gram, freq in grams2_freq]
        freqs = [freq for word, freq in grams2_freq]
        print(f"2grams are {grams2}")
        print(f"freqs are {freqs}")
        for j in range(len(grams2)):
            grams2[j] = ' '.join(grams2[j])
        ax13.barh(grams2, freqs, color='tab:blue')
        ylab=f"Top 2-Grams by President {df_president_speeches['prez'][i]}"
        ax13.set(xlabel='Mentions', ylabel=ylab, title=ylab)
        plt.tight_layout()
        plot13_filename = f"output/Top-2Grams-by-President-{i+1}-{df_president_speeches['prez'][i]}.png"
        fig13.savefig(plot13_filename, dpi=dpi)
        print()

    # let's plot the 3-grams by president!
    for i in range(len(df_president_speeches)):
        print(f"Creating 3-gram chart PER PRESIDENT {i+1}: {df_president_speeches['prez'][i]}...")
        fig14, ax14 = plt.subplots(figsize=(6,6))
        grams3_freq = df_president_speeches['top_3grams'][i]
        grams3 = [gram for gram, freq in grams3_freq]
        freqs = [freq for word, freq in grams3_freq]
        print(f"3grams are {grams3}")
        print(f"freqs are {freqs}")
        for j in range(len(grams2)):
            grams3[j] = ' '.join(grams3[j])
        ax14.barh(grams3, freqs, color='tab:blue')
        ylab=f"Top 3-Grams by President {df_president_speeches['prez'][i]}"
        ax14.set(xlabel='Mentions', ylabel=ylab, title=ylab)
        plt.tight_layout()
        plot14_filename = f"output/Top-3Grams-by-President-{i+1}-{df_president_speeches['prez'][i]}.png"
        fig14.savefig(plot14_filename, dpi=dpi)
        print()    
    
    # let's plot the 2-grams by year!
    for i in range(len(df_speeches_sorted)):
        print(f"Creating 2-gram chart PER YEAR, {i+1}: {df_speeches_sorted['prez'][i]}...")
        fig15, ax15 = plt.subplots(figsize=(6,6))
        grams2_freq = df_speeches_sorted['top_2grams'][i]
        grams2 = [gram for gram, freq in grams2_freq]
        freqs = [freq for word, freq in grams2_freq]
        print(f"2grams are {grams2}")
        print(f"freqs are {freqs}")
        for j in range(len(grams2)):
            grams2[j] = ' '.join(grams2[j])
        ax15.barh(grams2, freqs, color='tab:blue')
        ylab=f"Top 2-Grams by Year, {df_speeches_sorted['prez'][i]}, {df_speeches_sorted['year'][i]}"
        ax15.set(xlabel='Mentions', ylabel=ylab, title=ylab)
        plt.tight_layout()
        plot15_filename = f"output/Top-2Grams-by-Year-{df_speeches_sorted['year'][i]}-{df_speeches_sorted['prez'][i]}.png"
        fig15.savefig(plot15_filename, dpi=dpi)
        print()
    
    # let's plot the 3-grams by year!
    for i in range(len(df_speeches_sorted)):
        print(f"Creating 3-gram chart PER YEAR, {i+1}: {df_speeches_sorted['prez'][i]}...")
        fig16, ax16 = plt.subplots(figsize=(6,6))
        grams3_freq = df_speeches_sorted['top_3grams'][i]
        grams3 = [gram for gram, freq in grams3_freq]
        freqs = [freq for word, freq in grams3_freq]
        print(f"3grams are {grams3}")
        print(f"freqs are {freqs}")
        for j in range(len(grams2)):
            grams3[j] = ' '.join(grams3[j])
        ax16.barh(grams3, freqs, color='tab:blue')
        ylab=f"Top 3-Grams by Year, {df_speeches_sorted['prez'][i]}, {df_speeches_sorted['year'][i]}"
        ax16.set(xlabel='Mentions', ylabel=ylab, title=ylab)
        plt.tight_layout()
        plot16_filename = f"output/Top-3Grams-by-Year-{df_speeches_sorted['year'][i]}-{df_speeches_sorted['prez'][i]}.png"
        fig16.savefig(plot16_filename, dpi=dpi)
        print()    
    
    ################################ UNIQUE WORD PLOTTING ################################
    
    # plot the number of unique words per president, found ealier
    fig17, ax17 = plt.subplots(figsize=(6,6))
    ax17.barh(df_president_speeches['prez'], df_president_speeches['unique_words_per_speech'], color='tab:blue')
    ax17.set(xlabel='Average # Unique Words by President', ylabel='President', title='Avg. Unique Words by President (Chronological)')
    plt.tight_layout()
    plot17_filename = 'output/Unique-Words-by-President-CHRONO.png'
    fig17.savefig(plot17_filename, dpi=dpi)
    
    # plot the number of unique words per president, SORTED
    fig18, ax18 = plt.subplots(figsize=(6,6))
    df_sorted_unique_words = df_president_speeches.sort_values(by='unique_words_per_speech', ascending=False)
    ax18.barh(df_sorted_unique_words['prez'], df_sorted_unique_words['unique_words_per_speech'], color='tab:blue')
    ax18.set(xlabel='Unique Words by President', ylabel='President', title='Unique Words by President (Sorted)')
    plt.tight_layout()
    plot18_filename = 'output/Unique-Words-by-President-SORTED.png'
    fig18.savefig(plot18_filename, dpi=dpi)
    
    fig19, ax19 = plt.subplots(figsize=(6,6))
    #df_speeches_sorted_sentiment = df_speeches_sorted.sort_values(by='sentiment_score', ascending=False)
    ax19.plot(df_speeches_sorted_sentiment['year'], df_speeches_sorted_sentiment['sentiment_score'], color='tab:blue')
    ax19.set(xlabel='Sentiment Score by Year', ylabel='Year', title='Sentiment by Year')
    plt.tight_layout()
    plot19_filename = 'output/Line-Sentiment-by-Year-SORTED.png'
    fig19.savefig(plot19_filename, dpi=dpi)
    
    fig20, ax20 = plt.subplots(figsize=(6,6))
    ax20.plot(df_speeches_sorted['year'], df_speeches_sorted['sentiment_score'], color='tab:blue')
    ax20.set(xlabel='Sentiment Score by Year', ylabel='Year', title='Sentiment Score')
    plt.tight_layout()
    plot20_filename = 'output/Line-Sentiment-by-Year-CHRONO.png'
    fig20.savefig(plot20_filename, dpi=dpi)
    
    # Now to plot the sentiment over the course of a presidency
    fig21, ax21 = plt.subplots(figsize=(6,6))
    df_speeches_sorted.plot.scatter(x='year_of_prez', y='sentiment_score', c='DarkBlue', ax=ax21)
    ax21.set(xlabel='Sentiment Score by Year of Presidency', ylabel='Year of Presidency', title='Sentiment Score')
    plt.tight_layout()
    plot21_filename = 'output/Sentiment-by-Year-of-Presidency-CHRONO.png'
    fig21.savefig(plot21_filename, dpi=dpi)
    
    # I want to find the total average sentiment by year of presidency, and see if this tells us anything
    max_yop = df_speeches_sorted['year_of_prez'].max()
    min_yop = df_speeches_sorted['year_of_prez'].min()
    yop_values = []
    avg_sentiment_values = []
    for i in range(min_yop,9): #max_yop+1):
        avg_sentiment = df_speeches_sorted[df_speeches_sorted['year_of_prez'] == i]['sentiment_score'].mean()   
        print(f"Average sentiment for year of presidency {i}: {avg_sentiment}")
        yop_values.append(i)
        avg_sentiment_values.append(avg_sentiment)
        
    fig22, ax22 = plt.subplots(figsize=(6,6))
    ax22.plot(yop_values, avg_sentiment_values, color='tab:blue')
    ax22.set(xlabel='Year of Presidency', ylabel='Average Sentiment Score', title='Average Sentiment Score by Year of Presidency')
    plt.tight_layout()
    plot22_filename = 'output/Avg-Sentiment-by-Year-of-Presidency.png'
    fig22.savefig(plot22_filename, dpi=dpi)
    
    # I want to also find the sentiment a 2nd method, using the TextBlob library. My first approach was underwhelming
    df_speeches_sorted[['polarity', 'subjectivity']] = df_speeches_sorted['cleaned_text'].apply(lambda x: pd.Series(get_sentiment(' '.join(x))))
    df_president_speeches[['polarity', 'subjectivity']] = df_president_speeches['cleaned_text'].apply(lambda x: pd.Series(get_sentiment(' '.join(x))))
    
    fig23, ax23 = plt.subplots(figsize=(6,6))
    ax23.plot(df_speeches_sorted['year'], df_speeches_sorted['polarity'], color='tab:blue')
    ax23.set(xlabel='Year', ylabel='Polarity', title='Polarity by Year')
    plt.tight_layout()
    plot23_filename = 'output/Polarity-by-Year-CHRONO.png' #line
    fig23.savefig(plot23_filename, dpi=dpi)
    
    fig24, ax24 = plt.subplots(figsize=(6,6))
    ax24.plot(df_speeches_sorted['year'], df_speeches_sorted['subjectivity'], color='tab:blue')
    ax24.set(xlabel='Year', ylabel='Subjectivity', title='Subjectivity by Year')
    plt.tight_layout()
    plot24_filename = 'output/Subjectivity-by-Year-CHRONO.png' #line
    fig24.savefig(plot24_filename, dpi=dpi)
    
    fig25, ax25 = plt.subplots(figsize=(6,6))
    ax25.barh(df_president_speeches['prez'], df_president_speeches['polarity'], color='tab:blue')
    ax25.set(xlabel='President', ylabel='Polarity', title='Polarity by President')
    plt.tight_layout()
    plot25_filename = 'output/Polarity-by-President-CHRONO.png' #bar, or switch line axes
    fig25.savefig(plot25_filename, dpi=dpi)
    
    fig26, ax26 = plt.subplots(figsize=(6,6))
    ax26.plot(df_president_speeches['prez'], df_president_speeches['subjectivity'], color='tab:blue')
    ax26.set(xlabel='President', ylabel='Subjectivity', title='Subjectivity by President')
    plt.tight_layout()
    plot26_filename = 'output/Subjectivity-by-President-CHRONO.png'
    fig26.savefig(plot26_filename, dpi=dpi)
    
    fig27, ax27 = plt.subplots(figsize=(6,6))
    df_speeches_sorted_polarity = df_speeches_sorted.sort_values(by='polarity', ascending=False)  
    ax27.plot(df_speeches_sorted_polarity['year'], df_speeches_sorted_polarity['polarity'], color='tab:blue')
    ax27.set(xlabel='Year', ylabel='Polarity', title='Polarity by Year')
    plt.tight_layout()
    plot27_filename = 'output/Polarity-by-Year-SORTED.png'
    fig27.savefig(plot27_filename, dpi=dpi)
    
    fig28, ax28 = plt.subplots(figsize=(6,6))
    df_speeches_sorted_subjectivity = df_speeches_sorted.sort_values(by='subjectivity', ascending=False)
    ax28.plot(df_speeches_sorted_subjectivity['year'], df_speeches_sorted_subjectivity['subjectivity'], color='tab:blue')
    ax28.set(xlabel='Year', ylabel='Subjectivity', title='Subjectivity by Year')
    plt.tight_layout()
    plot28_filename = 'output/Subjectivity-by-Year-SORTED.png'
    fig28.savefig(plot28_filename, dpi=dpi)
    
    fig29, ax29 = plt.subplots(figsize=(6,6))
    df_president_speeches_polarity = df_president_speeches.sort_values(by='polarity', ascending=False)
    ax29.plot(df_president_speeches_polarity['prez'], df_president_speeches_polarity['polarity'], color='tab:blue')
    ax29.set(xlabel='President', ylabel='Polarity', title='Polarity by President')
    plt.tight_layout()
    plot29_filename = 'output/Polarity-by-President-SORTED.png'
    fig29.savefig(plot29_filename, dpi=dpi)
    
    fig30, ax30 = plt.subplots(figsize=(6,6))
    df_president_speeches_subjectivity = df_president_speeches.sort_values(by='subjectivity', ascending=False)
    ax30.plot(df_president_speeches_subjectivity['prez'], df_president_speeches_subjectivity['subjectivity'], color='tab:blue')
    ax30.set(xlabel='President', ylabel='Subjectivity', title='Subjectivity by President')
    plt.tight_layout()
    plot30_filename = 'output/Subjectivity-by-President-SORTED.png'
    fig30.savefig(plot30_filename, dpi=dpi)
    
    yop_values2 = []
    avg_polarity_values = []
    avg_subjectivity_values = []
    for i in range(min_yop,9): #max_yop+1):
        avg_polarity = df_speeches_sorted[df_speeches_sorted['year_of_prez'] == i]['polarity'].mean()   
        avg_subjectivity = df_speeches_sorted[df_speeches_sorted['year_of_prez'] == i]['subjectivity'].mean()   
        print(f"Average polarity, subjectivity for year of presidency {i}: {avg_polarity}, {avg_subjectivity}")
        yop_values2.append(i)
        avg_polarity_values.append(avg_polarity)
        avg_subjectivity_values.append(avg_subjectivity)
        
    fig31, ax31 = plt.subplots(figsize=(6,6))
    ax31.plot(yop_values2, avg_polarity_values, color='tab:blue')
    ax31.set(xlabel='Year of Presidency', ylabel='Average Polarity', title='Average Polarity by Year of Presidency')
    plt.tight_layout()
    plot31_filename = 'output/Avg-Polarity-by-Year-of-Presidency.png'
    fig31.savefig(plot31_filename, dpi=dpi)
    
    fig32, ax32 = plt.subplots(figsize=(6,6))
    ax32.plot(yop_values2, avg_subjectivity_values, color='tab:blue')
    ax32.set(xlabel='Year of Presidency', ylabel='Average Subjectivity', title='Average Subjectivity by Year of Presidency')
    plt.tight_layout()
    plot32_filename = 'output/Avg-Subjectivity-by-Year-of-Presidency.png'
    fig32.savefig(plot32_filename, dpi=dpi)
    
    print()
    print('DF_SPEECHES COLUMNS AND HEAD')
    print(df_speeches.columns)
    print(df_speeches.head())
    
    print()
    print('DF_SPEECHES_SORTED COLUMNS AND HEAD')
    print(df_speeches_sorted.columns)
    print(df_speeches_sorted.head())
    
    print()
    print('DF_PRESIDENT_SPEECHES COLUMNS AND HEAD')
    print(df_president_speeches.columns)
    print(df_president_speeches.head())
    
    print()
    print(f'DONE, TOTAL TIME: {timeit.default_timer() - start_time} seconds')
    