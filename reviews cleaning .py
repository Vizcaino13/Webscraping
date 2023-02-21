#!/usr/bin/env python
# coding: utf-8

# In[1]:


conda install pandas


# In[2]:


import pandas as pd


# In[3]:


import os


# In[4]:


path = os.path.join("data","BA_reviews.csv")
if os.path.isfile(path):
    data = pd.read_csv(path)
else:
    print(f"file {path} does not exist")


# In[5]:


data = pd.read_csv("data/BA_reviews.csv")


# In[6]:


print(data)


# In[7]:


data.describe(include='all')


# In[8]:


data[data['reviews'].str.contains("Thailand")]


# In[9]:


data.rename(columns={'Unnamed':'trip_verification'}, inplace=True)


# In[10]:


data = pd.read_csv("data/BA_reviews.csv")
data.head()  


# In[11]:


print(data)


# In[12]:


pip install country_list


# In[13]:


from country_list import available_languages, countries_for_language

for language in available_languages():
    print(language)
    break


# In[14]:


countries = dict(countries_for_language('en'))
print(countries['TH'])


# In[15]:


# Get a list of country names for a given language (in this case, English)
country_list = [country[1] for country in countries_for_language('en')]

# Create a function to extract country names
def extract_country(reviews):
    # Loop through the list of countries and check if the review contains a country name
    for country in country_list:
        if country in reviews:
            return country
    return None


data['countries'] = data['reviews'].apply(extract_country)

# Show the updated data frame
print(data)


# In[16]:


print(data.tail(5))


# In[17]:


data[data['reviews'].str.contains("London")]


# In[18]:


data["trip verification"] = data.reviews.str[:15]
data.tail()


# In[19]:


data['reviews'] = data['reviews'].str.replace('✅ Trip Verified', '')


# In[20]:


data.tail()


# In[21]:


data['reviews'] = data['reviews'].str.replace('Not Verified', '')


# In[22]:


data.tail()


# In[23]:


data.to_csv("modified_data.csv", index=False)




#questions for data: mention all of the times the British airlines is mentioned
# Count up all of the positive reviews with words such as good great excellent    


# In[24]:


print(data)


# In[25]:


pip install gensim


# In[26]:


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora


# In[27]:



# Define a function to preprocess the review data
def preprocess(reviews):
    return [word for word in simple_preprocess(reviews) if word not in STOPWORDS]

# Preprocess the reviews data
processed_reviews = data['reviews'].map(preprocess)

# Create a dictionary from the processed reviews data
dictionary = corpora.Dictionary(processed_reviews)

# Create a bag-of-words representation of the processed reviews data
bow_corpus = [dictionary.doc2bow(reviews) for reviews in processed_reviews]

# Train the topic modeling algorithm using the bag-of-words corpus
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

# Print the top 10 keywords for each of the 10 topics generated
for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic))


# In[28]:


pip install wordcloud


# In[29]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

reviews = data['reviews']
reviews_text = " ".join(review for review in reviews)
wordcloud = WordCloud(width=800, height=800, min_font_size=10).generate(reviews_text)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()


# In[30]:


lda_model.save('model_file.lda')


# In[31]:


print(data)


# In[32]:


from collections import Counter
import pandas as pd

# Assuming that 'data' is a DataFrame with a column named 'text' containing text data
words = Counter(" ".join(data['reviews']).split())
most_common_words = words.most_common(10)  # Returns the 10 most common words
print(most_common_words)


# In[ ]:





# In[33]:


# Assuming 'model' is the trained topic model
num_words = 10  # Number of most common words to print for each topic
topics = lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)

for topic in topics:
    print(f"Topic {topic[0]}: ")
    for word, weight in topic[1]:
        print(f"\t{word} ({weight:.2f})")


# In[34]:




# Assuming 'model' is the trained topic model
num_words = 10  # Number of most common words to print for each topic
topics = lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)

fig, ax = plt.subplots(figsize=(10, 8))

for topic in topics:
    words = [word for word, weight in topic[1]]
    weights = [weight for word, weight in topic[1]]
    ax.bar(words, weights, alpha=0.8, label=f"Topic {topic[0]}")

ax.set_xlabel("Word")
ax.set_ylabel("Weight")
ax.set_title("Top Words by Topic")
ax.legend()
plt.xticks(rotation=45)
plt.show()


# In[35]:


# Extract the value counts for the "trip verification" column
counts = data["trip verification"].value_counts()

fig, ax = plt.subplots(figsize=(6, 6))

ax.bar(counts.index, counts.values, alpha=0.8)

ax.set_xlabel("Verification Status")
ax.set_ylabel("Count")
ax.set_title("Trip Verification Counts")
plt.show()


# In[36]:


#if the trip is verfied is is more likley to be a negative review 


# In[37]:


negative_words = ["bad", "not good"]
negative_reviews = data.loc[(data["trip verification"] != "trip verified") & (data["reviews"].str.contains("|".join(negative_words))), "reviews"]
for review in negative_reviews:
    print(review)
    break


# In[38]:


import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Tokenize the words in the "review" column and convert to lowercase
all_words = [word.lower() for review in data["reviews"] for word in word_tokenize(review)]

# Define a set of negative words to search for
negative_words = {"bad", "not good", "poor", "terrible", "disappointing", "disappointed", "awful", "horrible", "miserable", "stupid", "sad","nothappy"}

# Count the frequency of each negative word in the "review" column
negative_word_counts = Counter([word for word in all_words if word in negative_words])

# Create a list of negative words and their frequencies, sorted in descending order by frequency
negative_word_list = sorted(negative_word_counts.items(), key=lambda x: x[1], reverse=True)

# Print the list of negative words and their frequencies
for word, count in negative_word_list:
    print(f"{word}: {count}")


# In[39]:


from nltk.tokenize import word_tokenize
from collections import Counter

# Define a set of negative words to search for
negative_words = {"bad", "not good", "poor", "terrible", "disappointing", "disappointed", "awful", "horrible", "miserable", "stupid", "sad","nothappy"}

# Define a function to count the negative words in a review
def count_negative_words(review):
    # Tokenize the words in the review and convert to lowercase
    words = [word.lower() for word in word_tokenize(review)]
    # Count the frequency of each negative word in the review
    negative_word_counts = Counter([word for word in words if word in negative_words])
    # Return the total count of negative words in the review
    return sum(negative_word_counts.values())

# Create a new column in the data dataframe with the count of negative words for each review
data["negative_word_count"] = data["reviews"].apply(count_negative_words)

# Group the reviews by the "trip verification" column and calculate the mean negative word count for each group
grouped_data = data.groupby("trip verification")["negative_word_count"].mean()

# Print the mean negative word count for each group
print(grouped_data)


# In[40]:


from nltk.tokenize import word_tokenize
from collections import Counter

# Define a set of negative words to search for
negative_words = {"bad", "not good", "poor", "terrible", "disappointing", "disappointed", "awful", "horrible", "miserable", "stupid", "sad", "nothappy"}

# Define a function to count the negative words in a review
def count_negative_words(review):
    # Tokenize the words in the review and convert to lowercase
    words = [word.lower() for word in word_tokenize(review)]
    # Count the frequency of each negative word in the review
    negative_word_counts = Counter([word for word in words if word in negative_words])
    # Return the total count of negative words in the review
    return sum(negative_word_counts.values())

# Create a new column in the data dataframe with the count of negative words for each review
data["negative_word_count"] = data["reviews"].apply(count_negative_words)

# Group the reviews by the "trip verification" column and calculate the total negative word count for each group
grouped_data = data.groupby("trip verification")["negative_word_count"].sum()

# Print the total negative word count for each group
print(grouped_data)


# In[41]:




# Define a set of negative words to search for
negative_words = {"bad", "not good", "poor", "terrible", "disappointing", "disappointed", "awful", "horrible", "miserable", "stupid", "sad","nothappy"}

# Define a function to count the negative words in a review
def count_negative_words(review):
    # Tokenize the words in the review and convert to lowercase
    words = [word.lower() for word in word_tokenize(review)]
    # Count the frequency of each negative word in the review
    negative_word_counts = Counter([word for word in words if word in negative_words])
    # Return the total count of negative words in the review
    return sum(negative_word_counts.values())

# Create a new column in the data dataframe with the count of negative words for each review
data["negative_word_count"] = data["reviews"].apply(count_negative_words)

# Group the reviews by the "trip verification" column and calculate the count of negative word for each group
grouped_data = data.groupby("trip verification")["negative_word_count"].sum()

# Convert the result to a DataFrame and plot a bar chart
ax = grouped_data.to_frame().plot(kind="bar", legend=False, color="blue")

# Set the chart title and axis labels
ax.set_title("Total Negative Word Counts by Trip Verification")
ax.set_xlabel("Trip Verification")
ax.set_ylabel("Total Negative Word Count")

# Show the chart
plt.show()


# In[42]:




# Create a pie chart with the percentage of negative reviews for each trip verification status
plt.pie(grouped_data.values, labels=grouped_data.index, autopct='%1.1f%%')

# Add title
plt.title('Percentage of Negative Reviews by Trip Verification Status')

# Show the chart
plt.show()


# In[43]:


positive_words = ["good", "great", "excellent"]
positive_reviews = data.loc[(data["trip verification"] != "trip verified") & (data["reviews"].str.contains("|".join(positive_words))), "reviews"]
for review in positive_reviews:
    print(reviews)
    break


# In[44]:


# Define a set of positive words to search for
positive_words = {"good", "excellent", "awesome", "fantastic", "great", "amazing", "superb", "wonderful", "happy", "satisfied"}

# Define a function to count the positive words in a review
def count_positive_words(reviews):
    # Tokenize the words in the review and convert to lowercase
    words = [word.lower() for word in word_tokenize(reviews)]
    # Count the frequency of each positive word in the review
    positive_word_counts = Counter([word for word in words if word in positive_words])
    # Return the total count of positive words in the review
    return sum(positive_word_counts.values())

# Create a new column in the data dataframe with the count of positive words for each review
data["positive_word_count"] = data["reviews"].apply(count_positive_words)

# Group the reviews by the "trip verification" column and calculate the mean positive word count for each group
grouped_data = data.groupby("trip verification")["positive_word_count"].mean()

# Print the mean positive word count for each group
print(grouped_data)



# In[45]:



# Define a set of positive words to search for
positive_words = {"good", "great", "excellent", "awesome", "fantastic", "amazing", "love", "like", "enjoy", "happy", "satisfied"}

# Define a function to count the positive words in a review
def count_positive_words(reviews):
    # Tokenize the words in the review and convert to lowercase
    words = [word.lower() for word in word_tokenize(reviews)]
    # Count the frequency of each positive word in the review
    positive_word_counts = Counter([word for word in words if word in positive_words])
    # Return the total count of positive words in the review
    return sum(positive_word_counts.values())

# Create a new column in the data dataframe with the count of positive words for each review
data["positive_word_count"] = data["reviews"].apply(count_positive_words)

# Group the reviews by the "trip verification" column and calculate the count of positive words for each group
grouped_data = data.groupby("trip verification")["positive_word_count"].sum()

# Print the count of positive words for each group
print(grouped_data)


# In[46]:




import matplotlib.pyplot as plt

# Define a set of positive words to search for
positive_words = {"good", "great", "excellent", "fantastic", "awesome", "amazing", "wonderful", "happy"}

# Define a function to count the positive words in a review
def count_positive_words(review):
    # Tokenize the words in the review and convert to lowercase
    words = [word.lower() for word in word_tokenize(review)]
    # Count the frequency of each positive word in the review
    positive_word_counts = Counter([word for word in words if word in positive_words])
    # Return the total count of positive words in the review
    return sum(positive_word_counts.values())

# Create a new column in the data dataframe with the count of positive words for each review
data["positive_word_count"] = data["reviews"].apply(count_positive_words)

# Group the reviews by the "trip verification" column and calculate the count of positive word occurrences for each group
grouped_data = data.groupby("trip verification")["positive_word_count"].sum()

# Plot the bar chart
fig, ax = plt.subplots()
grouped_data.plot(kind="bar", ax=ax)

# Set the chart title and axis labels
ax.set_title("Count of Positive Reviews by Trip Verification")
ax.set_xlabel("Trip Verification")
ax.set_ylabel("Count")

# Show the chart
plt.show()


# In[47]:


import matplotlib.pyplot as plt

# Define a set of positive words to search for
positive_words = {"good", "great", "excellent", "amazing", "wonderful", "fantastic", "terrific", "awesome", "satisfying", "pleasing", "enjoyable"}

# Define a function to count the positive words in a review
def count_positive_words(reviews):
    # Tokenize the words in the review and convert to lowercase
    words = [word.lower() for word in word_tokenize(reviews)]
    # Count the frequency of each positive word in the review
    positive_word_counts = Counter([word for word in words if word in positive_words])
    # Return the total count of positive words in the review
    return sum(positive_word_counts.values())

# Create a new column in the data dataframe with the count of positive words for each review
data["positive_word_count"] = data["reviews"].apply(count_positive_words)

# Group the reviews by the "trip verification" column and calculate the total positive word count for each group
grouped_data = data.groupby("trip verification")["positive_word_count"].sum()

# Plot a pie chart of the positive review counts
plt.pie(grouped_data, labels=grouped_data.index, autopct='%1.1f%%')
plt.title("Positive Review Counts by Trip Verification")
plt.show()


# In[48]:


# Define a set of positive words to search for
positive_words = {"good", "great", "excellent", "amazing", "wonderful", "fantastic", "terrific", "awesome", "satisfying", "pleasing", "enjoyable"}

# Define a function to count the positive words in a review
def count_positive_words(review):
    # Tokenize the words in the review and convert to lowercase
    words = [word.lower() for word in word_tokenize(review)]
    # Count the frequency of each positive word in the review
    positive_word_counts = Counter([word for word in words if word in positive_words])
    # Return the total count of positive words in the review
    return sum(positive_word_counts.values())

# Create a new column in the data dataframe with the count of positive words for each review
data["positive_word_count"] = data["reviews"].apply(count_positive_words)

# Find the countries with the most positive reviews
positive_reviews = data[data["positive_word_count"] > 0]
positive_reviews_by_country = positive_reviews.groupby("countries")["positive_word_count"].count()
best_countries = positive_reviews_by_country.sort_values(ascending=False)

# Print the best countries
print(best_countries.head(10))


# In[49]:


import plotly.express as px

# Replace "best_countries" with your own variable name for the grouped data
fig = px.choropleth(best_countries, locations=best_countries.index, locationmode="country names", color="positive_word_count", 
                    hover_name=best_countries.index, projection="natural earth", title="Countries with the most positive reviews")
fig.show()


# In[50]:


# Define a set of negative words to search for
negative_words = {"bad", "not good", "poor", "terrible", "disappointing", "disappointed", "awful", "horrible", "miserable", "stupid", "sad", "nothappy"}

# Define a function to count the negative words in a review
def count_negative_words(review):
    # Tokenize the words in the review and convert to lowercase
    words = [word.lower() for word in word_tokenize(review)]
    # Count the frequency of each negative word in the review
    negative_word_counts = Counter([word for word in words if word in negative_words])
    # Return the total count of negative words in the review
    return sum(negative_word_counts.values())

# Create a new column in the data dataframe with the count of negative words for each review
data["negative_word_count"] = data["reviews"].apply(count_negative_words)

# Find the countries with the most negative reviews
negative_reviews = data[data["negative_word_count"] > 0]
negative_reviews_by_country = negative_reviews.groupby("countries")["negative_word_count"].count()
worst_countries = negative_reviews_by_country.sort_values(ascending=False)

# Print the worst countries
print(worst_countries)


# In[51]:


import plotly.express as px

# Create a new dataframe with the count of negative reviews by country
negative_reviews_count = negative_reviews_by_country.reset_index(name='count')

# Create a choropleth map based on the negative reviews count by country
fig = px.choropleth(negative_reviews_count, locations="countries", locationmode='country names', color="count",
                    title="Negative Reviews by Country", color_continuous_scale="Reds")
fig.show()

