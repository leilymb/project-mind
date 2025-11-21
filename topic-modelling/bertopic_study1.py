# -*- coding: utf-8 -*-
:note: change runtime type to T4 gpu (should be free, and significantly speeds up)

**Summary of the Code**:
1.  Environment Setup
2.  Data Loading and Cleaning
3.  Load Embedding Model and Encode Responses
4.  Topic Modeling with BERTopic
5.  Visualize Results (common words for each topic, topics for each rating, etc)
6.  Evaluate how many clusters/topics we should have
7.  Save model and save data for additional analyses

The following code is simplified -- see other code files for testing/looping through options for clustering algorithm, clustering parameters, embedding models, etc.

**Environment Setup**:

Imports: The code imports necessary libraries for data manipulation, machine learning, and visualization.

Google Drive: It mounts Google Drive to access data stored there.

**Note**: we require Python 3.6 or higher, PyTorch 1.6.0 or higher and transformers v4.6.0 or higher. The code does not work with Python 2.7. We require bertopic 0.10.0 because topics_over_time breaks in newest version.
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install transformers # lots of output, so use capture to hide the mess

!pip install -U kaleido

import kaleido

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install bertopic

from google.colab import drive
from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic
from transformers.pipelines import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import pandas as pd
import re
from datetime import datetime
import plotly.io as pio
import numpy as np

drive.mount('/content/drive')

"""**Data Loading and Cleaning**:

Data Path: Defines the path to the CSV file containing responses.

Reading Data: Loads the data into a pandas DataFrame.

Cleaning Responses: Standardizes various representations of 'NA' and removes irrelevant or missing responses.
"""

# read in data
project_folder = '/content/drive/MyDrive/ProjectSafe_CleanCode/'
data_folder = '/content/drive/MyDrive/ProjectSafe_CleanCode/data/'
data_path = data_folder + 'all_responses_current.csv'
dat = pd.read_csv(data_path)

dat

"""Checking out the data to confirm unique participant numbers and the number of entries."""

#Confirming that all participants have a time 1 and time 2, and if they have all entries.
# Group by 'ResponseId.x' and aggregate
grouped = dat.groupby('ResponseId.x').agg(
    total_entries=('timepoint', 'count'),
    has_timepoint_2=('timepoint', lambda x: (x == 2).any())
).reset_index()

#Check for participants who do not have 36 entries or do not have time point 2
participants_missing_data = grouped[
    (grouped['total_entries'] != 36) | (~grouped['has_timepoint_2'])
]

#Count the number of such participants
num_participants_missing_data = participants_missing_data.shape[0]

print(f"Number of participants without time 2 or without 36 entries: {num_participants_missing_data}")

#Get the list of ResponseIds
response_ids_missing_data = participants_missing_data['ResponseId.x'].tolist()

#Print the number of participants and their ResponseIds
print(f"Number of participants without time 2 or without 36 entries: {len(response_ids_missing_data)}")
print("Their ResponseIds are:")
for rid in response_ids_missing_data:
    print(rid)

#More details about these participants:
print("\nDetailed information:")
print(participants_missing_data)

#All 160 have a T1 and T2, two pts are missing some entries.

#Confirming we have 160 unique participants in time 1.
#For timepoint 1
unique_time1 = dat[dat['timepoint'] == 1]['ResponseId.x'].nunique()
unique_time1

#Confirming we have 160 unique participants in time 2.
# For timepoint 2
unique_time2 = dat[dat['timepoint'] == 2]['ResponseId.x'].nunique()
unique_time2

"""
Confirmed we have 160 participants in Time 1 and 2!

"""

#Define a dictionary mapping common variations of "NA" to a standardised string ("NA").
replacement_dict = {
    'na':'NA',
    'na ':'NA',
    'Na':'NA',
    'n/a':'NA',
    'N/a':'NA',
    'N/A':'NA',
    'n\\a':'NA',
    'N\\A':'NA',
    'N\\A':'NA',
    'N/a.':'NA',
    'n/a':'NA',
    'n/A':'NA',
    '/a':'NA',
}
dat['response'] = dat['response'].replace(replacement_dict)
#Remove rows where response is missing, NA or "10".
dat = dat[dat['response'].notna()]
dat = dat[dat['response'] != 'NA']
dat = dat[dat['response'] != '10']

dat

"""**Data splitting**:

Types: Separates the data into different categories based on the 'type' column (urge, intent, desire).
"""

urge_df = dat[dat['type'] == 'urge']
intent_df = dat[dat['type'] == 'intent']
desire_df = dat[dat['type'] == 'desire']

urge_df

# Convert 'response' column to a list of strings for all documents
# BERTopic sometimes requires lists of strings
all_docs = dat.response.astype(str).values.tolist()

# Convert 'response' column to lists for each specific type
urge_docs = urge_df.response.astype(str).values.tolist()
intent_docs = intent_df.response.astype(str).values.tolist()
desire_docs = desire_df.response.astype(str).values.tolist()

#How many documents in urge? 1221!
len(urge_docs)

"""Model Loading: Loads a pre-trained SentenceTransformer model from the specified new_model_path. This model is fine-tuned for generating sentence embeddings relevant to the project's context.

Embedding Generation: Encodes the urge_docs (documents related to "urge") into vectors of numbers/embeddings using the loaded model. These embeddings will be used for clustering and topic modeling.
"""

new_model_path = '/content/drive/My Drive/ProjectSafe_CleanCode/embeddingModel/gpt4gen_newprompts_reranked_dev/training_positive_pairs_bge_large-SentenceTransformer/'
new_model = SentenceTransformer(new_model_path)
embeddings_new = new_model.encode(urge_docs, show_progress_bar=False)

#Save embeddings for later (heterogeneity analyses)
np.save(data_folder +'urge_embeddings_4_17.npy', embeddings_new)

"""**Topic Modeling with BERTopic**:

Initialization: Sets up BERTopic with a custom clustering model (Agglomerative Clustering) and no dimensionality reduction.

Fitting: Fits the model to the 'urge' documents and their embeddings.


If we were not using a embedding model specifically finetuned for this purpose, we would use some dimensionality reduction to reduce noise.
"""

from scipy.cluster import hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction

## Set-up

#Initialize an empty dimensionality reduction model (no reduction)
empty_dimensionality_model = BaseDimensionalityReduction()

#Initialize an Agglomerative Clustering model with 20 clusters #actually let's go with 13
cluster_model = AgglomerativeClustering(n_clusters=13)

#Initialize the BERTopic model with the specified embedding, clustering, and dimensionality reduction models
topic_model = BERTopic(embedding_model=new_model,
                       hdbscan_model=cluster_model,
                       umap_model=empty_dimensionality_model)

## Fit
# Fit the BERTopic model on the 'urge' documents and their embeddings
topics, probs = topic_model.fit_transform(urge_docs, embeddings = embeddings_new)

# Initialize a CountVectorizer with English stop words and n-grams ranging from 1 to 2.
# This is done so that our word-frequency plots don't contain a bunch of words like 'and' 'a' 'the' etc
# n-gram from 1 to 2 means that the bar plots have up to two-word combinations that show up
vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(2,2)) #Decided to go with 2,2, so both the min and max n-gram length is 2

# google CountVectorizer BertTOpic and https://maartengr.github.io/BERTopic/api/ctfidf.html#bertopic.vectorizers._ctfidf.ClassTfidfTransformer

# Update the topics in the BERTopic model using the new vectorizer
topic_model.update_topics(urge_docs, vectorizer_model=vectorizer_model)

# Create a bar chart visualization of the top 30 topics with 4 top words each
barplot = topic_model.visualize_barchart(top_n_topics=30, n_words = 4, custom_labels=True)#, height = 400, n_words = 5)

barplot

# Create the barchart with adjusted parameters
barplot = topic_model.visualize_barchart(
    top_n_topics=30,
    n_words=4,
    custom_labels=True,
    height=400,  #Increase height to accommodate labels
    width=300    #Adjust width for better proportions
)
# Make subplot titles smaller
for annotation in barplot['layout']['annotations']:
    annotation['font'] = dict(size=12)

# Update the layout to adjust font sizes and spacing
barplot.update_layout(
    title_font=dict(size=15),  #Adjust main title size
    font=dict(size=8),        #Set base font size for all text
    margin=dict(t=100, b=50)    #Add more margin at top and bottom
)

# Update y-axis properties for better label display
barplot.update_yaxes(
    tickfont=dict(size=11),    #Adjust label font size
    automargin=True            #Automatically adjust margins to prevent cutoff
)

# How many documents are there in each topic?
pd.Series(topics).value_counts()

df_topics = pd.DataFrame({
    "Document": urge_docs,
    "Topic": topics
})

# Filter for Topic 11
topic_11_docs = df_topics[df_topics["Topic"] == 12]["Document"]

# Display the first few
topic_11_docs.head(10).tolist()

#Display barplot
barplot

"""Based on reading which documents are in each group (13 total) (manually going through the .csv written out), we can label topics."""

manual_labels = [
    "Normal life thoughts",
    "Existential reflections",
    "Family & friends reactions",
    "Thinking of plans",
    "Ruminative ideation",
    "Passive suicidal ideation",
    "Depressed, exhausted",
    "Self harm & suicide methods",
    "Happy life thoughts",
    "Low self-esteem",
    "Active suicidal ideation",
    "Not sure/I don't know",
    "None/No thoughts"
]

# Assign the manual labels to the topics in the BERTopic model
topic_model.set_topic_labels(manual_labels)

# one thing to know: I think the clustering is a determenistic process, based on fixed embeddings.
# so the groupings should always be the same, and there shouldn't be any
# randomness leading to different results over different runs.
# however, sometimes the specific topic order switches, so make sure to check that the label ordering is correct.

# Define the output file name for the bar plot HTML
outwrite_name_bar = project_folder + "figures/new_embeds3_26_25_barplots.html"

# Save the bar plot as an HTML file
barplot.write_html(outwrite_name_bar, include_plotlyjs='cdn')

import kaleido

import kaleido
kaleido.get_chrome_sync()

import plotly.io as pio

# Save as a high-res PNG
pio.write_image(barplot, project_folder + "figures/barplot_highres_13.png", format='png', width=1000, height=600, scale=3)

project_folder

barplot

"""**View documents and associated documents**"""

# Get detailed information about topics and their document counts

topic_model.get_topic_info()

# Get information about each document's topic assignment

topic_documents_df = topic_model.get_document_info(urge_docs)
topic_documents_df

"""**Save document and topic assignment information**"""

# Sort the DataFrame by 'topic' column before saving
#topic_documents_df = topic_documents_df.sort_values(by='Topic', ascending=True)

# Get the current date and time
current_datetime = datetime.now().strftime('%m.%d.%y_%H.%M')

# Generate the filename with the current date and time
filename = f'URGE_new_embeds_{current_datetime}_topic_documents.csv'

# Save the sorted dataframe to a CSV file
topic_documents_df.to_csv(project_folder + filename, index=False)

# Alternatively, save with a specific filename (commented out)
# topic_documents_df.to_csv(project_folder + 'new_embeds_7.22.24_topic_documents.csv', index=False)

"""**View topics by ratings**"""

# Extract just the documents and their corresponding timestamps (ratings), for use in future plots
just_docs = urge_df.response.astype(str).values.tolist()
just_timestamps = urge_df.rating.astype(int).values.tolist()

topics_over_time = topic_model.topics_over_time(docs = just_docs, timestamps = just_timestamps)
plot = topic_model.visualize_topics_over_time(topics_over_time, custom_labels = True)

#Modify the title of the plot
plot.update_layout(title='Frequency of Topics over Ratings')

#Display the updated figure
plot.show()

# Define the output file name for the topics over time plot using the current date and time

# outwrite_name = project_folder + "figures/new_embeds7_23_24_topics_rating.html"

filename = f'URGE_new_embeds_{current_datetime}_topics_rating.html'
outwrite_name = project_folder + 'figures/' + filename

plot = topic_model.visualize_topics_over_time(topics_over_time,custom_labels = True)
plot.write_html(outwrite_name)

outwrite_name

"""Alternative view of topics per rating"""

topics_per_class = topic_model.topics_per_class(docs = just_docs,
                                                classes = just_timestamps, )

topic_model.visualize_topics_per_class(topics_per_class,
                                       custom_labels = True,
                                       top_n_topics= 20 )

# Visualize the hierarchical structure of topics with custom labels
# This is post-hoc combination of topics. It takes (I believe, need to check)
# the center of each cluster, and then performs additional hierarchical clustering on them

linkage_function = lambda x: sch.linkage(x, 'single', optimal_ordering=True)
hierarchical_topics = topic_model.hierarchical_topics(urge_docs, linkage_function=linkage_function)

topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics,  custom_labels = True)

"""**Clustering Evaluation**:

How many clusters should we have?

Silhouette Analysis: Evaluates the quality of clustering across a range of cluster numbers (2 to 300) using silhouette scores.

Plotting: Visualizes the silhouette scores to help determine the optimal number of clusters. +1 is best, -1 is worst.

In addition to silhouette, we also calculated Calinski-Harabasz and Davies Bouldin scores as well as a variance plot.
"""

# can check out additional metrics here: https://scikit-learn.org/dev/modules/clustering.html#silhouette-coefficient

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.spatial.distance import cdist

# Assuming `embeddings_new` is your dataset
X = embeddings_new

# Range of clusters to evaluate
range_n_clusters = range(2, 50)
#Lists for scores
silhouette_avg_scores = []
variance_scores = []
calinski_harabasz_scores = []
davies_bouldin_scores = []
#Cluster loop, iterates over the range of cluster numbers defined, performing the demanded clustering and calculating metrics for each.
for n_clusters in range_n_clusters:
    #Perform the clustering
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(X)

    #Calculate the average silhouette score
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_avg_scores.append(silhouette_avg)

    #Calculate total within-cluster variance, sum of squared distances from each point to its cluster's centroid.
    total_variance = 0
    for i in range(n_clusters):
        cluster_points = X[cluster_labels == i]
        centroid = np.mean(cluster_points, axis=0)
        variance = np.sum(np.sum((cluster_points - centroid) ** 2))
        total_variance += variance

    variance_scores.append(total_variance)

    #Calculate Calinski-Harabasz score
    ch_score = calinski_harabasz_score(X, cluster_labels)
    calinski_harabasz_scores.append(ch_score)

    #Calculate Davies-Bouldin score
    db_score = davies_bouldin_score(X, cluster_labels)
    davies_bouldin_scores.append(db_score)

    print(f"For n_clusters = {n_clusters}:")
    print(f"  Silhouette Score: {silhouette_avg:.4f}")
    print(f"  Variance: {total_variance:.4f}")
    print(f"  Calinski-Harabasz Score: {ch_score:.4f}")
    print(f"  Davies-Bouldin Score: {db_score:.4f}")

# Plot the metrics
fig, axs = plt.subplots(2, 2, figsize=(20, 12))

# Silhouette plot
axs[0, 0].plot(range_n_clusters, silhouette_avg_scores, marker='o')
axs[0, 0].set_xlabel('Number of Clusters')
axs[0, 0].set_ylabel('Silhouette Score')
axs[0, 0].set_title('Silhouette Analysis')

# Variance plot
axs[0, 1].plot(range_n_clusters, variance_scores, marker='o')
axs[0, 1].set_xlabel('Number of Clusters')
axs[0, 1].set_ylabel('Total Within-Cluster Variance')
axs[0, 1].set_title('Variance Analysis')

# Calinski-Harabasz plot
axs[1, 0].plot(range_n_clusters, calinski_harabasz_scores, marker='o')
axs[1, 0].set_xlabel('Number of Clusters')
axs[1, 0].set_ylabel('Calinski-Harabasz Score')
axs[1, 0].set_title('Calinski-Harabasz Index')

# Davies-Bouldin plot
axs[1, 1].plot(range_n_clusters, davies_bouldin_scores, marker='o')
axs[1, 1].set_xlabel('Number of Clusters')
axs[1, 1].set_ylabel('Davies-Bouldin Score')
axs[1, 1].set_title('Davies-Bouldin Index')

plt.tight_layout()
plt.show()

"""Interpretation: more clusters is always better. So, we pick a cluster number that still has a reasonable number of responses in each cluster, and is interpretable. In the pre-registration we said 20, so let's stick to that (unless it's incoherent, which it seems to not be). Update: We went with 13!

**Exporting Results For Future Analyses**

Saves topic frequency data in a wide format suitable for further analysis. (specify which analysis)
"""

topics_ratings_frequency = topics_over_time[['Topic', 'Frequency', 'Timestamp']]
topics_ratings_frequency

# Pivot the DataFrame
frequencies_wide = topics_ratings_frequency.pivot(index='Timestamp', columns='Topic', values='Frequency')

# Resetting the index to make Timestamp a column again
frequencies_wide.reset_index(inplace=True)

# Rename columns for better readability
frequencies_wide.columns.name = None  # remove the name of the columns
frequencies_wide.columns = ['Timestamp'] + [f'Topic_{col}' for col in frequencies_wide.columns if col != 'Timestamp']

frequencies_wide.fillna(0, inplace=True)

# Display the wide format DataFrame
print(frequencies_wide)

frequencies_wide.to_csv(project_folder + 'URGE_analysis_new_embeds_7.23.24_topic_frequencies_wide.csv', index=False)

"""**Model Saving**

Saves the trained BERTopic model using efficient serialization.

"""

# To save: model, reduced embeddings, representative docs
!pip install safetensors

topic_model.save(project_folder + "topicModel_new_embeds_12.9.24", serialization="safetensors", save_ctfidf=True)

project_folder

#loaded_model = BERTopic.load(project_folder + "topicModel_new_embeds_7.23.24")
