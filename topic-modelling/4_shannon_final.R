# The goal of this code is to answer the question:
# Does breadth of topics differ as a function of rating?
# We’ll quantify breadth of topics discussed using the Shannon Diversity Index.

# -------------------
# Set up libraries and read in data
# -------------------

#Set the working directory to the directory of the active RStudio document
currentPath <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(currentPath)

# Load necessary libraries for data manipulation, visualization, and analysis
library(ggplot2)    # For creating plots
library(reshape2)   # For data reshaping
library(plotly)     # For interactive plots
library(vegan)      # For ecological diversity analysis, which adapt to topic diversity
library(boot)       # For bootstrap methods
library(dplyr)      # For data manipulation
library(tidyr)
library(readr)

# Read in the dataset containing topic frequencies in wide format
urge_wide <- read_csv("~/Desktop/URGE_llm_topic_frequencies_wide_forSDI.csv")# Define manual labels for the columns to replace default column names
# I did this without clinical knowledge, so better to label them again.

manual_labels <- c("Timestamp",
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
                   )

# Assign the manual labels to the dataframe columns
colnames(urge_wide) <- manual_labels

# -------------------
# Visualize topics by ratings
# -------------------

# The plot gets a little difficult to read if the scale goes all the way up to 34
# and the rest are lower. So, this is a workaround that should be fixed later, but
# let's take that specific data entry where Timestamp is 0 and 'Happy and content' equals 34 
# and set it to 20 for now. Again, fix later.

# urge_wide$`Happy and content`[urge_wide$Timestamp == 0 & urge_wide$`Happy and content` == 34] <- 20

# Reshape the data from wide to long format for easier plotting with ggplot2
df <- melt(urge_wide, id = 'Timestamp')

# Create a heatmap using ggplot2 to visualize the frequency of each topic across timestamps
p <- ggplot(df, aes(Timestamp, variable)) +
  geom_raster(aes(fill=value)) +
  scale_fill_continuous(type = 'viridis') +  # Use Viridis color scale for better color perception.   #scale_fill_distiller(palette = "OrRd", direction = 1) + # Alternative color scale
  labs(x="Rating",
       y="Topic",
       title = "Topic as function of rating",
       fill = "Frequency") +
  theme(text = element_text(family = 'Fira Sans'), # Set font family
        plot.title = element_text(hjust = 0.5))  # Center the plot title

## Convert the ggplot to an interactive Plotly plot
ggplotly(p)

# Add horizontal lines to separate topics in the heatmap for better readability
q <- p + geom_hline(aes(yintercept = as.numeric(variable) + 0.5), 
                    color = "black", 
                    linewidth = 0.2)

# Convert the new ggplot with horizontal lines to an interactive Plotly plot
ggplotly(q)

# -------------------
# Shannon Index Calculation
# -------------------

# Select only the topic frequency columns, excluding the Timestamp
data = urge_wide %>% select(-Timestamp)

# Calculate the Shannon diversity index for each row
shan <- vegan::diversity(data, "shannon")
shan # Display the Shannon diversity index values

# Plot the Shannon diversity index
plot(shan)

# -------------------
# Bootstrap Analysis to get Shannon Confidence Interval
# -------------------

# Define the number of bootstrap samples to generate
n_boot <- 100000

# Initialize a matrix to store the bootstrap results
# Rows represent bootstrap samples

bootstrap_results <- matrix(NA, nrow = n_boot, ncol = nrow(data))

# Define a bootstrap function to calculate the Shannon diversity index for a single row
my_boot <- function(my_row){
 
  # Create a vector where each topic is repeated according to its count in the row
  topics_vector <- unlist(lapply(names(my_row), function(topic) {
    rep(topic, my_row[topic])
  }))
  
  # Sample topics with replacement to create a bootstrap sample
  sampled_vector <- sample(topics_vector, length(topics_vector), replace = TRUE)
  
  # Create a frequency table from the sampled topics
  sampled_table <- table(sampled_vector)
  
  # Convert the table to a list of counts
  sampled_counts <- as.list(sampled_table)
  
  # Ensure all original topics are represented in the sampled counts, assigning 0 if not present
  for (topic in names(my_row)) {
    if (!topic %in% names(sampled_counts)) {
      sampled_counts[[topic]] <- 0
    }
  }
  
  # Reorder the sampled counts to match the original order and convert to numeric
  sampled_counts <- sampled_counts[names(my_row)] %>% as.numeric()
  
  # Calculate and return the Shannon diversity index for the sampled counts
  return(diversity(sampled_counts, "shannon"))
}

# Perform bootstrapping by iterating over the number of bootstrap samples
for(i in 1:n_boot){
  # Iterate over each row in the data
  for (j in 1:nrow(data)) {
    # Apply the bootstrap function and store the result
    bootstrap_results[i,j] <- my_boot(data[j, , drop = FALSE])
  }
}

# Calculate the mean and 95% confidence intervals for the Shannon diversity index across bootstrap samples
ci <- apply(bootstrap_results, 2, function(x) {
  c(mean = mean(x), lower = quantile(x, 0.025), upper = quantile(x, 0.975))
})

# Convert the confidence intervals matrix to a data frame for plotting
ci_df <- as.data.frame(t(ci))
ci_df$Timestamp <- factor(1:nrow(data)-1, levels = 1:nrow(data)-1)
colnames(ci_df) <- c('mean', 'lower', 'upper', 'Timestamp')

# Optionally, write the confidence intervals to a CSV file
# write.csv(ci_df, paste0(currentPath, '/shannon_CI.csv')) 

# Optionally, read the confidence intervals from a CSV file
# ci_df <- read.csv(paste0(currentPath, '/shannon_CI.csv'))


# Create a ggplot to visualize the mean Shannon diversity index with error bars representing the confidence intervals
p <- ggplot(ci_df, aes(x = as.numeric(as.character(Timestamp)), y = mean)) +
  geom_point(size = 3, color = "#2C3E50") +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.15, color = "#2C3E50") +
  scale_x_continuous(breaks = 0:10) +
  labs(
    title = "Topic Diversity Across Suicidal Urge Ratings",
    subtitle = "Bootstrapped Shannon Diversity Index with 95% Confidence Intervals",
    x = "SI Urge Rating",
    y = "Shannon Diversity Index"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 16),
    plot.subtitle = element_text(hjust = 0.5, size = 13),
    axis.text.x = element_text(angle = 0),
    axis.title = element_text(size = 14),
    panel.grid.major = element_line(color = "#dddddd"),
    panel.grid.minor = element_blank()
  )


# Display the plot
p
# Define the path to save the Shannon diversity index plot
plot_save_path <- paste0(currentPath, "/DESIRE_shannon_diversity_index_plot.png")

# Save the plot as a high-resolution PNG file
ggsave(
  plot_save_path, 
  plot = p, 
  width = 8, 
  height = 6, 
  dpi = 1200, 
  bg = "white"
)
