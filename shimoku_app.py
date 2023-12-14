#%%

import shimoku_api_python as Shimoku
import pandas as pd
# Client initialization with playground mode
s = Shimoku.Client()


#%%
s = Shimoku.Client(
    local_port=8080,
    async_execution=True,
    verbosity='INFO', 
)


#%%
# Necessary for compatibility with cloud execution
s.set_workspace() 

# Set the group of the menu
s.set_board('Custom Board')

#%%


# Set the menu path 'catalog' with the sub-path 'bar-example'
s.set_menu_path('Technical Test', 'Context')

s.plt.html(html=f'<h3> Context </h3>'
                f"<p>We are developing a lead scoring algorithm for a client's Event Management SaaS application. We receive raw CRM data "
                f"from the client and prepare it to train a classification algorithm that predicts lead conversion probability. The "
                f"datasets track lead journeys and sales offer outcomes, helping us make data-driven decisions. We receive two datasets: "
                f"'leads.csv' for potential clients and 'offers.csv' for those who reached the demo meeting. The target column, 'Status' "
                f"in 'offers.csv,' indicates whether a client purchased the product.</p>"
                f'<h3> Content </h3>'
                f'<p> 1. Exploration and Data Understanding </p>'
                f'<p> 2. Univariate and Multivariate Analysis </p>'
                f'<p> 3. Business Analysis </p>'
                f'<p> 4. Data Cleaning and Preprocessing </p>'
                f'<p> 5. Model Training and Hyperparameter Search </p>'
                f'<p> 6. Results </p>'
                f'<p> 7. Model Validation </p>',
            order=0, cols_size=22, rows_size=5, padding='1,1,1,1')
s.run()


#%%

# Set the menu path 'catalog' with the sub-path 'bar-example'
s.set_menu_path('Technical Test', '1. Exploration and Data Understanding')

s.plt.html(html=f'<h3> Exploration and Data Understanding </h3>'
                f'<p>This phase involved understanding and exploring the data, its source, and identifying any potential issues related to quality or completeness. During this step, we explored two files: "leads" and "offers" and gathered the following insights:</p>'
                f'<p>1. The data is imbalanced, with a significantly higher number of leads compared to offers.</p>'
                f'<p>2. The data is incomplete, containing missing values in some columns.</p>'
                f'<p>3. Certain columns require transformation or the creation of new features to make them suitable for modeling (e.g., date columns).</p>'
                f'<p>4. Some columns are irrelevant for the model, such as ID and Discount code.</p>'
                f'<p>5. To complete the information, it is necessary to merge the two files.</p>',
            order=0, cols_size=22, rows_size=5, padding='1,1,1,1')
s.run()

#%%

# Set the menu path 'catalog' with the sub-path 'bar-example'
s.set_menu_path('Technical Test', '2. Univariate and Multivariate Analysis')

s.plt.html(html=f'<h3>Univariate and Multivariate Analysis</h3>'
                f'<p>In this section, we will delve into both univariate and multivariate analysis to gain a comprehensive understanding of the data and uncover potential insights.</p>'
                f'<p>During our analysis, we made several key observations:</p>'
                f'<p>1. There is a wide variety of successful offers, but there are relatively few instances of unsuccessful ones. For instance, we noticed that many offers, initially categorized as qualified or existing in the "offers" dataset, were later reclassified as "closed lost."</p>'
                f'<p>2. We encountered a substantial number of leads that could not be definitively classified as either successful or unsuccessful, for example the "Negotiatio" ones.</p>'
                f'<p>3. Additionally, we identified numerous types of acquisition campaigns that needed to be grouped together to facilitate a better grasp of the data.</p>'
                f'<p>Our univariate and multivariate analyses have laid the foundation for deeper insights into the dataset, which will be crucial for subsequent stages of our analysis and modeling.</p>',
            order=0, cols_size=22, rows_size=5, padding='1,1,1,1')
s.run()


#%%

# Set the menu path 'catalog' with the sub-path 'bar-example'
s.set_menu_path('Technical Test', '3. Business Analysis')

s.plt.html(html=f'<h3>Business Analysis</h3>'
                f'<p>The Business Analysis phase is a critical step in our process, as it requires a focused examination of the data with an analytics mindset. This step involves identifying key variables that may not be available at the time the model is intended for use. Some of these variables include:</p>'
                f'<ul>'
                f'<li>Discarded/Nurturing Reason: Understanding why certain leads were discarded or nurtured rather than converted into customers is crucial for future decision-making and optimization.</li>'
                f'<li>Close Date: Analyzing the historical close dates of successful deals can provide valuable insights into sales cycles and seasonality.</li>'
                f'<li>Loss Reason: Investigating the reasons behind lost opportunities can help pinpoint areas for improvement in sales strategies and product offerings.</li>'
                f'</ul>'
                f'<p>By identifying these variables and their potential impact on our modeling and analysis, we can develop strategies to address any data gaps and ensure that our predictive model remains effective and relevant in real-world scenarios. A comprehensive business analysis is essential for making informed decisions and driving actionable insights from our data.</p>',
                order=0, cols_size=22, rows_size=5, padding='1,1,1,1')
s.run()

#%%

s.set_menu_path('Technical Test', '4. Data Cleaning and Preprocessing')

s.plt.html(html=f'<h3>Data Cleaning and Preprocessing</h3>'
                f'<p>This step should include data preprocessing, feature engineering, and feature selection. These technical aspects are critical for preparing the data for modeling.</p>'
                f'<p>The 5 most relevant processes that were undertaken are:</p>'
                f'<ol>'
                f'<li>The creation of the real target column, which took into account the actual lost and won opportunities based on both datasets and avoided the inconclusive ones.</li>'
                f'<li>The creation of the new column "Acquisition Campaign," which grouped the different types of campaigns.</li>'
                f'<li>The creation of columns for the weekday, month day, and month of the opportunity creation date.</li>'
                f'<li>The encoding of the categorical variables.</li>'
                f'<li>Imputation of missing values based on the respective columns.</li>'
                f'</ol>',

                order=0, cols_size=22, rows_size=5, padding='1,1,1,1')
s.run()


#%%

s.set_menu_path('Technical Test', '5. Model Training and Hyperparameter Search')

s.plt.html(html=f'<h3>Model Training and Hyperparameter Search</h3>'
                f'<p>Training various models and tuning hyperparameters was essential for finding the best performing model. It was important to use techniques like cross-validation to evaluate model performance and avoid overfitting, in this it was used a K-fold = 5. The chosen models are well-known for their effectiveness in classification tasks and offer a range of flexibility and accuracy.</p>'
                f'<p>For this case, the following models were used:</p>'
                f'<ul>'
                f'<li>XGB Classifier: Known for its speed and performance, particularly in sparse data environments.</li>'
                f'<li>Random Forest Classifier: A robust model that is effective for handling large datasets with higher dimensionality, providing insights into feature importance.</li>'
                f'<li>Gradient Boosting Classifier: An accurate and effective model, especially useful for unbalanced datasets.</li>'
                f'</ul>'
                f'<p>Each model’s hyperparameters were optimized through a Randomized Search approach, focusing particularly on the recall metric for the successful class. This metric was prioritized as it is crucial in our context to correctly identify as many positive instances (successful class) as possible, even at the expense of making some incorrect positive predictions (false positives). This approach helps in minimizing the risk of overlooking potential opportunities, which is pivotal in scenarios where the cost of missing a true positive is high.</p>',
                order=0, cols_size=22, rows_size=5, padding='1,1,1,1')
s.run()


#%%

s.set_menu_path('Technical Test', '6. Results')

s.plt.html(html=f'<h3>Results</h3>'
                f"<p>The Randomized Search for the RandomForestClassifier has identified the following optimal hyperparameters, each contributing uniquely to the model's performance:</p>"
                f'<ul>'
                f'<li>N Estimators (500): Number of trees in the forest. Higher numbers generally lead to better performance but increase computational cost.</li>'
                f'<li>Min Samples Split (2): The minimum number of samples required to split an internal node. Lower values can lead to a more complex tree, whereas higher values prevent overfitting.</li>'
                f'<li>Min Samples Leaf (7): The minimum number of samples required to be at a leaf node. This parameter further guards against overfitting by smoothing the model.</li>'
                f'<li>Max Features (log2): The number of features to consider when looking for the best split. Using "log2" features reduces variance but increases bias.</li>'
                f'<li>Max Depth (15): The maximum depth of the tree. Limits the growth of the tree to prevent overfitting.</li>'
                f'<li>Class Weight (Balanced): Adjusts weights inversely proportional to class frequencies in the input data. This is crucial for datasets with imbalanced classes.</li>'
                f'</ul>'
                f'<p>With these parameters, the RandomForestClassifier achieved a recall score of 0.8017, highlighting its effectiveness in accurately identifying positive cases. Notably, this model demonstrated superior performance consistently, reinforcing its reliability and robustness as a tool in our predictive analysis.</p>',
                order=0, cols_size=22, rows_size=5, padding='1,1,1,1')
s.run()



# %%



# Create a DataFrame from the provided CSV data
data = pd.read_csv("shap_values.csv" , header=None, names=['Feature', 'Value'])

data
#%%
# Set the menu path 'Technical Test' with the sub-path 'Model'
s.set_menu_path('Technical Test', 'Model')

# Add a description of the model preparation
s.plt.html(html=f'<h3>Model Preparation</h3>'
                f'<p>Using machine learning, we can analyze and visualize the data:</p>',
                order=0, cols_size=22, rows_size=5, padding='1,1,1,1')

# Create a bar chart to display the data
s.plt.bar(order=1, title='Feature Importance', data=data, x='Feature', y='Value')

# Run the Shimoku Python code
s.run()

#%%


data = pd.read_csv("shap_values.csv", header=None, names=['Feature', 'Value'])

data.dropna(inplace=True)
data['feature_group'] = data['Feature'].apply(lambda x: x.split('_')[0])

data = data.groupby('feature_group')['Value'].sum().sort_values(ascending=False).reset_index()
# Convert the DataFrame to a dictionary
data_dict = data.to_dict(orient='records')

# Set the menu path 'Technical Test' with the sub-path 'Model'
s.set_menu_path('Technical Test', '6. Results: Variable Analysis')

# Add the model analysis text
s.plt.html(html=f'<h3> Feature Impact on Lead Conversion Prediction </h3>'
                f'<p>Insights from SHAP values have revealed the importance of different feature groups in predicting the success of lead conversion. These insights are crucial for strategic planning and optimization:</p>'
                f'<p>1. <strong>Use Case</strong>: With the highest value, this indicates the critical role of categorizing leads based on their use case for conversion success.</p>'
                f'<p>2. <strong>Source</strong>: The origin of leads (inbound or outbound) is the second most influential factor, essential for refining marketing strategies.</p>'
                f'<p>3. <strong>Acquisition Campaign</strong>: Specific campaigns are significant in determining lead quality, highlighting the need for campaign effectiveness analysis.</p>'
                f'<p>4. <strong>Campaign Group</strong>: Different campaign groups vary in their impact on lead conversion, suggesting a potential for resource reallocation.</p>'
                f'<p>5. <strong>Month</strong>: Indicates possible seasonality effects in lead conversion rates, which can guide the timing of marketing efforts.</p>'
                f'<p>6. <strong>City</strong>: Geographic location impacts lead success, pointing to the importance of localized marketing approaches.</p>'
                f'<p>7. <strong>Day of Month</strong>: Suggests specific days may be more favorable for lead engagement and conversion activities.</p>'
                f'<p>8. <strong>Weekday</strong>: Has a smaller but still notable effect, potentially indicating the best days to contact leads.</p>',
            order=2, cols_size=22, rows_size=5, padding='1,1,1,1')


# Plotting the classification metrics
s.plt.bar(
    order=3,  # Adjust the order to fit inqto your dashboard layout
    title='Classification Metrics',
    data=data_dict,
    x='feature_group',
    y="Value"
)
# Run the Shimoku Python code
s.run()

#%%


# Set the menu path
s.set_menu_path('Technical Test', '7. Model Analysis')

# Add the model analysis text
model_analysis_html = (
    '<h3>Model Analysis on Test Dataset</h3>'
    '<p>For unsuccessful leads (class \'0\'), we have a moderate recall of 66.60%, indicating room for improvement in identifying all unsuccessful leads.</p>'
    '<p>Importantly, for successful leads (class \'1\'), our model achieves a recall of 71.31%. This means it correctly identifies over 71% of the actual successful leads, which is crucial for our business as it ensures most potential opportunities are captured, minimizing the risk of missing out on valuable leads.</p>'
    '<p>In summary, the model is effective in capturing a significant portion of successful leads, making it a valuable tool in our lead qualification process, particularly in ensuring that most genuine opportunities are not overlooked.</p>'
)

s.plt.html(html=model_analysis_html, order=0, cols_size=22, rows_size=5, padding='1,1,1,1')


dicct_ = {'0': {'precision': 0.9534689328503761,
  'recall': 0.6660179057999221,
  'f1-score': 0.7842328406096024,
  'support': 5138.0},
 '1': {'precision': 0.19474425152510558,
  'recall': 0.7130584192439863,
  'f1-score': 0.3059343899741983,
  'support': 582.0},
 'accuracy': 0.6708041958041958,
 'macro avg': {'precision': 0.5741065921877408,
  'recall': 0.6895381625219542,
  'f1-score': 0.5450836152919003,
  'support': 5720.0},
 'weighted avg': {'precision': 0.8762700229672805,
  'recall': 0.6708041958041958,
  'f1-score': 0.7355668094435527,
  'support': 5720.0}}

# Transforming dicct_ into a plottable format
metrics_data = []
for key, value in dicct_.items():
    if key not in ['accuracy', 'macro avg', 'weighted avg', 'support']:
        # Exclude 'support' from each class
        class_metrics = {metric: val for metric, val in value.items() if metric != 'support'}
        class_metrics['class'] = key
        metrics_data.append(class_metrics)



# Plotting the classification metrics
s.plt.bar(
    order=2,  # Adjust the order to fit into your dashboard layout
    title='Classification Metrics',
    data=metrics_data,
    x='class',
    y=['precision', 'recall', 'f1-score']
)

# Run to update the dashboard
s.run()
