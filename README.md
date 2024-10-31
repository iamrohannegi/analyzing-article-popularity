# Analyzing and Predicting Article Popularity with Big Data Analytics

# Abstract 
For this project, we are using a dataset that has 39,797 articles from Mashable,a news website, published over two years. The dataset contains 58 features, like the length of the article, the day it was published, and some statistics about its content. The main goal of our project is to predict how many times each article will be shared on social media. This can help us understand what makes some articles more popular than others. We will use different data analysis methods, like classification and regression, to try to predict how many shares an article will get. Some of the techniques we’ll use include machine learning models like Random Forests and Neural Networks. We’ll also clean and process the data to make sure it’s ready for analysis and use graphs to see if we can find patterns in the data. This project will help us learn more about what factors make articles go viral and how businesses can use this information to make their content more shareable. With almost 40,000 articles and many features to analyze, this is a great dataset to practice big data techniques and learn how to work with large amounts of information.

# Analysis

A. Preprocessing 

In the preprocessing phase of this project, we first refined the dataset by cleaning and organizing the data for effective analysis. We began by ensuring that the column names were properly formatted, removing any unwanted spaces to standardize the field names. Additionally, rows where the content had zero tokens were filtered out, as these entries do not contribute any meaningful information for the analysis. This helped eliminate noise and improved the overall quality of the dataset. To better understand how the articles performed, we introduced a ”Popularity” measure, which classified articles based on how their number of shares compared to others in the dataset. This was achieved by calculating specific percentile ranges, where articles were categorized into different levelsof popularity: ”Very Poor,” ”Poor,” ”Average,” ”Good,” ”Very Good,” and ”Excellent.” This process allowed us to grade articles according to their relative success in terms of social media shares, providing a useful target variable for further analysis. Lastly, given the distribution of the ”shares” variable, which exhibited extreme outliers and skewed the data, we applied a logarithmic transformation to the number of shares. This transformation helped reduce the effect of highly skewed data, making patterns easier to identify and improving the performance of predictive models. This preprocessing step ensured that the data was ready for the subsequent stages of analysis and machine learning modeling. 

B. Statistical Analysis 

In this project, four key attributes from the dataset were selected for detailed statistical analysis. These attributes were chosen because they offer important insights into the structure and content of the articles and help in understanding factors that may influence article virality, as measured by the number of shares. The attributes selected were the number of words in the article’s content (n tokens content), the number of images (num imgs), the sentiment polarity of the content (global sentiment polarity), and the number of keywords (num keywords). 

Along with mean, mode, range, the other two metrics chosen for the analysis were standard deviation and median, both of which provide valuable insights into the dataset and its attributes. Standard deviation is particularly useful for understanding the spread or variability in the data. For attributes such as ”n tokens content,” ”num imgs,” ”global sentiment polarity,” and ”num keywords,” there is likely significant variation between articles. Standard deviation helps to quantify this variation by showing how much these values deviate from the mean. This is critical in identifying outliers and understanding the consistency or volatility in these attributes. For example, a high standard deviation in the number of images could suggest that certain articles rely heavily on visual content, while others may not, which could affect how widely they are shared. 

On the other hand, median is a robust measure of central tendency, less sensitive to outliers than the mean. Given that some articles may have unusually high or low values for shares or content length, the median provides a more accurate representation of the ”typical” article. For attributes such as ”shares” and ”n tokens content,” which may have skewed distributions due to extreme values, the median helps in understanding the central distribution of the data. This is especially relevant for identifying trends in article virality, where a few articles may achieve disproportionately high shares, but the median provides insight into the broader pattern of engagement. The range of the number of tokens in the content spans from 18 to 8474, indicating substantial variability in article length. The mean number of tokens is approximately 563, with a median of 423, suggesting that many articles tend to be shorter, but there are also significantly longer articles skewing the distribution. The standard deviation of 468 further supports this, showing a broad spread around the mean. 

In terms of number of images, the range is from 0 to 128, with a mean of about 4.56 and a median of 1, indicating that most articles include only one or very few images, while a smaller subset has a significantly higher image count. The large standard deviation of 8.3 suggests considerable variation in the number of images across articles. The global sentiment polarity values range from -0.39 to 0.73, with the mean and median both hovering around 0.12, indicating that the articles tend to have a slightly positive sentiment on average. The relatively small standard deviation (0.09) implies that most articles’ sentiment does not vary dramatically from this slight positivity. 

For number of keywords, the range is from 1 to 10, with a mean of around 7.21 and a median of 7. The mode is also 7, suggesting that many articles are optimized with this number of keywords. The standard deviation of approximately 1.92 shows a moderate level of variation, indicating that some articles use significantly fewer or more keywords than the norm. 


# VISUALIZATION 

The relationship between article popularity and average shares was visualized by grouping the data into popularity categories and calculating the mean number of shares for each. 

Fig. 1. Average Shares by Popularity Category 

![Figure_1](https://github.com/user-attachments/assets/0923c26d-c64b-4f33-8efa-7eff9f82d9fd)

The bar graph shows a clear upward trend: articles rated ”Very Poor” averaged 772 shares, while those in the ”Poor” and ”Average” categories had 1,234 and 1,717 shares, respectively. A notable increase is seen in the ”Good” and ”Very Good” categories, with averages of 2,634 and 4,529 shares. The ”Excellent” category stands out with a significant jumpto over 18,000 shares, indicating that higher popularity leads to exponentially greater social media engagement. 

The second graph visualizes the distribution of articles across different content categories and their corresponding popularity levels. The categories—Lifestyle, Entertainment, Business, Social Media, Tech, and World—are plotted on the x-axis, with the number of articles within each category on the y-axis. The graph uses color coding to represent different levels of article popularity, from ”Very Poor” to ”Excellent.” The ”Very Poor” popularity category dominates most content 


Fig. 2. Number of Articles by Content Category and Popularity 
![Figure_2](https://github.com/user-attachments/assets/6abfd993-8185-4780-bd4e-2ef8cf8d5f3e)


areas, particularly in Entertainment, Business, and World, indicating that a significant portion of content in these domains struggles to gain substantial engagement. The Entertainment category, despite having the highest number of articles overall, exhibits a wide range of success, with noticeable peaks in both the ”Good” and ”Very Good” popularity levels. The Tech and World categories show a high concentration of ”Very Poor” articles, though Tech content demonstrates a more balanced distribution across all popularity levels, suggesting its potential for higher success. The Social Media category, while having the fewest articles, performs relatively well, with most content in the ”Poor” to ”Good” range. In summary, Entertainment and World dominate in volume but underperform in popularity, whereas Tech stands out for its capacity to achieve broader success. 

The next graph is a scatter plot that illustrates the relationship between the number of tokens (content length) and the logarithm of the number of shares, with different colors representing varying levels of popularity The concentration of articles in the ”Very Poor” and ”Poor” popularity categories, predominantly seen in shorter articles (fewer tokens), suggests that shorter content is less likely to achieve significant shares. In contrast, articles within the ”Good”, ”Very Good”, and ”Excellent” categories are more evenly distributed across varying content lengths, with longer articles being more frequently associated with higher popularity outcomes. As content length increases, there is a logarithmic increase in the number of shares, especially in higher popularity 

Fig. 3. Number of Tokens in Content vs. Log of Shares 
![Figure_3](https://github.com/user-attachments/assets/32364158-0543-4e5d-bbfe-363dace566d9)


categories, reflecting a broader variance in shares for longer articles. A non-linear relationship is observed, where articles with moderate lengths (1000–3000 tokens) exhibit a wide range of popularity, while very long articles (over 5000 tokens) are rare but can occur across different popularity levels. This suggests that although longer articles tend to perform better, shorter articles consistently face challenges in achieving substantial popularity. 


The fourth figure is a pair plot that is visualizing the relationships between the global rate of positive and negative words and the number of shares, colored by article popularity. The global rate of positive words is primarily concentrated 


Fig. 4. Pair Plot of Global Positive/Negative Words and Shares by Popularity
![Figure_4](https://github.com/user-attachments/assets/bdf7684c-d91e-4b9b-9c93-738afff1bac6)


between 0.00 and 0.05, with higher-density clusters in lower popularity articles (”Very Poor” and ”Poor”). Although higher popularity articles (”Good” to ”Excellent”) also exhibit positive word usage in this range, articles with very high rates of positive words (0.10) are rare but tend to be more popular. Negative word distribution is broader, with lower popularityarticles containing higher rates of negative words. Articles with fewer negative words (closer to 0) are more likely to achieve higher popularity, suggesting that excessive negativity reduces engagement. No clear linear relationship is evident between shares and the rate of positive or negative words. Articles with high shares typically maintain a moderate rate of positive words (0.01–0.05) and low to moderate levels of negative words. In summary, a balanced use of positive words and minimal negative words correlates with higher popularity and shares, whereas excessive negative language is linked to lower performance. 


The final figure is a scatter plot that is visualizing the relationship between the number of images in an article and the logarithmic transformation of the number of social media shares. 

Fig. 5. Number of Images vs. Log of Shares 
![Figure_5](https://github.com/user-attachments/assets/8f1b3855-095e-4bdf-9c9e-356d51bb57ec)


The data reveals that while articles with fewer images (under 20) are distributed across all popularity levels, a higher proportion of less popular articles (in blue) is seen in this range. Articles with more than 20 images are more evenly distributed across popularity levels, with a notable presence in the higher popularity categories (in red), suggesting that including more images might be linked to greater social media success. However, despite this general trend, the logtransformed shares indicate that articles can achieve high popularity regardless of the number of images, and there is no strict linear relationship. The color gradient also highlights that the most popular articles tend to have shares above a log value of 10, while less popular articles cluster below 6. There are a few outliers with many images but low popularity, suggesting that other factors, beyond image count, contribute to virality. Overall, the findings imply that while the number of images can influence social media shares, it is not the sole factor driving an article’s popularity. 



# REFERENCES 

[1] K. Fernandes, P. Vinagre, P. Cortez, and P. Sernadela. ”Online News Popularity,” UCI Machine Learning Repository, 2015. [Online]. Available: https://doi.org/10.24432/C5NS3V.
