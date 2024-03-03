
# Wiki-CS GNN-based Article Recommender
## Project Outline
<ul>
  <li>Our dataset contains CS-related Wikipedia articles with their content summarized into text embeddings. These articles are already categorized into different topics of Computer Science (CS-Subtopics).</li>
  <li>We first create a model to predict an article's correct CS-Subtopic, given the content text embedding as an input.</li>
  <li>We use GNN explainer to find key motifs (subgraphs that summarize the overarching graph) and analyze their connections based on content similarity. By color-coding nodes by subtopic, we visualize subtopic relationships.</li>
</ul>

## Installation
Before running the Jupyter Notebook, ensure you have PyTorch and Pytorch Geometric installed. You can install it via pip:

<code>pip install torch</code>

<code>pip install torch-geometric</code>

#### Dataset source: [WikiCS Dataset](https://github.com/pmernyei/wiki-cs-dataset)

## Conclusion
This project demonstrates the application of Graph Neural Networks in classifying CS-related Wikipedia articles into various topics, obtaining an accuracy of around <em>70%</em> across the associated GNN models.
Additionally, we utilize GNNExplainer to interpret the predictions made by our model, providing insights into the importance of different features in the classification process.
