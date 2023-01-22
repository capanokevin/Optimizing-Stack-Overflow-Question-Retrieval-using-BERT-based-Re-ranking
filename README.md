# Optimizing Stack Overflow Question Retrieval using BERT-based Re-ranking
This project presents a novel approach for an information retrieval system in the domain of Stack Overflow question-answers. 
The proposed system consists of two main parts: a first answer retrieval phase and a re-ranking phase. The base information retrieval system is designed to take a Stack Overflow question as input and return the most relevant answers for that question, using Word2Vec centroids. 
To improve the performance of the IR system, a re-ranking phase is performed using a fine-tuned BERT model, trained with negative sampling on a relevant-non relevant binary task. The re-ranking is used to boost the performance of the IR system, which is evaluated using the Recall@5, 10, and 20. 
The proposed system achieved promising results, demonstrating the effectiveness of this approach.

You can download the dataset at this [link](https://drive.google.com/file/d/1ezE_WWssoO5NkyTXLgK4ZwFTv9p7qauA/view?usp=share_link).
