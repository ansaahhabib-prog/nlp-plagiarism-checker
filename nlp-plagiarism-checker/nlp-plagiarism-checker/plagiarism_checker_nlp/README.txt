# NLP-Based Plagiarism Checker

## Overview
This project implements a plagiarism detection system using classical Natural Language Processing (NLP) techniques. 
The system compares multiple text documents and computes similarity scores to identify potential plagiarism.

The project was developed as an academic AI/NLP project with a focus on understanding text similarity and document comparison.

## Methodology
The plagiarism detection pipeline consists of the following steps:
- Text preprocessing (lowercasing, punctuation removal)
- Tokenization and stopword removal
- Word frequency analysis
- Similarity computation between document pairs
- Generation of plagiarism similarity reports

## Project Structure
- src/ : Python source code
- data/ : Input text documents
- results/ : Generated similarity results

## Technologies Used
- Python
- NLP preprocessing techniques
- Basic similarity metrics
- File handling and text analysis

## How to Run
1. Clone the repository:
   git clone https://github.com/ansaahhabib-prog/nlp-plagiarism-checker
2. Navigate to the project directory
3. Place text files inside the `data/` folder
4. Run the script:
   python src/plagiarism_checker.py

## Results
The system generates a similarity report that shows plagiarism percentages between document pairs.
This helps in identifying potentially plagiarized content.

## Future Improvements
- Integration of TF-IDF and cosine similarity
- Use of semantic similarity models
- Support for larger document collections
- Visualization of similarity scores
