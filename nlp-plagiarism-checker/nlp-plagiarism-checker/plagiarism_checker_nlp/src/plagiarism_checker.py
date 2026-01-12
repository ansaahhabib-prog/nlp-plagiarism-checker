# plagiarism_checker.py
import os
import re
import math
import string
from collections import Counter, defaultdict

class PlagiarismChecker:
    """Plagiarism checker with NLP techniques."""
    
    def __init__(self):
        # Common English stopwords
        self.stopwords = {
            'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for', 'yet', 'so',
            'in', 'on', 'at', 'by', 'to', 'from', 'up', 'down', 'of', 'with',
            'about', 'against', 'between', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'is', 'am', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
            'doing', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'
        }
    
    def read_file(self, filepath):
        """Read text from file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read()
    
    def preprocess(self, text):
        """Apply NLP preprocessing: lowercase, remove punctuation/numbers, tokenize."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Split into words
        words = text.split()
        
        # Remove stopwords and short words
        words = [word for word in words if word not in self.stopwords and len(word) > 2]
        
        return words
    
    def create_bigrams(self, words):
        """Create word bigrams for better phrase matching."""
        if len(words) < 2:
            return words
        
        bigrams = []
        for i in range(len(words) - 1):
            bigrams.append(f"{words[i]}_{words[i+1]}")
        
        return bigrams
    
    def compute_tfidf(self, all_docs_words):
        """Compute TF-IDF vectors for all documents."""
        # Compute TF (Term Frequency) for each document
        tfs = []
        for words in all_docs_words:
            total = len(words)
            tf = Counter(words)
            # Normalize
            tf = {word: count/total for word, count in tf.items()}
            tfs.append(tf)
        
        # Compute IDF (Inverse Document Frequency)
        total_docs = len(all_docs_words)
        idf = {}
        
        # Count documents containing each word
        doc_freq = defaultdict(int)
        for words in all_docs_words:
            unique_words = set(words)
            for word in unique_words:
                doc_freq[word] += 1
        
        # Calculate IDF with smoothing
        for word, freq in doc_freq.items():
            idf[word] = math.log((total_docs + 1) / (freq + 1)) + 1
        
        # Compute TF-IDF
        tfidf_vectors = []
        for tf in tfs:
            tfidf = {}
            for word, tf_value in tf.items():
                tfidf[word] = tf_value * idf.get(word, 0)
            tfidf_vectors.append(tfidf)
        
        return tfidf_vectors
    
    def jaccard_similarity(self, words1, words2):
        """Calculate Jaccard similarity between two sets of words."""
        set1 = set(words1)
        set2 = set(words2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two TF-IDF vectors."""
        # Get all words from both vectors
        all_words = set(vec1.keys()) | set(vec2.keys())
        
        # Calculate dot product
        dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in all_words)
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(value ** 2 for value in vec1.values()))
        mag2 = math.sqrt(sum(value ** 2 for value in vec2.values()))
        
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def ngram_similarity(self, text1, text2, n=3):
        """Calculate character n-gram similarity."""
        def get_ngrams(text, n):
            return [text[i:i+n] for i in range(len(text) - n + 1)]
        
        ngrams1 = set(get_ngrams(text1.lower(), n))
        ngrams2 = set(get_ngrams(text2.lower(), n))
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union
    
    def check_all_documents(self, directory):
        """Compare all text files in directory with each other."""
        # Get all text files
        files = []
        for file in os.listdir(directory):
            if file.lower().endswith(('.txt', '.md', '.rtf')):
                files.append(os.path.join(directory, file))
            elif '.' not in file:  # Files without extensions
                files.append(os.path.join(directory, file))
        
        if len(files) < 2:
            print(f"Need at least 2 text files. Found {len(files)} file(s).")
            return []
        
        print(f"Found {len(files)} documents in '{directory}':")
        
        # Read and process all documents
        file_names = []
        all_texts = []
        all_words = []
        
        for file_path in files:
            text = self.read_file(file_path)
            words = self.preprocess(text)
            
            # Create bigrams for better phrase matching
            bigrams = self.create_bigrams(words)
            combined_tokens = words + bigrams
            
            file_names.append(os.path.basename(file_path))
            all_texts.append(text)
            all_words.append(combined_tokens)
            
            print(f"  - {file_names[-1]} ({len(words)} words)")
        
        # Compute TF-IDF vectors
        tfidf_vectors = self.compute_tfidf(all_words)
        
        # Compare all pairs
        print(f"\nComparing {len(files)} documents ({len(files)*(len(files)-1)//2} comparisons)...")
        results = []
        
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                # Calculate multiple similarity measures
                jaccard = self.jaccard_similarity(all_words[i], all_words[j])
                cosine = self.cosine_similarity(tfidf_vectors[i], tfidf_vectors[j])
                ngram = self.ngram_similarity(all_texts[i], all_texts[j])
                
                # Combined similarity (weighted average)
                similarity = (jaccard * 0.3 + cosine * 0.5 + ngram * 0.2)
                similarity_percent = round(similarity * 100, 2)
                
                results.append({
                    'file1': file_names[i],
                    'file2': file_names[j],
                    'similarity_percent': similarity_percent,
                    'word_count1': len(all_words[i]),
                    'word_count2': len(all_words[j])
                })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity_percent'], reverse=True)
        
        return results
    
    def save_results(self, results, output_file="plagiarism_results.txt"):
        """Save results to a file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("PLAGIARISM CHECK RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total comparisons: {len(results)}\n\n")
            
            f.write("SIMILARITY PERCENTAGES (sorted highest to lowest):\n")
            f.write("-" * 60 + "\n\n")
            
            for result in results:
                f.write(f"{result['file1']} vs {result['file2']}: {result['similarity_percent']}%\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("MOST SIMILAR DOCUMENTS:\n")
            f.write("-" * 60 + "\n\n")
            
            # Show top 10 most similar
            top_n = min(10, len(results))
            for i, result in enumerate(results[:top_n], 1):
                f.write(f"{i}. {result['file1']} vs {result['file2']}: {result['similarity_percent']}%\n")
            
            # Summary
            high = sum(1 for r in results if r['similarity_percent'] >= 80)
            medium = sum(1 for r in results if 50 <= r['similarity_percent'] < 80)
            low = sum(1 for r in results if r['similarity_percent'] < 50)
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("SUMMARY:\n")
            f.write("-" * 60 + "\n")
            f.write(f"High similarity (≥80%): {high} pair(s)\n")
            f.write(f"Medium similarity (50-80%): {medium} pair(s)\n")
            f.write(f"Low similarity (<50%): {low} pair(s)\n")
    
    def print_results(self, results):
        """Print results to console."""
        if not results:
            return
        
        print(f"\n{'='*60}")
        print("RESULTS")
        print('='*60)
        
        print(f"\nTotal comparisons: {len(results)}")
        print(f"\nSimilarity percentages (sorted highest to lowest):")
        print('-'*60)
        
        for result in results:
            print(f"{result['file1']} vs {result['file2']}: {result['similarity_percent']}%")
        
        print(f"\n{'='*60}")
        print("TOP 5 MOST SIMILAR DOCUMENTS:")
        print('-'*60)
        
        for i, result in enumerate(results[:5], 1):
            print(f"{i}. {result['file1']} vs {result['file2']}: {result['similarity_percent']}%")
        
        # Count by similarity level
        high = sum(1 for r in results if r['similarity_percent'] >= 80)
        if high > 0:
            print(f"\n⚠️  Found {high} potential plagiarism case(s) (≥80% similarity)")

def main():
    """Main function."""
    print("=" * 60)
    print("PLAGIARISM CHECKER WITH NLP TECHNIQUES")
    print("=" * 60)
    print("Compares all documents in a directory\n")
    
    # Get directory path
    directory = input("Enter directory path with documents (or press Enter for 'documents'): ").strip()
    
    if not directory:
        directory = "documents"
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"\nDirectory '{directory}' not found!")
        create = input(f"Create '{directory}' folder? (y/n): ").lower()
        if create == 'y':
            os.makedirs(directory)
            print(f"\nCreated '{directory}' folder.")
            print(f"Please add your text files to '{directory}' and run again.")
            return
        else:
            print("Please specify an existing directory.")
            return
    
    # Create checker and run analysis
    checker = PlagiarismChecker()
    
    print(f"\nAnalyzing documents in '{directory}'...")
    results = checker.check_all_documents(directory)
    
    if not results:
        return
    
    # Save results to file
    output_file = "plagiarism_results.txt"
    checker.save_results(results, output_file)
    
    # Print results to console
    checker.print_results(results)
    
    print(f"\nDetailed results saved to: {output_file}")
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

# Simple version for quick use
def quick_check(directory="documents"):
    """Quick function to check all documents in a directory."""
    checker = PlagiarismChecker()
    results = checker.check_all_documents(directory)
    
    if results:
        print(f"\nSimilarity percentages:")
        for result in results:
            print(f"{result['file1']} vs {result['file2']}: {result['similarity_percent']}%")
        
        # Save to file
        with open("results.txt", "w") as f:
            for result in results:
                f.write(f"{result['file1']} vs {result['file2']}: {result['similarity_percent']}%\n")
        
        print(f"\nResults saved to results.txt")

if __name__ == "__main__":
    main()
    # Or for quick use: quick_check("your_folder_name")