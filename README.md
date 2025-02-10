# Document Search Engine

A Python-based search engine that uses TF-IDF (Term Frequency-Inverse Document Frequency) algorithm with autocorrect capabilities to find relevant documents based on search queries.

## Main Features

1. **Smart Word Importance Calculation**
   - Uses TF-IDF to determine word importance
   - TF (Term Frequency): How often word appears in document
   - IDF (Inverse Document Frequency): How unique word is across all documents
   - Common words get lower importance scores automatically

2. **Spelling Mistake Handling**
   - Uses Levenshtein distance for spell correction
   - Can handle typos like "serch" → "search"
   - Customizable threshold for word similarity

## How It Works

### Document Processing
1. **Text Cleaning (Preprocessing)**
   - Convert to lowercase
   - Remove punctuation marks (. , ! ? : ;)
   - Remove extra spaces

2. **Vocabulary Building**
   - Create list of unique words
   - Assign index number to each word
   - Store word-index pairs for quick lookup

3. **Word Counting**
   - Create frequency matrix
   - Count word appearances in each document
   - Store counts in efficient NumPy matrix

### Search Process
1. **Query Processing**
   - Clean input text
   - Fix spelling mistakes
   - Convert to word indices

2. **Score Calculation**
   - Calculate TF for each word
   - Calculate IDF across documents
   - Multiply TF and IDF for final score

3. **Result Ranking**
   - Sort documents by score
   - Return top 5 matches

## Limitations

Current implementation has some limitations:
- Doesn't understand word meaning
- Doesn't consider word order
- No context understanding
- Performance issues with large document sets

## Dependencies

- NumPy: For matrix operations and efficient calculations

## Usage Example

```python
# Create engine
search_engine = DocumentSearchEngine()

# Add documents
search_engine.addDocument("Your text here")

# Build frequency matrix
search_engine.build_freq_matrix_for_all_documents()

# Search (handles typos automatically)
# Example: "serch" will match "search"
```

## Future Improvements

- Better memory management
- Faster processing for large documents
- Improved edge case handling
- Enhanced spell correction
