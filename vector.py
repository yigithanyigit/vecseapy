import numpy as np

class DocumentSearchEngine:

    def __init__(self):
        self.vocab = {}
        self.vocab_idx = 0
        self.vocab_letters = {}
        self.document_vectors =  []
        self.document_freq_matrix = np.zeros((0, 0))

    def preprocess(self, document):
        return document.lower().replace(",", "").replace(".", "").replace("!", "").replace("?", "").replace(":", "").replace(";", "").strip()

    def addDocument(self, document, add_to_vectors=True) -> np.array:
        document = self.preprocess(document)
        for word in document.split():
            if self.vocab.get(word, None) == None:
                self.vocab[word] = self.vocab_idx
                self.vocab_idx += 1

        self.populate_vocab()

        document_vector = self._generate_vector(document)
        if add_to_vectors:
            # self.document_vectors = np.append(self.document_vectors, document_vector)
            self.document_vectors.append(document_vector)
        return document_vector

    def populate_vocab(self):
        for key, _ in self.vocab.items():
            self.vocab_letters[key] = [l for l in key]

    def autocorrect(self, word):
        if self.vocab.get(word, None) != None:
            return word

        for key, _ in self.vocab_letters.items():
            res = self._levenshtein_distance(word, key)

            if res <= max(len(word) // 3, 2):
                return key

    def _levenshtein_distance(self, s, t):
        if len(s) == 0:
            return len(t)
        if len(t) == 0:
            return len(s)
        if s[0] == t[0]:
            return self._levenshtein_distance(s[1:], t[1:])

        a = self._levenshtein_distance(s[1:], t)
        b = self._levenshtein_distance(s, t[1:])
        c = self._levenshtein_distance(s[1:], t[1:])

        return 1 + min(a, b, c)

    def buildFreqMatrix(self, document_vector, add_to_matrix=True):
        matrix = self._generate_freq_matrix(document_vector)
        if add_to_matrix:
            if self.document_freq_matrix.size == 0:
                self.document_freq_matrix = matrix
            else:
                assert self.document_freq_matrix.shape[1] == matrix.shape[1], "Matrix column size does not match, maybe try adding all the documents first?"
                self.document_freq_matrix = np.append(self.document_freq_matrix, matrix, axis=0)
        return matrix

    def build_freq_matrix_for_all_documents(self):
        for document_vector in self.document_vectors:
            self.buildFreqMatrix(document_vector)

    def _generate_vector(self, document):
        vector = []
        for word in document.split():
            vector.append(self.vocab[word])

        return np.array(vector)

    def _generate_freq_matrix(self, document_vector):
        columns = len(self.vocab)
        matrix = np.zeros((1, columns))
        u, c = np.unique(document_vector, return_counts=True)
        dict_freq = dict(zip(u, c))

        for key, value in dict_freq.items():
            matrix[0][key] = value

        return matrix

    def tf(self, document_freq_matrix, t):
        vector = np.zeros((1, document_freq_matrix.shape[0]))
        for idx, freq_matrix in enumerate(document_freq_matrix):
            if self.vocab.get(t, None) != None:
                # vector[0][idx] = 1 + np.log10(freq_matrix[self.vocab[t]]) if freq_matrix[self.vocab[t]] > 0 else 0
                vector[0][idx] = freq_matrix[self.vocab[t]] / np.sum(freq_matrix)
        return vector

    def calculate_idf(self, document_freq_matrix: np.matrix, t) -> np.array:
        def df(freq_matrix, t):
            document_freq_vector = np.zeros((1, freq_matrix.shape[0])) # 1D row vector for calculating document frequency for t
            if freq_matrix[t] > 0:
                # document_freq_vector[0][t] = freq_matrix[t]
                document_freq_vector[0][t] = 1
            return document_freq_vector

        def idf(freq_matrix, t):
            sum = np.sum(df(freq_matrix, t))
            return np.log10(freq_matrix.shape[0] / sum) if sum > 0 else 0

        document_idf_matrix = np.zeros((0, 0))
        for freq_matrix in document_freq_matrix:
            document_idf_matrix = np.append(document_idf_matrix, idf(freq_matrix, self.vocab.get(t, 0)))

        return document_idf_matrix

    def tf_idf(self, t):
        return self.tf(self.document_freq_matrix, t) * self.calculate_idf(self.document_freq_matrix, t)


if __name__ == "__main__":
    import documents

    vocab_vector = DocumentSearchEngine()

    for key, value in documents.documents.items():
        vocab_vector.addDocument(value)


    input_keyword = None
    input_keyword = input("Please write keyword to search: ")

    assert input_keyword != None, "Please write a keyword to search"

    vocab_vector.build_freq_matrix_for_all_documents()
    score = np.zeros((1, vocab_vector.document_freq_matrix.shape[0]))

    for keyword in [vocab_vector.autocorrect(word) for word in vocab_vector.preprocess(input_keyword).split()]:
        score += vocab_vector.tf_idf(keyword)

    score = score.flatten()
    idx_score = np.argsort(score)[::-1]

    for idx in idx_score[:5]:
        if score[idx] > 0:
            print(documents.documents[idx])
            print(score[idx])
            print("\n")
