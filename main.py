import time

import pandas
import sklearn.feature_extraction.text
import sklearn.metrics.pairwise

class RecommenderSystem:
    def __init__(self) -> None:
        self.dataset = None
        self.data_store = {}

    def train(self, dataset_name: str) -> None:
        start_time = time.time()
        self.dataset = pandas.read_csv(dataset_name)
        print(f"{time.time() - start_time} seconds to read data")

        start_time = time.time()
        term_frequency = sklearn.feature_extraction.text.TfidfVectorizer(
            analyzer="word", ngram_range=(1, 3), min_df=0, stop_words="english"
        )
        term_frequency_inverse_document_frequency_matrix = term_frequency.fit_transform(
            self.dataset["description"]
        )

        cosine_similarities = sklearn.metrics.pairwise.linear_kernel(
            term_frequency_inverse_document_frequency_matrix,
            term_frequency_inverse_document_frequency_matrix,
        )

        for idx, row in self.dataset.iterrows():
            similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
            similar_items = [
                (cosine_similarities[idx][i], self.dataset["id"][i])
                for i in similar_indices
            ]
            self.data_store[row["id"]] = similar_items[1:]
        print(f"{time.time() - start_time} seconds to train")

    def recommend(self, item_id: str, num_recommendations: int = 3) -> None:
        return self.data_store[item_id][:num_recommendations]


recommender = RecommenderSystem()
recommender.train("data.csv")
