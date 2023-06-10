import pandas

class RecommenderSystem:
    def __init__(self) -> None:
        self.dataset = None

    def train(self, dataset_name: str) -> None:
        self.dataset = pandas.read_csv("data.csv")

    # def recommend(self, id: str, num_items: int = 3) -> None:
