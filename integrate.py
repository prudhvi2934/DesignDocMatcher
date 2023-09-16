import os
import numpy as np
import logging
from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()


class EmbeddingComparator:
    def __init__(self):
        """Initializes the comparator with OpenAIEmbeddings."""

        self.embedder = OpenAIEmbeddings()

    def generate_embedding(self, text):
        """
        Generates an embedding for a given text.

        Args:
        - text (str): The input text.

        Returns:
        - ndarray: The generated embedding.
        """

        return self.embedder.embed_query(text)

    def parse_ui_string(self, ui_string):
        """
        Splits the UI string by comma and strips whitespace.

        Args:
        - ui_string (str): The input UI string.

        Returns:
        - list: List of UI elements.
        """
        return [element.strip() for element in ui_string.split(',')]

    def average_embedding(self, elements):
        """
        Computes the average embedding for a list of elements.

        Args:
        - elements (list): The input list of elements.

        Returns:
        - ndarray or None: The average embedding, or None if the list is empty.
        """

        embeddings = [self.generate_embedding(e) for e in elements if e]
        return np.mean(embeddings, axis=0) if embeddings else None

    def validate_test_cases_semantically(self, test_cases, clusters, SOME_THRESHOLD=0.8):
        """
        Validates test cases based on semantic similarity with clusters.

        Args:
        - test_cases (list): List of test case descriptions.
        - clusters (dict): Dictionary with cluster IDs as keys and their 
            elements as values.
        - SOME_THRESHOLD (float, optional): Threshold for similarity. 
            Defaults to 0.8.

        Returns:
        - dict: Results with test case descriptions as keys and their status
            and best matched cluster (if passed) as values.
        """
        
        logger.info('Validating test cases semantically.')
        results = {}

        test_case_embeddings = {
            desc: self.generate_embedding(desc) for desc in test_cases}
        cluster_embeddings = {cluster_id: self.average_embedding(self.parse_ui_string(elements))
                              for cluster_id, elements in clusters.items()}

        for desc, test_case_emb in test_case_embeddings.items():
            similarities = {}
            for cluster_id, cluster_emb in cluster_embeddings.items():
                similarity = cosine_similarity([test_case_emb], [cluster_emb])[0][0] \
                    if test_case_emb is not None and cluster_emb is not None else 0
                similarities[cluster_id] = similarity

            best_match_cluster = max(similarities, key=similarities.get)

            if similarities[best_match_cluster] > SOME_THRESHOLD:
                results[desc] = {'status': 'passed',
                                 'best_matched_cluster': best_match_cluster}
            else:
                results[desc] = {'status': 'failed'}

        return results
