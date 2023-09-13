from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
# from test2 import ObjectDetection, UIElementCluster
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()


# class EmbeddingComparator:
#     def __init__(self):
#         self.embedder = OpenAIEmbeddings()

#     def generate_embedding(self, text):
#         logger.info('Generating embedding for text: %s', text)
#         return self.embedder.embed_query(text)

#     def parse_ui_string(self, ui_string):
#         # Split by comma and strip whitespace
#         return set(element.strip() for element in ui_string.split(','))

#     def average_embedding(self, elements):
#         embeddings = [self.generate_embedding(e) for e in elements]
#         return np.sum(embeddings, axis=0) / len(embeddings)

#     def validate_test_cases_semantically(self, test_cases, clusters, SOME_THRESHOLD):
#         logger.info('Validating test cases semantically.')
#         results = {}

#         # Convert the UI elements to average embeddings
#         test_case_embeddings = {desc: self.average_embedding(
#             elements) for desc, elements in test_cases}
#         cluster_embeddings = {cluster_id: self.average_embedding(
#             self.parse_ui_string(elements)) for cluster_id, elements in clusters.items()}

#         for desc, test_case_emb in test_case_embeddings.items():
#             similarities = {}
#             for cluster_id, cluster_emb in cluster_embeddings.items():
#                 if test_case_emb is not None and cluster_emb is not None:
#                     similarity = cosine_similarity(
#                         [test_case_emb], [cluster_emb])[0][0]
#                 else:
#                     similarity = 0
#                 similarities[cluster_id] = similarity

#             # Find the cluster with the highest similarity
#             best_match_cluster = max(similarities, key=similarities.get)

#             # Define SOME_THRESHOLD, e.g., 0.8
#             if similarities[best_match_cluster] > SOME_THRESHOLD:
#                 results[desc] = {'status': 'passed',
#                                  'best_matched_cluster': best_match_cluster}
#             else:
#                 results[desc] = {'status': 'failed'}

#         return results


# Assuming OpenAIEmbeddings class exists and works as expected
# If not, it might need adjustments

logger = logging.getLogger()


class EmbeddingComparator:
    def __init__(self):
        self.embedder = OpenAIEmbeddings()

    def generate_embedding(self, text):
        # logger.info('Generating embedding for text: %s', text)
        return self.embedder.embed_query(text)

    def parse_ui_string(self, ui_string):
        # Split by comma and strip whitespace
        return [element.strip() for element in ui_string.split(',')]

    def average_embedding(self, elements):
        embeddings = [self.generate_embedding(e) for e in elements if e]
        # Ensure that there are valid embeddings to average
        return np.mean(embeddings, axis=0) if embeddings else None

    def validate_test_cases_semantically(self, test_cases, clusters, SOME_THRESHOLD=0.8):
        logger.info('Validating test cases semantically.')
        results = {}

        # Convert the UI elements to average embeddings
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

            # Find the cluster with the highest similarity
            best_match_cluster = max(similarities, key=similarities.get)

            if similarities[best_match_cluster] > SOME_THRESHOLD:
                results[desc] = {'status': 'passed',
                                 'best_matched_cluster': best_match_cluster}
            else:
                results[desc] = {'status': 'failed'}

        return results

# Assuming OpenAIEmbeddings exists
# If you can provide more information about it, I can help better
# For now, let's test the code using dummy embeddings and clusters


# Test
# test_cases = {
#     "Test case: Ensure the page has a banner image with a call-to-action button.": ["button"],
#     "Test case: Ensure the page has a banner image with a call-to-action button.": ["image", "button"],
#     "Test case: Check if the Bloqor button is displayed.": ["button"],
#     "Test case: Confirm that the navigation bar has multiple links.": ["link", "link", "link"],
#     "Test case: Ensure the page has a button and a relevant heading about individual or professional choices.": ["button", "heading", "heading"]
# }

# detection_results = [
#     {"bbox": ..., "text": "banner", "class_name": "image", "confidence": 0.9},
#     {"bbox": ..., "text": "submit", "class_name": "button", "confidence": 0.8}
# ]


# clusters = {
#     0: 'button',
#     1: 'image, button',
#     2: 'button Bloqor',
#     3: 'button',
#     4: 'link, link, link, link, link, link',
#     5: 'button, heading Areyouanindividualor, heading professionalz'
# }
# comparator = EmbeddingComparator()
# # Validate the test cases
# results = comparator.validate_test_cases_semantically(
#     test_cases, clusters, 0.5)
# print(results)
