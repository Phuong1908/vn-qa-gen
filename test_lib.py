import py_vncorenlp
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VnCorenlpWrapper:
    def __init__(self, save_dir: str):
        """Initialize VnCoreNLP wrapper"""
        self.model = py_vncorenlp.VnCoreNLP(save_dir=save_dir)

    def parse_annotation(self, annotation: List[Dict]) -> List[Dict]:
        """Parse the annotation list into structured format"""
        sentences = []

        # Each item in the list represents a sentence
        for sentence in annotation:
            current_sentence = []

            # Process each token in the sentence
            for i in range(len(sentence)):
                token_info = {
                    'index': int(sentence[i][0]),  # Index
                    'word': sentence[i][1],        # Word
                    'pos': sentence[i][2],         # POS tag
                    'ner': sentence[i][3],         # NER tag
                    'head': int(sentence[i][4]),   # Head index
                    'dep_rel': sentence[i][5]      # Dependency relation
                }
                current_sentence.append(token_info)

            sentences.append(current_sentence)

        return sentences

    def get_dependency_path(self, sentence_tokens: List[Dict], start_idx: int, end_idx: int) -> Tuple[List[str], List[str]]:
        """Find dependency path between two tokens"""
        # Create adjacency list representation
        adj_list = {}
        for token in sentence_tokens:
            idx = token['index']
            head = token['head']
            if idx not in adj_list:
                adj_list[idx] = []
            if head not in adj_list:
                adj_list[head] = []

            # Add bidirectional edges with dependency relations
            adj_list[idx].append((head, token['dep_rel'], 'up'))
            adj_list[head].append((idx, token['dep_rel'], 'down'))

        # Find path using BFS
        def bfs_path(start: int, end: int):
            if start == end:
                return [], [], []

            visited = {start}
            queue = [(start, [], [], [])]  # node, relations, directions, words

            while queue:
                node, path_rels, path_dirs, path_words = queue.pop(0)

                for next_node, rel, direction in adj_list.get(node, []):
                    if next_node == end:
                        # Found path to target
                        return (
                            path_rels + [rel],
                            path_dirs + [direction],
                            path_words + [next_node]
                        )

                    if next_node not in visited:
                        visited.add(next_node)
                        queue.append((
                            next_node,
                            path_rels + [rel],
                            path_dirs + [direction],
                            path_words + [next_node]
                        ))
            return None, None, None

        # Get path
        relations, directions, path_indices = bfs_path(start_idx, end_idx)

        if not relations:
            return None, None

        # Convert indices to words
        index_to_word = {t['index']: t['word'] for t in sentence_tokens}
        path_words = [index_to_word[idx]
                      for idx in path_indices] if path_indices else []

        # Create path description
        path_desc = []
        for rel, direction in zip(relations, directions):
            path_desc.append(f"{rel}_{direction}")

        return path_desc, path_words

    def test_dependency_path(self, text: str, word1: str, word2: str):
        """Test dependency path between two words"""
        try:
            logger.info(f"\nProcessing text: {text}")

            # Get annotation
            annotated = self.model.annotate_text(text)

            # Convert the dictionary format to our expected format
            sentences = []
            for sent_idx, sent_data in annotated.items():
                current_sentence = []
                for token in sent_data:
                    token_info = {
                        'index': token['index'],
                        # Note: using 'wordForm' instead of previous format
                        'word': token['wordForm'],
                        # Note: using 'posTag' instead of previous format
                        'pos': token['posTag'],
                        # Note: using 'nerLabel' instead of previous format
                        'ner': token['nerLabel'],
                        'head': token['head'],
                        # Note: using 'depLabel' instead of previous format
                        'dep_rel': token['depLabel']
                    }
                    current_sentence.append(token_info)
                sentences.append(current_sentence)

            # Process each sentence
            for sent_idx, sentence in enumerate(sentences):
                logger.info(f"\nProcessing sentence {sent_idx + 1}:")
                logger.info(f"Tokens: {[t['word'] for t in sentence]}")

                # Find word indices
                word1_token = None
                word2_token = None

                for token in sentence:
                    if token['word'] == word1:
                        word1_token = token
                    if token['word'] == word2:
                        word2_token = token

                if word1_token and word2_token:
                    logger.info(f"\nFound words in sentence {sent_idx + 1}:")
                    logger.info(f"Word 1: {word1_token}")
                    logger.info(f"Word 2: {word2_token}")

                    # Get dependency path
                    path_desc, path_words = self.get_dependency_path(
                        sentence,
                        word1_token['index'],
                        word2_token['index']
                    )

                    if path_desc:
                        logger.info(f"\nDependency path:")
                        logger.info(f"Path description: {path_desc}")
                        logger.info(f"Path words: {path_words}")
                        return path_desc, path_words
                    else:
                        logger.info("No path found between words")
                        return None, None

            logger.info("Words not found in same sentence")
            return None, None

        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None


def main():
    try:
        # Initialize
        save_dir = '/Users/phuongnguyen/study/vn-qa-gen/vncorenlp'
        vncore = VnCorenlpWrapper(save_dir)

        # Test cases with corrected tokenization
        test_cases = [
            {
                "text": "Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội.",
                "word1": "Nguyễn_Khắc_Chúc",  # This is correct as per tokenization
                "word2": "làm_việc"           # This is correct as per tokenization
            },
            {
                "text": "Bà Lan, vợ ông Chúc, cũng làm việc tại đây.",
                "word1": "Lan",               # This is correct as per tokenization
                "word2": "làm_việc"           # This is correct as per tokenization
            }
        ]

        # Run tests
        for i, test in enumerate(test_cases, 1):
            logger.info(f"\n=== Test Case {i} ===")
            logger.info(f"Text: {test['text']}")
            logger.info(
                f"Finding path between: '{test['word1']}' and '{test['word2']}'")

            path_desc, path_words = vncore.test_dependency_path(
                test['text'],
                test['word1'],
                test['word2']
            )

            logger.info(f"Result for test case {i}:")
            logger.info(f"Path description: {path_desc}")
            logger.info(f"Path words: {path_words}")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
