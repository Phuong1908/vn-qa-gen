from typing import List, Dict, Tuple, Optional
import py_vncorenlp
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_PATH = "/Users/phuongnguyen/study/vn-qa-gen/vncorenlp"


class VnCorenlpHandler:
    _instance = None
    _initialized = False

    def __new__(cls, dir_path: str = DEFAULT_PATH):
        if cls._instance is None:
            logger.info("Creating new VnCorenlpHandler instance")
            cls._instance = super(VnCorenlpHandler, cls).__new__(cls)
        return cls._instance

    def __init__(self, dir_path: str = DEFAULT_PATH):
        """Initialize VnCoreNLP handler"""
        if self._initialized:
            return

        try:
            logger.info(
                f"Initializing VnCoreNLP handler with path: {dir_path}")
            self.model = py_vncorenlp.VnCoreNLP(save_dir=dir_path)
            self._initialized = True
            logger.info("VnCoreNLP handler initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing VnCoreNLP: {str(e)}")
            raise

    @classmethod
    def get_instance(cls) -> 'VnCorenlpHandler':
        """Get or create singleton instance with default path"""
        if cls._instance is None:
            cls._instance = cls(DEFAULT_PATH)
        return cls._instance

    @staticmethod
    def process_vncorenlp_tokens(tokens: List[str]) -> List[str]:
        """Process VnCoreNLP tokens by replacing underscores with spaces"""
        return [t.replace('_', ' ') for t in tokens]

    def get_dependency_info(self, sentence: str) -> Optional[Dict]:
        """Get dependency parsing information for a sentence"""
        try:
            # Get annotation
            annotated = self.model.annotate_text(sentence)

            if not annotated or 0 not in annotated:
                logger.error("No annotation results")
                return None

            # Process first sentence
            sent_data = annotated[0]

            # Create structured token information using defaultdict and process tokens
            tokens = []
            for token in sent_data:
                token_info = defaultdict(str)
                token_info.update({
                    'index': token['index'],
                    'word': self.process_vncorenlp_tokens([token['wordForm']])[0],
                    'pos': token['posTag'],
                    'ner': token['nerLabel'],
                    'head': token['head'],
                    'dep_rel': token['depLabel']
                })
                tokens.append(dict(token_info))

            result = {
                'tokens': tokens,
                'text': sentence,
                'raw_annotation': sent_data
            }

            return result

        except Exception as e:
            logger.error(
                f"Error getting dependency info: {str(e)}", exc_info=True)
            return None

    def get_dependency_distance(self, sentence: str, start_idx: int, end_idx: int) -> Optional[Tuple[int, List[str]]]:
        """Get dependency distance between two token positions"""
        try:
            dep_info = self.get_dependency_info(sentence)
            if not dep_info:
                return None

            tokens = dep_info['tokens']

            # Create bidirectional adjacency list
            adj_list = defaultdict(list)
            for token in tokens:
                idx = token['index']
                head = token['head']
                rel = token['dep_rel']

                adj_list[idx].append((head, rel, 'up'))
                adj_list[head].append((idx, rel, 'down'))

            # Find shortest path using BFS
            def bfs_shortest_path(start: int, end: int) -> Tuple[Optional[List[str]], Optional[List[str]]]:
                if start == end:
                    return [], []

                visited = {start}
                queue = [(start, [], [])]

                while queue:
                    node, path_rels, path_dirs = queue.pop(0)

                    for next_node, rel, direction in adj_list[node]:
                        if next_node == end:
                            return path_rels + [rel], path_dirs + [direction]

                        if next_node not in visited:
                            visited.add(next_node)
                            queue.append((
                                next_node,
                                path_rels + [rel],
                                path_dirs + [direction]
                            ))

                return None, None

            # +1 because VnCoreNLP indices start at 1
            relations, directions = bfs_shortest_path(
                start_idx + 1, end_idx + 1)

            if relations is None:
                logger.warning("No dependency path found")
                return abs(start_idx - end_idx), []

            path_desc = [f"{rel}_{dir}" for rel,
                         dir in zip(relations, directions)]
            return len(relations), path_desc

        except Exception as e:
            logger.error(
                f"Error calculating dependency distance: {str(e)}", exc_info=True)
            return None
