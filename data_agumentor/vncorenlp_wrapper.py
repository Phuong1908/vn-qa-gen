from typing import List, Dict, Tuple, Optional
import py_vncorenlp
import logging
from underthesea import word_tokenize
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VnCorenlpHandler:
    def __init__(self, dir_path: str):
        """Initialize VnCoreNLP handler"""
        self.model = py_vncorenlp.VnCoreNLP(save_dir=dir_path)

    def get_word_mapping(self, sentence: str) -> Dict[Tuple[str, int], int]:
        """
        Create mapping between underthesea and VnCoreNLP tokenization
        Returns: Dict mapping (word, underthesea_idx) to vncorenlp_idx
        """
        try:
            # Get both tokenizations
            underthesea_tokens = word_tokenize(sentence)

            # Get VnCoreNLP tokenization
            annotated = self.model.annotate_text(sentence)
            if not annotated or 0 not in annotated:
                logger.error("No VnCoreNLP annotation results")
                return {}

            vncorenlp_tokens = [token['wordForm'] for token in annotated[0]]

            # Create mapping
            mapping = {}
            under_pos = 0
            vn_pos = 0

            while under_pos < len(underthesea_tokens) and vn_pos < len(vncorenlp_tokens):
                under_token = underthesea_tokens[under_pos]
                vn_token = vncorenlp_tokens[vn_pos]

                # Normalize tokens for comparison
                under_norm = under_token.replace(" ", "_")
                vn_norm = vn_token.replace(" ", "_")

                # Direct match after normalization
                if under_norm == vn_norm:
                    mapping[(under_token, under_pos)] = vn_pos
                    under_pos += 1
                    vn_pos += 1
                    continue

                # Try combining tokens from underthesea
                combined_under = under_norm
                temp_under_pos = under_pos + 1
                found_match = False

                while temp_under_pos < len(underthesea_tokens):
                    combined_under += "_" + \
                        underthesea_tokens[temp_under_pos].replace(" ", "_")
                    if combined_under == vn_norm:
                        for i in range(under_pos, temp_under_pos + 1):
                            mapping[(underthesea_tokens[i], i)] = vn_pos
                        under_pos = temp_under_pos + 1
                        vn_pos += 1
                        found_match = True
                        break
                    temp_under_pos += 1

                if found_match:
                    continue

                # Try combining tokens from VnCoreNLP
                combined_vn = vn_norm
                temp_vn_pos = vn_pos + 1
                found_match = False

                while temp_vn_pos < len(vncorenlp_tokens):
                    combined_vn += "_" + \
                        vncorenlp_tokens[temp_vn_pos].replace(" ", "_")
                    if combined_vn == under_norm:
                        mapping[(under_token, under_pos)] = vn_pos
                        under_pos += 1
                        vn_pos = temp_vn_pos + 1
                        found_match = True
                        break
                    temp_vn_pos += 1

                if not found_match:
                    # No match found, skip both tokens
                    under_pos += 1
                    vn_pos += 1

            return mapping

        except Exception as e:
            logger.error(
                f"Error creating word mapping: {str(e)}", exc_info=True)
            return {}

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

            # Create structured token information using defaultdict
            tokens = []
            for token in sent_data:
                token_info = defaultdict(str)
                token_info.update({
                    'index': token['index'],
                    'word': token['wordForm'],
                    'pos': token['posTag'],
                    'ner': token['nerLabel'],
                    'head': token['head'],
                    'dep_rel': token['depLabel']
                })
                tokens.append(dict(token_info))

            result = {
                'tokens': tokens,
                'text': sentence,
                'raw_annotation': sent_data,
                'word_mapping': self.get_word_mapping(sentence)
            }

            return result

        except Exception as e:
            logger.error(
                f"Error getting dependency info: {str(e)}", exc_info=True)
            return None

    def get_dependency_distance(self, sentence: str, start_idx: int, end_idx: int,
                                underthesea_tokens: List[str] = None) -> Optional[Tuple[int, List[str]]]:
        """Get dependency distance between two token positions"""
        try:
            dep_info = self.get_dependency_info(sentence)
            if not dep_info:
                return None

            tokens = dep_info['tokens']
            word_mapping = dep_info['word_mapping']

            # Map underthesea indices to VnCoreNLP indices
            if underthesea_tokens:
                try:
                    start_word = underthesea_tokens[start_idx]
                    end_word = underthesea_tokens[end_idx]

                    start_key = (start_word, start_idx)
                    end_key = (end_word, end_idx)

                    if start_key not in word_mapping:
                        logger.warning(
                            f"Could not map start word: {start_word}")
                        return abs(start_idx - end_idx), []

                    if end_key not in word_mapping:
                        logger.warning(f"Could not map end word: {end_word}")
                        return abs(start_idx - end_idx), []

                    start_idx = word_mapping[start_key]
                    end_idx = word_mapping[end_key]
                except IndexError:
                    logger.error("Index out of range when mapping tokens")
                    return None

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

            relations, directions = bfs_shortest_path(start_idx, end_idx)

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

    def __del__(self):
        """Cleanup when object is deleted"""
        try:
            self.model.close()
        except Exception as e:
            logger.error(f"Error closing model: {str(e)}")
