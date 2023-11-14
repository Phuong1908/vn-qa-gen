# -*- coding: utf-8 -*-
"""
Metric function for question generation.
"""
from ignite.metrics import Bleu, RougeL

def result_to_file(f_gold, f_pred, gold, pred):
    """
    Write gold and predict sentences to files.
    :param f_gold: gold file name
    :param f_pred: predict file name
    :gold: 3d list, [[[gold 1 words]], [gold 2 words], ...]
    :pred: 2d list, [[pred 1 words], [pred 2 words], ...]
    """
    assert len(gold) == len(pred)
    fg = open(f_gold, "wb")
    fp = open(f_pred, "wb")
    for idx in range(len(gold)):
        gold_sentence = " ".join(gold[idx][0]).replace("\n", " ") + "\n"
        pred_sentence = " ".join(pred[idx]).replace("\n", " ") + "\n"
        fg.write(gold_sentence.encode("utf-8"))
        fp.write(pred_sentence.encode("utf-8"))
    fg.close()
    fp.close()


def compute_metrics_by_list(gold, pred, f_gold="gold.txt", f_pred="pred.txt"):
    result_to_file(f_gold, f_pred, gold, pred)
    metrics = compute_metrics_by_file([f_gold], f_pred)
    return metrics


def compute_metrics_by_file(references, hypothesis):
    """
    Given a list of gold file names and a predict result file,
    calculate metrics. Same line number corresponds to the same
    instance to calculate metric.
    Ref: https://github.com/Maluuba/nlg-eval
    :param references: list of gold file names.
    :param hypothesis: predict file name.
    :return: a list of metric results.
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (RougeL(), "ROUGE_L")]

    def _strip(s):
        return s.strip()

    with open(hypothesis, encoding='utf-8') as f:
        hyp_list = f.readlines()
    ref_list = []
    for iidx, reference in enumerate(references):
        with open(reference, encoding='utf-8') as f:
            ref_list.append(f.readlines())
    ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
    refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    assert len(refs) == len(hyps)

    ret_scores = {}

    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, hyps)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                # print("%s: %0.6f" % (m, sc))
                ret_scores[m] = sc
        else:
            # print("%s: %0.6f" % (method, score))
            ret_scores[method] = score

    return ret_scores


# def calc_style_acc(gold, predict):
#     predict_questions = [" ".join(q) for q in predict]
#     gold_questions = [" ".join(q[0]) for q in gold]
#     predict_types = []
#     gold_types = []
#     for i in range(len(predict_questions)):
#         type_str, type_id = get_question_type(predict_questions[i])
#         predict_types.append(type_id)
#         type_str, type_id = get_question_type(gold_questions[i])
#         gold_types.append(type_id)
#     acc = accuracy_score(gold_types, predict_types)
#     return acc


if __name__ == "__main__":
    gold = [["Which game console\nhehe could".split()],
            ["Which game c2222 ?".split()]]
    predict = ["Who does the CBS Sports apps".split(),
               "Which game c2222 ?".split()]
    print("result by compute_metrics_by_list: ", compute_metrics_by_list(
        gold, predict, f_gold="gold.txt", f_pred="pred.txt"))
