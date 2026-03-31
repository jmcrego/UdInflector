import json
import sys
import re
from collections import defaultdict

def parseXML(file):

    total_inflections = 0
    total_terms = 0
    total_repetitions = 0

    term2inflections = defaultdict()

    curr_term = None
    curr_inflections = set()

    for line in open(file, encoding="utf-8"):
        line = line.strip()

        # detect: <inflected.*>implement</inflected>
        if re.match(r"<inflected.*>(.*)</inflected>", line):
            inflection = re.findall(r"<inflected.*>(.*)</inflected>", line)[0]
            curr_inflections.add(inflection)

        # detect: <source>implement (verb)</source>
        elif re.match(r"<source>(.*)</source>", line):
            curr_term = re.findall(r"<source>(.*)</source>", line)[0]
            continue

        # detect: </entry>
        elif line == "</entry>":
            total_inflections += len(curr_inflections)
            total_terms += 1

            # print(out)
            if curr_term in term2inflections:
                if term2inflections[curr_term] != curr_inflections:
                    print(f"Repeated term: {curr_term} => {term2inflections[curr_term]}\n new: {curr_inflections}", file=sys.stderr)
                    total_repetitions += 1
            term2inflections[curr_term] = curr_inflections

            curr_term = None
            curr_inflections = set()
            total_terms += 1

    print(f"Total terms: {total_terms}", file=sys.stderr)
    print(f"Total inflections: {total_inflections}", file=sys.stderr)
    print(f"Total repeated terms: {total_repetitions}", file=sys.stderr)
    return term2inflections


def parseTSV(file):

    total_inflections = 0
    total_terms = 0

    term2inflections = defaultdict()

    for line in open(file, encoding="utf-8"):
        toks = line.strip().split("\t")
        if len(toks) != 4:
            continue
        inflections = toks[2].split(";")
        term2inflections[f"{toks[0]} ({toks[1]})"] = inflections
        total_terms += 1
        total_inflections += len(inflections)
    
    print(f"Total terms: {total_terms}", file=sys.stderr)
    print(f"Total inflections: {total_inflections}", file=sys.stderr)
    return term2inflections

def evaluate(hyp2infl, ref2infl, verbose=False):
    # Count intersection of sets of terms in refs and hyps
    intersection = set(ref2infl.keys()) & set(hyp2infl.keys())
    union = set(ref2infl.keys()) | set(hyp2infl.keys())
    print(f"#terms in refs: {len(ref2infl)}")
    print(f"#terms in hyps: {len(hyp2infl)}")
    print(f"#intersection: {len(intersection)}")
    print(f"missing terms in hyps: {len(set(ref2infl.keys()) - set(hyp2infl.keys()))}")

    # Compute global Precision, Recall, F1 over all terms in the intersection
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for term in intersection:
        infl_ref = set(ref2infl[term])
        infl_hyp = set(hyp2infl[term])
        total_tp += len(infl_ref & infl_hyp)
        total_fp += len(infl_hyp - infl_ref)
        total_fn += len(infl_ref - infl_hyp)

    global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0
    print(f"Global Precision: {global_precision:.3f}")
    print(f"Global Recall: {global_recall:.3f}")
    print(f"Global F1: {global_f1:.3f}")

    if verbose:
        for term in set(ref2infl.keys()) | set(hyp2infl.keys()):
            print(f"Term: {term}")
            if term not in ref2infl:
                ref2infl[term] = []
            if term not in hyp2infl:
                hyp2infl[term] = []
            print(f"  Refs & !Hyps: {set(ref2infl[term]) - set(hyp2infl[term])}")
            print(f"  Hyps & !Refs: {set(hyp2infl[term]) - set(ref2infl[term])}")


# usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="The script evaluates the performance of an inflection generator by comparing its output against a reference dataset.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('refs_file', type=str, help='Path to the refs XML file (Systran codging engine output)')
    parser.add_argument('hyps_file', type=str, help='Path to the hyps TSV file (UDInflector output)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed comparisons for each term')
    parser.epilog = """
- The XML file should have entries with <source>term (pos)</source> and <inflected>inflection</inflected> tags.
- Example usage: python eval_inflector.py glossary.xml glossary.tsv
"""
    args = parser.parse_args()
    hyp2infl = parseXML(args.refs_file)
    ref2infl = parseTSV(args.hyps_file)
    evaluate(hyp2infl, ref2infl, args.verbose)
