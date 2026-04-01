import json
import sys
import ast
import re
from collections import defaultdict
from SystranUD2glossary import fix_pos, fix_lem


def parseXML(file):
    term2inflections = defaultdict(list)

    inflections = set()

    for line in open(file, encoding="utf-8"):
        line = line.lower().strip()

        # detect: <inflected.*>implement</inflected>
        if re.match(r"<inflected.*>(.*)</inflected>", line):
            inflection = re.findall(r"<inflected.*>(.*)</inflected>", line)[0]
            inflections.add(inflection)
            continue


        # detect lem pos in : <source>royaume (adj)</source>
        elif re.match(r"<source>(.*) \(([^\)]*)\).*</source>", line):
            lem, pos = re.findall(r"<source>(.*) \(([^\)].*)\).*</source>", line)[0]
            lem = fix_lem(lem, pos)
            pos = fix_pos(pos)
            continue

        # detect: </entry>
        elif line == "</entry>":
            # pos = fix_pos(pos)
            # lem = fix_lem(lem, pos)
            term2inflections[f"{lem} ({pos})"] = inflections
            # print(f"XML Term: {lem} ({pos}) -> Inflections: {inflections}")
            inflections = set()
            continue


    print(f"(XML) Read {len(term2inflections)} terms")
    return term2inflections


def parseTSV(file):
    # caractériser (verb) ||| to characterize
    # ['caractériser', 'caractérisant', 'caractérisé', 'caractérisée', 'caractérisés', 'caractérisées']
    # idx: 0, French, verb, caractériser, infinitif / gérondif / participe

    term2inflections = defaultdict(list)

    for line in open(file, encoding="utf-8"):

        toks = line.lower().strip().split("\t")

        if len(toks) != 3:
            continue

        ud = toks[0] #caractériser (verb) ||| to characterize
        term = ud.split("|||")[0].strip() #caractériser (verb)
        inflections = ast.literal_eval(toks[1]) #['caractériser', 'caractérisant', 'caractérisé', 'caractérisée', 'caractérisés', 'caractérisées']
        # print(f"TSV Term: {term} -> Inflections: {inflections}")
        term2inflections[term] += inflections
    
    print(f"(TSV) Read {len(term2inflections)} terms")
    return term2inflections

def evaluate(ref2infl, hyp2infl, verbose=False):
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
    # total_int_inflections = 0
    # total_hyp_inflections = 0
    # total_ref_inflections = 0
    for term in intersection:
        infl_ref = set(ref2infl[term])
        infl_hyp = set(hyp2infl[term])
        total_tp += len(infl_ref & infl_hyp)
        total_fp += len(infl_hyp - infl_ref)
        total_fn += len(infl_ref - infl_hyp)
        # total_int_inflections += len(infl_ref & infl_hyp)
        # total_hyp_inflections += len(infl_hyp)
        # total_ref_inflections += len(infl_ref)

    global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0
    global_accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 1.0
    print(f"Global Acc: {global_accuracy:.3f} P: {global_precision:.3f} R: {global_recall:.3f} F1: {global_f1:.3f}")
#    print(f"Inflections: #hyp: {total_hyp_inflections} #ref: {total_ref_inflections} #intersection: {total_int_inflections} ({total_int_inflections/total_hyp_inflections:.3f}) ({total_int_inflections/total_ref_inflections:.3f})")

    if verbose:
        # Missing terms in hyps
        print("\n* Missing terms in hyps:")
        missing_terms = set(ref2infl.keys()) - set(hyp2infl.keys())
        print("  " + ", ".join(missing_terms))

        # Missing terms in refs
        print("\n* Missing terms in refs:")
        missing_terms = set(hyp2infl.keys()) - set(ref2infl.keys())
        print("  " + ", ".join(missing_terms))

        for term in set(ref2infl.keys()) | set(hyp2infl.keys()):
            if term not in ref2infl:
                ref2infl[term] = []
            if term not in hyp2infl:
                hyp2infl[term] = []
            Refs_noHyps = set(ref2infl[term]) - set(hyp2infl[term])
            Hyps_noRefs = set(hyp2infl[term]) - set(ref2infl[term])
            if len(Refs_noHyps) > 0 or len(Hyps_noRefs) > 0:
                print(f"\n* Term: {term}")
                if len(Refs_noHyps) > 0:
                    print(f"  Missing in hyps: {Refs_noHyps}")
                if len(Hyps_noRefs) > 0:
                    print(f"  Missing in refs: {Hyps_noRefs}")


# usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="The script evaluates the performance of an inflection generator by comparing its output against a reference dataset.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('refs_file', type=str, help='Path to the refs XML file (Systran codging engine output)')
    parser.add_argument('hyps_file', type=str, help='Path to the hyps TSV file (UDInflector output)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed evaluations for each term')
    parser.epilog = """
- The XML file should have entries with <source>term (pos)</source> and <inflected>inflection</inflected> tags.
- Example usage: python eval_inflector.py glossary.xml glossary.tsv
"""
    args = parser.parse_args()
    ref2infl = parseXML(args.refs_file)
    hyp2infl = parseTSV(args.hyps_file)
    evaluate(ref2infl, hyp2infl, args.verbose)
