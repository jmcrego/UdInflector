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


# usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse XML glossary file and extract inflections as JSON.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('refs_file', type=str, help='Path to the hyps XML file')
    parser.add_argument('hyps_file', type=str, help='Path to the refs TSV file')
    parser.epilog = """- The XML file should have entries with <source>term (pos)</source> and <inflected>inflection</inflected> tags.
- Example usage: python parse_xml.py glossary.xml > inflections.jsonl
- Output will be a JSONL file with each line containing a JSON object with 'term' and 'inflections' (list of inflected forms)."""
    args = parser.parse_args()
    hyp2infl = parseXML(args.refs_file)
    ref2infl = parseTSV(args.hyps_file)

    # Count intersection of sets of terms in refs and hyps
    intersection = set(ref2infl.keys()) & set(hyp2infl.keys())
    union = set(ref2infl.keys()) | set(hyp2infl.keys())
    print(f"Total terms in refs: {len(ref2infl)}", file=sys.stderr)
    print(f"Total terms in hyps: {len(hyp2infl)}", file=sys.stderr)
    print(f"Total terms in intersection: {len(intersection)}", file=sys.stderr)
    print(f"Total terms in union: {len(union)}", file=sys.stderr)

    for term in sorted(intersection):
        infl_ref = set(ref2infl[term])
        infl_hyp = set(hyp2infl[term])
        if infl_ref != infl_hyp:
            #print inflections in ref not in hyp and vice versa
            print(f"Term: {term}", file=sys.stderr)
            print(f"  Inflections in ref not in hyp: {infl_ref - infl_hyp}", file=sys.stderr)
            print(f"  Inflections in hyp not in ref: {infl_hyp - infl_ref}", file=sys.stderr)

            # print(f"  Refs inflections: {infl_ref}", file=sys.stderr)
            # print(f"  Hyps inflections: {infl_hyp}", file=sys.stderr)

    # print terms not available in both sets
    for term in sorted(union - intersection):
        print(f"Term not available in both sets: {term}", file=sys.stderr)

