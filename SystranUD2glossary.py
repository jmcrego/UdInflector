#!/usr/bin/env python3
import sys
import time
import argparse

def load_ud_file(path):
    fms = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            term, pos = parts[0], parts[1] #discard third column (note)
            fms.append((term, pos))
    print(f"Loaded {len(fms)} entries from {path}", file=sys.stderr)
    return fms

if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="Convert two parallel Systran UD files to TSV glossary format.", formatter_class=argparse.RawTextHelpFormatter)
    parse.add_argument("ud1", help="First Systran UD file (e.g. source language)")
    parse.add_argument("ud2", help="Second Systran UD file (e.g. target language)")
    # add two comments at end of parse section
    parse.epilog = """- Systran UD line format is: ^term \\t (pos) \\t note$. 
- Example usage: python SystranUD2glossary.py resources/ud-enfr_en.dic resources/ud-enfr_fr.dic > resources/UD-enfr.tsv"""
    args = parse.parse_args()
    

    fms1 = load_ud_file(args.ud1)
    fms2 = load_ud_file(args.ud2)
    if len(fms1) != len(fms2):
        print("Warning: UD files have different number of entries", file=sys.stderr)
        sys.exit(1)

    for (term1, pos1), (term2, pos2) in zip(fms1, fms2):
        if pos1 != pos2:
            print(f"Warning: POS mismatch ({pos1} != {pos2}) for terms '{term1}' and '{term2}'", file=sys.stderr)  
            continue
        print(f"{term1}\t{pos1}\t{term1} ({pos1}) ||| {term2}")