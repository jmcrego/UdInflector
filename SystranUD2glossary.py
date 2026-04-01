#!/usr/bin/env python3
import sys
import argparse

def fix_pos(pos):
    pos = pos.lower()
    if pos.startswith("proper noun"):
        return "proper noun"
    return pos

def fix_lem(lem, pos, language):
    if language == "en" and pos == "verb" and lem.startswith("to "):
        lem = lem[3:] # remove "to " from verb infinitive form in English (e.g. "to speak" -> "speak")
    if language == "fr" and pos == "verb" and lem.startswith("se "):
        lem = lem[3:] # remove "se " from verb infinitive form in French (e.g. "se laver" -> "laver")
    return lem


def load_tsv_file(path, language=None):
    fms = []
    POSes = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            #Anna proper noun        feminine        {4845}
            if language == "fr" and parts[0] == "Anna proper noun" and parts[1] == "feminine":
                parts[0] = "Anna"
                parts[1] = "proper noun"

            lem, pos = parts[0], parts[1] #discard third column (note)
            pos = fix_pos(pos)
            lem = fix_lem(lem, pos, language)
            fms.append((lem, pos))
            POSes.add(pos)
    print(f"Loaded {len(fms)} entries from {path}", file=sys.stderr)
    print(f"POS tags in {path}: {POSes}", file=sys.stderr)
    return fms
    
def uds_to_glossary(ud1, ud2, oname, lang1, lang2):

    ud1 = load_tsv_file(ud1, language=lang1)
    ud2 = load_tsv_file(ud2, language=lang2)

    if len(ud1) != len(ud2):
        print("Warning: UD files have different number of entries", file=sys.stderr)
        sys.exit(1)

    with open(f"{oname}-{lang1}{lang2}.tsv", "w", encoding="utf-8") as out1, open(f"{oname}-{lang2}{lang1}.tsv", "w", encoding="utf-8") as out2:
        for (term1, pos1), (term2, pos2) in zip(ud1, ud2):
            if pos1 != pos2: # and pos1 not in pos2 and pos2 not in pos1:
                print(f"Warning: POS mismatch ({pos1} != {pos2}) for terms '{term1}' and '{term2}'", file=sys.stderr)  
                continue
            out1.write(f"{pos1}\t{term1}\t{term2}\n")
            out2.write(f"{pos2}\t{term2}\t{term1}\n")



if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="Convert two parallel Systran UD files to TSV glossary format.", formatter_class=argparse.RawTextHelpFormatter)
    parse.add_argument("ud1", help="First Systran UD file")
    parse.add_argument("ud2", help="Second Systran UD file")
    parse.add_argument("--oname", default="ud", help="Output file prefix name (creates [oname]-[lang1][lang2].tsv and [oname]-[lang2][lang1].tsv)")
    parse.add_argument("--lang1", default="en", help="language tag in ud1 (default: en)")
    parse.add_argument("--lang2", default="fr", help="language tag in ud2 (default: fr)")
    parse.epilog = """* Systran UD line format is: ^term \\t pos \\t note$. 
* ud1 and ud2 must have the same number of lines (stop otherwise).
* corresponding entries (same line number) are considered translations of each other.
* corresponding entries (same line number) must have the same POS tag in both files (discarded entry otherwise).
* Example usage: python SystranUD2glossary.py resources/ud-enfr_en.dic resources/ud-enfr_fr.dic --oname resources/ud --lang1 en --lang2 fr"""
    args = parse.parse_args()
    
    uds_to_glossary(args.ud1, args.ud2, args.oname, args.lang1, args.lang2)
    print(f"Done! Generated {args.oname}-{args.lang1}{args.lang2}.tsv and {args.oname}-{args.lang2}{args.lang1}.tsv", file=sys.stderr)
