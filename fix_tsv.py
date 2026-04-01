
import sys

for line in sys.stdin:
    l = line.strip()

    toks = l.split("\t")
    if len(toks) == 3:

        lem, pos, tag = toks
        changed = False

        if lem.startswith("to ") and pos == "verb":
            lem = lem[3:] # remove "to " from verb infinitive form in English (e.g. "to speak" -> "speak")
            changed = True

        if lem.startswith("se ") and pos == "verb":
            lem = lem[3:] # remove "se " from verb infinitive form in French (e.g. "se parler" -> "parler")
            changed = True

        if lem.find(" ") != -1 and pos in ["verb", "noun", "adj"]:
            pos = pos + ' phrase'
            changed = True

        if changed:
            print(f"{lem}\t{pos}\t{tag}", file=sys.stderr)
            print(f"{lem}\t{pos}\t{tag}")
            continue


    print(line.strip("\n"))
