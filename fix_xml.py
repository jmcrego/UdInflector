
import sys

for line in sys.stdin:
    l = line.strip()

    #  <source>paradise (noun)</source>
    if l.startswith("<source>") and l.endswith("</source>"):
        changed = False
        term = l[len("<source>"):-len("</source>")]
        pos = term[term.find("(")+1:term.find(")")]
        lem = term[:term.find("(")].strip()
        # remove "to " from verb infinitive form in English (e.g. "to speak" -> "speak")
        if lem.startswith("to ") and pos == "verb":
            lem = lem[3:] 
            changed = True
        if lem.startswith("se ") and pos == "verb":
            lem = lem[3:] 
            changed = True
        # phrases
        if lem.find(" ") != -1 and pos in ["verb", "noun", "adj"]:
            pos = pos + ' phrase'
            changed = True
        term = f"{lem} ({pos})"
        if changed:
            print(f"{term}", file=sys.stderr)
        print(f"  <source>{term}</source>")
        continue


    print(line.strip("\n"))
