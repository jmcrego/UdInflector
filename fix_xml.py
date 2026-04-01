
import os
import sys

for line in sys.stdin:
    l = line.strip()

    #  <source>paradise (noun)</source>
    if l.startswith("<source>") and l.endswith("</source>"):
        term = l[len("<source>"):-len("</source>")]
        pos = term[term.find("(")+1:term.find(")")]
        lem = term[:term.find("(")].strip()
        # remove "to " from verb infinitive form in English (e.g. "to speak" -> "speak")
        if lem.startswith("to ") and pos == "verb":
            lem = lem[3:] 
        # phrases
        if lem.find(" ") != -1 and pos in ["verb", "noun", "adj"]:
            pos = pos + ' phrase'
        term = f"{lem} ({pos})"
        print(f"<source>{term}</source>")
        continue


    print(line)
