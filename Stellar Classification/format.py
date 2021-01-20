import csv
import re

def proc_string(s):
    category = None
    if s:
        if s[0] in "DOBAFGKMsdgSCYWRN": # or s[:2] == "sd":
            star = re.match("(O|B|A|F|G|K|M|S|C|Y|W|R|N|g)\d? ?/?(O|B|A|F|G|K|M|S|C|Y|W|R|N|g)?\d? ?(I*V?I*)", s)
            sdegen  = re.match("(D*|sd|d)", s)
            if star or sdegen:
                if star:
                    if '' not in star.groups():
                        # print("s: ", star.groups(), s, len(star.groups()))
                        category = (star.groups()[0], star.groups()[-1])
                        if category[1] == "VII":
                            category = ("D", "D")
                elif sdegen:
                    # print("sd: ", sdegen.groups(), s)
                    category = ("D", "D")
            else:
                category = None
        else:
            category = None
            pass
    return category

with open("outdata.csv", "w") as wf:
    with open("hygdata_v3.csv", "r") as f:
        cf = csv.reader(f)
        next(cf)
        for line in cf:
            r = proc_string(line[15])
            if r is not None:
                # print(r)
                wf.write(', '.join(line) + ', ' +  r[1] + ', ' + r[0] + '\n')
                # pass
            else:
                pass
                # print('no match: ', line[15])


