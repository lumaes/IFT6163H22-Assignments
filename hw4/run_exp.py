import json
import os

questions = json.load(open('exp.json', 'r'))

ask = input('Exp run list (a=all): $ ')

all = (ask.strip().lower() == 'all') or (ask.strip().lower() == 'a') or (ask.strip().lower() == '')
cmd_to_run = []

# Check if not run all command
if not all:
    cmd_to_run = [i.strip() for i in ask.strip().split(',')]
    print(cmd_to_run)
else:
    cmd_to_run = [str(i) for i in range(1,len(questions)+1)]

try:

    for question in questions:
        if question in cmd_to_run:
            print("======================")
            print(f"    Question n°{question}")
            print("======================")

            for idx, exp in enumerate(questions[question]):

                print(f'***** Command n° {idx} ***** ({exp})')
                os.system(exp)
        else:
            print("======================")
            print(f"   Question n°{question} (SKIPPED)")
            print("======================")
except KeyboardInterrupt:
    print("[KILL] - Program stopped manually !")
    sys.exit(0)
