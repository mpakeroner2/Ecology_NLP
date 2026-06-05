import os
from datetime import datetime

current_dir = os.getcwd()
file = os.path.join(current_dir, "log.txt")

def clean_log(file = 'log_fastapi.txt'):
    with open(file, 'w') as f:
        f.write("")

def log(*args, file='log.txt'):
    if len(args) > 1:
        now = datetime.now().strftime("%d.%m %H:%M:%S")
        text = f"{now} " + str(args[0])
        for i in args[1:]:
            text += " " + str(i)
    else:
        text = str(args[0])
    with open(file, 'a', encoding='utf-8') as f:
        f.write(text + '\n')   
        print(text)