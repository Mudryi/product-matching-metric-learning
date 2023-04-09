import os

train_id = "12qVCej0ouA8xkBnkBWcsLu8fTriUCszT"
test_id = "1-03bhKmN5tPVT73UnHhWOkluB59dgp0y"
MCS_id = "1PwF04h2bN8dXx4owIdhoULC5EH0wkwbp"

train_name = "train.zip"
test_name = "test.zip"
MCS_name = "MCS2023_development_test_data.zip"

if not os.path.exists("train"):
    os.system(f"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={test_id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={test_id}" -O {test_name} && rm -rf /tmp/cookies.txt""")
    os.system("""unzip -qq 'test.zip'""")
    os.system(f"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={train_id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={train_id}" -O {train_name} && rm -rf /tmp/cookies.txt""")
    os.system(""" unzip -qq 'train.zip' """)
    os.system(f"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={MCS_id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={MCS_id}" -O {MCS_name} && rm -rf /tmp/cookies.txt""")
    os.system("""unzip -qq 'MCS2023_development_test_data.zip'""")



