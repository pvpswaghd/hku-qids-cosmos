import requests

# ===============================================================================================================================================================
# Some inputs that you need to adjust
STRATEGY_PATH = r"D:\HKU QIDS Affairs\strategy.py"  # TODO: Change to your file address for strategy.py
REQUIREMENTS_PATH = r"D:\HKU QIDS Affairs\requirements.txt"  # TODO: Change to your file address for requirement.txt
GROUP_ID = "G000"  # TODO: Change to "G" + group_id
ACCESS_TOKEN = "wish_you_good_luck"  # TODO: Change to access_token
# ===============================================================================================================================================================

URL = "http://competition.hkuqids.com:8880/upload_submission"
files = {"strategy": open(STRATEGY_PATH, "rb"),
         "requirements": open(REQUIREMENTS_PATH, "rb")
         }
values = {
        "access_token": ACCESS_TOKEN,
        "group_id": GROUP_ID
        }
r = requests.post(URL, files = files, data = values)
print(r.text)
