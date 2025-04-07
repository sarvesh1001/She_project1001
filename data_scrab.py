import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
df = pd.read_csv('shl_pre_packaged.csv')  # Upload file via GUI

descriptions = []
times = []
for url in df['link']:
    print("Scraping:", url)
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'html.parser')

        # Get Description
        desc_header = soup.find('h4', string="Description")
        if desc_header:
            p = desc_header.find_next('p')
            descriptions.append(p.text.strip())
        else:
            descriptions.append("NA")

        # Get Assessment Length
        length_header = soup.find('h4', string="Assessment length")
        if length_header:
            p = length_header.find_next('p')
            match = re.search(r'\d+', p.text)
            times.append(int(match.group()) if match else "NA")
        else:
            times.append("NA")
    except Exception as e:
        print("Error:", e)
        descriptions.append("NA")
        times.append("NA")
df['description'] = descriptions
df['time'] = times

df.to_csv('shl_prepackaged_updated.csv', index=False)
