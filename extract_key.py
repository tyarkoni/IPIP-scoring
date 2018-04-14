'''
Copyright 2018 Tal Yarkoni

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import re
import unicodedata
import json
from collections import defaultdict, OrderedDict
from glob import glob


def sanitize_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(u"[\u2012\u2013\u2014\u2015\x96]+", "-", text, re.DOTALL)
    text = re.sub(u"[\x92]+", "'", text, re.DOTALL)
    return text.strip()


if not os.path.exists('keys'):
    os.mkdir('keys')

# Default config
default_config = json.load(open(os.path.join('configs', 'default.json'), 'r'))

# Read config files
measure_configs = glob('configs/*.json')

# Loop over measure and create keys
for f in measure_configs:

    if 'default' in f:
        continue

    config = default_config.copy()
    config.update(json.load(open(f, 'r')))
    name = os.path.basename(f).split('.')[0]

    print("Processing %s..." % name)

    soup = BeautifulSoup(requests.get(config['url']).text)

    tags = []
    for td in soup.find_all('td'):
        sub_tds = td.find_all('td')
        subs = []
        for t in sub_tds:
            subs.append(t.get_text(strip=True))
            t.replaceWith('')
        tags.append(td.get_text(strip=True))

    tags = [t for t in tags if re.search('[a-zA-Z]+', t)]

    sign = config.get('default_sign', '+')
    title = None
    valid = False
    skip = False
    scales = {}
    n_tags = len(tags)

    # Loop over lines, store in scale dictionary
    for i, t in enumerate(tags):

        text = sanitize_text(t)

        # Make sure we're within the valid processing bounds
        if not valid:
            if re.search(config['start_line'], text):
                valid = True
            else:
                continue
        elif re.search(config['end_line'], text):
            break

        # Skip lines--including in entire blocks
        if not skip and re.search(config['skip_onset'], text, re.DOTALL):
            skip = True
            continue
        elif skip:
            if re.search(config['skip_offset'], text, re.DOTALL):
                skip = False
            else:
                continue

        if re.search(config['skip'], text):
            continue

        # Check if line indicates sign of scoring
        m = re.search(config['key_sign'], text)
        if m:
            sign = m.group(1)

        # Check if line defines a scale name
        elif re.search(config['scale_title'], text, re.DOTALL):
            title = re.sub('[\r\n\s]+', ' ', text)
            title = re.search(config['extract_title'], title).group(1)
            scales[title] = defaultdict(list)

        # Check if this is a regular item, and if so, add it
        elif re.search(config['item'], text):
            text = re.sub('[\r\n\s]+', ' ', text, re.DOTALL)
            text = re.sub('[\[\]]+', '', text)
            text = re.search(config['extract_item'], text).group(1)
            scales[title][sign].append(text)

    # Convert key to DF form
    items = []
    item_map = OrderedDict()
    counter = -1

    # Drop any empty scales (e.g., because of higher-order labels)
    scales = {k: v for k, v in scales.items() if v.get('+') or v.get('-')}

    for i, (scale, v) in enumerate(scales.items()):
        for sign, sign_items in v.items():
            for text in sign_items:
                sign = 1 if sign == '+' else -1
                if text not in item_map:
                    counter += 1
                item_map[text] = counter
                items.append((scale, i, sign, text, item_map[text]))

    key = np.zeros((counter + 1, len(scales)), dtype=int)

    for s, s_id, sign, i, i_id in items:
        key[i_id, s_id] = sign

    df = pd.DataFrame(key, index=list(item_map.keys()),
                      columns=list(scales.keys()))

    df.to_csv(os.path.join('keys', '%s.tsv' % name), sep='\t',
              index_label='item')
