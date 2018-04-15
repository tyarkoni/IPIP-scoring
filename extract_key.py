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
import warnings
from os.path import join


# Include column for IPIP item IDs in keys?
INCLUDE_IPIP_ID = True


def sanitize_text(text):
    ''' Replace certain unicode characters and clean up strings. '''
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(u"[\u2012\u2013\u2014\u2015\x96]+", "-", text, re.DOTALL)
    text = re.sub(u"[\x92]+", "'", text, re.DOTALL)
    text = re.sub('[\r\n\s]+', ' ', text)
    return text.strip()


# Output location
if not os.path.exists('keys'):
    os.mkdir('keys')

# Default config
default_config = json.load(open(join('configs', 'default.json'), 'r'))

# Read config files
measure_configs = glob('configs/*.json')

# Read IPIP items if we'll need them later. Remove punctuation/casing to
# increase odds of matching
if INCLUDE_IPIP_ID:
    ipip_items = pd.read_csv(join('support', 'ipip_items.txt'), sep='\t',
                             names=['text', 'id'])
    ipip_items['text'] = ipip_items['text'].str.lower() \
                                           .str.replace('[^\w\-\s]+', '')
    ipip_items = ipip_items.set_index('text')

    # There are typos/phrasing errors in some items, so remap these
    remap_items = pd.read_csv(join('support', 'remap_items.txt'), sep='\t')
    remap_items = dict(remap_items.values)

# At least one measure has broken HTML that combines items into one cell;
# we need to split these.
items_to_split = pd.read_csv(join('support', 'merged_items.txt'), sep='\t')
items_to_split = dict(items_to_split.values)

# Loop over measure and create keys
for f in measure_configs:

    if 'default' in f:
        continue

    # Overwrite default config
    config = default_config.copy()
    config.update(json.load(open(f, 'r')))

    name = os.path.basename(f).split('.')[0]
    print("Processing %s..." % name)

    # Extract all <td> tags into a single list--this requires some
    # hackery to deal with arbitrary/inconsistent nesting across
    # different inventories
    soup = BeautifulSoup(requests.get(config['url']).text)
    tags = []

    for td in soup.find_all('td'):

        sub_tds = td.find_all('td')
        subs = []

        for t in sub_tds:

            subs.append(t.text)
            t.replaceWith('')

        tags.append(td.text)

    # Drop empty tags
    tags = [t for t in tags if re.search('[a-zA-Z]+', t)]

    # Initialize state vars
    sign = config.get('default_sign', '+')
    title = None
    valid = False
    skip = False
    scales = {}
    n_tags = len(tags)

    # Loop over lines, store in scale dictionary
    for i, t in enumerate(tags):

        text = sanitize_text(t)

        # Skip until we find start line, and quit once we hit
        # end line
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
            title = re.search(config['extract_title'], text).group(1)
            scales[title] = defaultdict(list)

        # Check if this is a regular item, and if so, add it
        elif re.search(config['item'], text):

            text = re.sub('[\[\]]+', '', text)
            text = re.search(config['extract_item'], text).group(1)

            # split incorrectly merged items in two
            if text in items_to_split:
                first = items_to_split[text]
                scales[title][sign].append(first)
                text = text.replace(first, '').strip()

            scales[title][sign].append(text)

    # Convert key to DF form
    items = []
    item_map = OrderedDict()
    counter = -1

    # Drop any empty scales (e.g., because of higher-order labels)
    scales = {k: v for k, v in scales.items() if v.get('+') or v.get('-')}

    # Loop over extracted scale and store a list of item info tuples
    for i, (scale, v) in enumerate(scales.items()):
        for sign, sign_items in v.items():
            for text in sign_items:
                _sign = 1 if sign == '+' else -1
                if text not in item_map:
                    counter += 1
                item_map[text] = counter
                items.append((scale, i, _sign, text, item_map[text]))

    # Initialize empty scoring matrix and populate it
    key = np.zeros((counter + 1, len(scales)), dtype=int)

    for s, s_id, sign, i, i_id in items:
        key[i_id, s_id] = sign

    # Convert to DF
    df = pd.DataFrame(key, index=list(item_map.keys()),
                      columns=list(scales.keys()))

    # Optionally add column for IPIP item IDs
    if INCLUDE_IPIP_ID:
        _index = df.index.copy()
        df.index = df.index.str.lower().str.replace('[^\w\-\s]+', '')

        # Replace broken items
        df.index = [remap_items[x] if x in remap_items else x
                    for x in df.index]

        df = df.merge(ipip_items, left_index=True, right_index=True,
                      how='left')
        df.insert(0, 'ipip_id', df.pop('id'))

        # Display missing items
        missing = set(df.index) - set(ipip_items.index)
        if missing:
            missing = '\n\t' + '\n\t'.join(list(missing))
            warnings.warn("The following items in the %s scoring key could "
                          "not be found in the full IPIP item list. This is "
                          "most likely due to a typo or difference in phrasing"
                          " somewhere: %s" %
                          (name, missing))

        # Restore original index (i.e., with punctuation intact)
        df.index = _index

    # Save
    df.to_csv(join('keys', '%s.tsv' % name), sep='\t', index_label='item')
