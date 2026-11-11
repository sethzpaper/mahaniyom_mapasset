from urllib.request import urlopen
from html import unescape
import re

url = 'https://luangta.org/resume/resume_luangta.php'
with urlopen(url, timeout=30) as response:
    html = response.read().decode('cp874', errors='replace')
text = re.sub(r'<[^>]+>', ' ', html)
text = re.sub(r'\s+', ' ', unescape(text))
from pathlib import Path
Path(__file__).with_name('luangta-excerpts.txt').write_text(
    '\n'.join(term + ' ' + (text[max(0, text.find(term) - 250):text.find(term) + 350]
                              if text.find(term) >= 0 else 'not found')
              for term in ('พระอาจารย์มั่น', 'วัดป่าบ้านตาด', 'ช่วยชาติ')), encoding='utf-8')
