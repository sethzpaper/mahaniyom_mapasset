import json
import wave
from pathlib import Path

from faster_whisper import WhisperModel

root = Path(__file__).resolve().parent.parent
source = next((root / 'Sound').glob('*.wav'))
cut = root / '.venv-audio' / 'voice-gap.wav'
with wave.open(str(source), 'rb') as src, wave.open(str(cut), 'wb') as dst:
    dst.setparams(src.getparams())
    src.setpos(int(64 * src.getframerate()))
    dst.writeframes(src.readframes(int(36 * src.getframerate())))
model = WhisperModel('small', device='cpu', compute_type='int8',
                     download_root=str(root / '.venv-audio' / 'models'))
segments, _ = model.transcribe(str(cut), language='th', vad_filter=False,
    initial_prompt='หลวงปู่หงษ์ สุรินทร์ หลวงปู่หมุน ศรีสะเกษ ร้อยเอ็ด หลวงปู่ศรี มหาวีโร หลวงปู่ทองมา ถาวโร')
rows = []
for segment in segments:
    rows.append({'start': segment.start + 64, 'end': segment.end + 64, 'text': segment.text.strip()})
    print(f'{segment.start + 64:.2f} - {segment.end + 64:.2f}', flush=True)
(root / 'research-kajiwat' / 'voice-gap-timing.json').write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
