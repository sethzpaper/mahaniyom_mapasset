import json
from pathlib import Path

from faster_whisper import WhisperModel

root = Path(__file__).resolve().parent.parent
audio_files = list((root / 'Sound').glob('*.wav'))
if len(audio_files) != 1:
    raise RuntimeError(f'Expected exactly one WAV in Sound, found {len(audio_files)}')

model = WhisperModel('small', device='cpu', compute_type='int8',
                     download_root=str(root / '.venv-audio' / 'models'))
segments, info = model.transcribe(
    str(audio_files[0]), language='th', vad_filter=True, word_timestamps=True,
    initial_prompt='นครราชสีมา หลวงพ่อคูณ นครพนม หลวงปู่คำพันธ์ สุรินทร์ หลวงปู่หงษ์ ศรีสะเกษ หลวงปู่หมุน ร้อยเอ็ด หลวงปู่ศรี หลวงปู่ทองมา หลวงปู่ผาง หลวงตามหาบัว')
rows = []
output = root / 'research-kajiwat' / 'voice-timing.json'
for segment in segments:
    rows.append({'start': segment.start, 'end': segment.end, 'text': segment.text.strip(),
                 'words': [{'start': w.start, 'end': w.end, 'word': w.word} for w in (segment.words or [])]})
    output.write_text(json.dumps({'language': info.language, 'segments': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{segment.start:.2f} - {segment.end:.2f}', flush=True)
output.write_text(json.dumps({'language': info.language, 'segments': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
print('Saved timing JSON.', flush=True)
