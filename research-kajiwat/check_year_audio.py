"""Transcribe a short excerpt twice without supplying an expected year."""
import json
import wave
from pathlib import Path
from faster_whisper import WhisperModel

root = Path(__file__).resolve().parent.parent
source = next((root / 'Sound').glob('*.wav'))
cut = root / '.venv-audio' / 'voice-year-check.wav'
with wave.open(str(source), 'rb') as src, wave.open(str(cut), 'wb') as dst:
    dst.setparams(src.getparams())
    src.setpos(int(116 * src.getframerate()))
    dst.writeframes(src.readframes(int(13 * src.getframerate())))
model = WhisperModel('small', device='cpu', compute_type='int8',
                     download_root=str(root / '.venv-audio' / 'models'))
result = []
for beam in (1, 5):
    segments, _ = model.transcribe(str(cut), language='th', vad_filter=False,
                                   beam_size=beam, condition_on_previous_text=False)
    result.append({'beam_size': beam, 'segments': [
        {'start': s.start + 116, 'end': s.end + 116, 'text': s.text.strip()} for s in segments]})
(root / 'research-kajiwat' / 'voice-year-check.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print('Completed independent short excerpt transcription; human listening remains necessary.')
