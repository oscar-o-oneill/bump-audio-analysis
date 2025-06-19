# Bump Audio Analysis

Audio analysis tool to detect bump/knock sounds in long WAV recordings.

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python bump_detector.py your_audio.wav
```

Options:
- `--threshold 2.5` - Adjust sensitivity (lower = more sensitive, default: 3.0)
- `--output results.json` - Custom output file path
- `--quiet` - Minimal console output

## Output

The script generates:
- Console output with timestamps of detected events
- JSON file with detailed analysis results

Example output:
```
Analysis Results:
Duration: 02:30:45.123
Total detections: 23
Background level: 0.012345
Detection threshold: 0.037035

Detected events:
  1. 00:05:23.456
  2. 00:12:34.789
  ...
```

## How It Works

The detector uses two complementary methods:
1. **Onset Detection** - Identifies sudden spectral changes using librosa
2. **Amplitude Thresholding** - Finds peaks above background noise level

Results are combined and filtered to minimize false positives while capturing actual bump/knock events.