# Claude Development Notes

## Project Purpose
Audio analysis tool to detect bump/knock sounds in apartment noise recordings. Built to help identify when loud banging sounds occur in long audio files without manual review.

## Setup Commands
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies  
pip install -r requirements.txt
```

## Testing
```bash
# Run the detector on a test file
python bump_detector.py test_audio.wav

# Run with custom sensitivity
python bump_detector.py test_audio.wav --threshold 2.5
```

## Architecture
- `bump_detector.py` - Main analysis script with dual detection methods
- Uses librosa for onset detection and scipy for amplitude analysis
- Outputs timestamped results in both console and JSON format

## Dependencies
- librosa>=0.10.0 - Audio analysis and onset detection
- numpy>=1.21.0 - Numerical computations
- scipy>=1.7.0 - Signal processing and peak detection