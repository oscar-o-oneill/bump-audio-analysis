#!/usr/bin/env python3
"""
Audio Bump/Knock Detection Script

Analyzes WAV files to detect sudden loud sounds like bumps or knocks.
Outputs timestamps of detected events for manual review.
"""

import argparse
import numpy as np
import librosa
import scipy.signal
from typing import List, Tuple
import json
import os


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file and return audio data and sample rate."""
    try:
        audio, sr = librosa.load(file_path, sr=None)
        return audio, sr
    except Exception as e:
        raise Exception(f"Error loading audio file: {e}")


def detect_onsets(audio: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Detect onset times using spectral flux."""
    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,
        pre_max=20,
        post_max=20,
        pre_avg=100,
        post_avg=100,
        delta=0.2,
        wait=15
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    return onset_times


def detect_amplitude_peaks(audio: np.ndarray, sr: int, threshold_factor: float = 3.0) -> np.ndarray:
    """Detect peaks based on amplitude thresholding."""
    # Calculate RMS energy in windows
    window_size = int(0.1 * sr)  # 100ms windows
    hop_size = int(0.05 * sr)    # 50ms hop
    
    rms_values = []
    times = []
    
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i + window_size]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
        times.append(i / sr)
    
    rms_values = np.array(rms_values)
    times = np.array(times)
    
    # Calculate threshold based on background noise
    background_level = np.percentile(rms_values, 75)  # 75th percentile as baseline
    threshold = background_level * threshold_factor
    
    # Find peaks above threshold
    peaks, _ = scipy.signal.find_peaks(
        rms_values,
        height=threshold,
        distance=int(1.0 * sr / hop_size)  # Minimum 1 second between detections
    )
    
    peak_times = times[peaks]
    return peak_times, threshold, background_level


def combine_detections(onset_times: np.ndarray, peak_times: np.ndarray, 
                      tolerance: float = 0.5) -> np.ndarray:
    """Combine onset detection and amplitude peak detection results."""
    all_times = []
    
    # Add all onset times
    all_times.extend(onset_times)
    
    # Add peak times that don't have nearby onsets
    for peak_time in peak_times:
        if not any(abs(peak_time - onset_time) < tolerance for onset_time in onset_times):
            all_times.append(peak_time)
    
    return np.sort(all_times)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def analyze_audio(file_path: str, threshold_factor: float = 3.0) -> dict:
    """Main analysis function."""
    print(f"Loading audio file: {file_path}")
    audio, sr = load_audio(file_path)
    
    duration = len(audio) / sr
    print(f"Audio duration: {format_timestamp(duration)}")
    
    print("Detecting onsets...")
    onset_times = detect_onsets(audio, sr)
    
    print("Detecting amplitude peaks...")
    peak_times, threshold, background = detect_amplitude_peaks(audio, sr, threshold_factor)
    
    print("Combining detections...")
    combined_times = combine_detections(onset_times, peak_times)
    
    # Create results
    results = {
        'file_path': file_path,
        'duration_seconds': float(duration),
        'sample_rate': int(sr),
        'threshold_factor': threshold_factor,
        'background_level': float(background),
        'detection_threshold': float(threshold),
        'total_detections': len(combined_times),
        'detections': [
            {
                'time_seconds': float(t),
                'timestamp': format_timestamp(t)
            }
            for t in combined_times
        ]
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Detect bump/knock sounds in audio files')
    parser.add_argument('audio_file', help='Path to WAV audio file')
    parser.add_argument('--threshold', '-t', type=float, default=3.0,
                       help='Amplitude threshold factor (default: 3.0)')
    parser.add_argument('--output', '-o', help='Output JSON file (optional)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        return 1
    
    try:
        results = analyze_audio(args.audio_file, args.threshold)
        
        if not args.quiet:
            print(f"\nAnalysis Results:")
            print(f"Duration: {format_timestamp(results['duration_seconds'])}")
            print(f"Total detections: {results['total_detections']}")
            print(f"Background level: {results['background_level']:.6f}")
            print(f"Detection threshold: {results['detection_threshold']:.6f}")
            
            if results['detections']:
                print(f"\nDetected events:")
                for i, detection in enumerate(results['detections'], 1):
                    print(f"{i:3d}. {detection['timestamp']}")
        
        # Save results
        if args.output:
            output_file = args.output
        else:
            base_name = os.path.splitext(args.audio_file)[0]
            output_file = f"{base_name}_detections.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())