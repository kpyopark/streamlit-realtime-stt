# Speech Transcription Analysis System 🎤

A comprehensive speech transcription and analysis system built with Streamlit, featuring real-time transcription and CER (Character Error Rate) analysis capabilities.

## Features

### 1. Real-time Speech Transcription
- Live audio capture and transcription
- Support for multiple languages (Korean, English, Chinese, Japanese)
- Choice of transcription engines (Gemini, Google STT)
- Real-time display of transcription results with confidence scores
- Export transcription results in multiple formats
- Audio preprocessing and WAV conversion capabilities

### 2. CER/SER Analysis
- Character Error Rate (CER) and Sentence Error Rate (SER) calculation
- Support for multiple input formats (JSON, CSV, TXT)
- Detailed error analysis (substitutions, deletions, insertions)
- Text cleaning and preprocessing
- Visual presentation of analysis results
- Export capabilities for analysis results

## Prerequisites

### Environment Variables
Create a `.env` file with the following variables:
```
PROJECT_ID=your_google_cloud_project_id
LOCATION=your_preferred_location
SAMPLING_RATE=16000
```

### Required Libraries
```bash
pip install streamlit
pip install google-cloud-speech
pip install pydub
pip install pandas
pip install python-dotenv
pip install pyaudio
```

### Google Cloud Setup
1. Set up a Google Cloud Project
2. Enable the Speech-to-Text API
3. Create and download service account credentials
4. Set up appropriate environment variables

## Usage

### Starting the Application
```bash
streamlit run main.py
```

### Real-time Transcription Tab
1. Select your preferred language
2. Choose the transcription engine
3. Upload an audio file or use microphone input
4. Start transcription
5. Export results as needed

### CER Analysis Tab
1. Input the original text
2. Upload transcription results (JSON, CSV, or TXT format)
3. Choose analysis type (full text or auto-split)
4. View detailed analysis results including:
   - Character Error Rate (CER)
   - Sentence Error Rate (SER)
   - Detailed error breakdowns
   - Cleaned text comparison

## Technical Details

### Audio Processing
- Sampling Rate: 16000 Hz
- Chunk Duration: 100ms (configurable)
- Supported Input Formats: MP3, AAC, WAV
- Audio Preprocessing Options: WAV conversion, noise reduction

### Transcription Models
- Support for multiple Google Speech-to-Text models:
  - Standard (long)
  - Enhanced (chirp_2)
- Language-specific optimizations
- Real-time streaming capabilities

### Analysis Metrics
- Character Error Rate (CER)
- Sentence Error Rate (SER)
- Substitution counts
- Deletion counts
- Insertion counts

## File Structure
```
├── main.py                    # Main application entry point
├── realtime_stt.py           # Real-time transcription module
├── cer_analysis.py           # CER analysis implementation
├── stt_utility.py            # Utility functions for STT
├── audio_processor.py        # Audio processing functions
└── requirements.txt          # Project dependencies
```

## Error Handling
- Audio format validation
- Connection error handling
- Invalid input detection
- Transcription error recovery
- Graceful session management

## Notes
- The system requires an active internet connection
- Transcription accuracy may vary based on audio quality
- Large audio files may require additional processing time
- Real-time transcription has a streaming limit of 240 seconds per session

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.