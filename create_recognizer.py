import streamlit as st
from pydub import AudioSegment
import threading
import queue
import asyncio
from google.cloud import speech_v2 as speech
from typing import Optional, Callable
import wave
import pyaudio
import time
import io
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
CHIRP_RECOGNIZER_ID = os.getenv("REGONIZER_ID")

def create_recognizer(recognizer_id: str) -> speech.types.Recognizer:
    """Сreates a recognizer with an unique ID and default recognition configuration.
    Args:
        recognizer_id (str): The unique identifier for the recognizer to be created.
    Returns:
        cloud_speech.Recognizer: The created recognizer object with configuration.
    """
    # Instantiates a client
    client = speech.SpeechClient()

    request = speech.types.CreateRecognizerRequest(
        parent=f"projects/{PROJECT_ID}/locations/{LOCATION}",
        recognizer_id=recognizer_id,
        recognizer=speech.types.Recognizer(
            default_recognition_config=speech.types.RecognitionConfig(
                language_codes=["cmn-Hans-CN", "cmn-Hant-TW", "yue-Hant-HK"], model="chirp_2"
            ),
        ),
    )
    # Sends the request to create a recognizer and waits for the operation to complete
    operation = client.create_recognizer(request=request)
    recognizer = operation.result()

    print("Created Recognizer:", recognizer.name)
    return recognizer

response = create_recognizer(CHIRP_RECOGNIZER_ID)
print(response)