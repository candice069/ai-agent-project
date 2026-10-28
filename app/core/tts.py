import requests
import logging
import os
import tempfile

from app.config import (
    TTS_SERVICE,
    TTS_API_KEY,
    COQUI_MODEL,
    COQUI_SPEAKER_WAV,
    COQUI_LANGUAGE,
)

logger = logging.getLogger(__name__)

class TextToSpeech:
    """Text-to-speech service that converts text to audio"""

    def __init__(self):
        """Initialize the TTS service"""
        self.service = TTS_SERVICE
        self.api_key = TTS_API_KEY

        self.media_type = "audio/wav" if self.service == "coqui" else "audio/mpeg"
        if not self.api_key and self.service != "coqui":
            logger.warning(f"No API key provided for TTS service: {self.service}")

    def synthesize(self, text: str, voice: str = "default") -> bytes:
        """Convert text to speech and return audio data

        Args:
            text: Text to convert to speech
            voice: Voice identifier to use

        Returns:
            Audio data as bytes
        """
        if not text:
            raise ValueError("Text cannot be empty")

        logger.info(f"Synthesizing text with {self.service} TTS (length: {len(text)})")

        if self.service == "coqui":
            return self._coqui_tts(text, voice)
        if self.service == "google":
            return self._google_tts(text, voice)
        elif self.service == "azure":
            return self._azure_tts(text, voice)
        elif self.service == "elevenlabs":
            return self._elevenlabs_tts(text, voice)
        else:
            raise ValueError(f"Unsupported TTS service: {self.service}")

    def _coqui_tts(self, text: str, voice: str) -> bytes:
        """Synthesize locally with Coqui XTTS v2.

        `voice` may point to a speaker WAV. Otherwise `COQUI_SPEAKER_WAV` is
        used. XTTS voice cloning requires one of them.
        """
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise RuntimeError(
                "Coqui support requires: pip install -r requirements-voice.txt"
            ) from exc

        speaker_wav = voice if voice != "default" else COQUI_SPEAKER_WAV
        if not speaker_wav:
            raise ValueError("COQUI_SPEAKER_WAV is required for XTTS synthesis")

        model = TTS(COQUI_MODEL)
        path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output:
                path = output.name
            model.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=COQUI_LANGUAGE,
                file_path=path,
            )
            with open(path, "rb") as audio_file:
                return audio_file.read()
        finally:
            if path and os.path.exists(path):
                os.remove(path)

    def _google_tts(self, text: str, voice: str) -> bytes:
        """Google Cloud TTS implementation

        Args:
            text: Text to convert
            voice: Voice identifier

        Returns:
            Audio data
        """
        try:
            from google.cloud import texttospeech

            # Instantiate a client
            client = texttospeech.TextToSpeechClient()

            # Set the text input to be synthesized
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Map voice parameter to Google voices, or use default
            if voice == "default":
                voice_name = "en-US-Neural2-F"
            else:
                voice_name = voice

            # Build the voice request
            voice_config = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=voice_name
            )

            # Select the type of audio file
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )

            # Perform the synthesis request
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice_config,
                audio_config=audio_config
            )

            # Return the audio content as bytes
            return response.audio_content

        except Exception as e:
            logger.error(f"Error with Google TTS: {str(e)}")
            raise RuntimeError("Google TTS synthesis failed") from e

    def _azure_tts(self, text: str, voice: str) -> bytes:
        """Azure TTS implementation

        Args:
            text: Text to convert
            voice: Voice identifier

        Returns:
            Audio data
        """
        try:
            import azure.cognitiveservices.speech as speechsdk

            # Create a speech configuration with the subscription key and region
            speech_config = speechsdk.SpeechConfig(
                subscription=self.api_key,
                region="eastus"  # Replace with your Azure region
            )

            # Map voice parameter to Azure voices, or use default
            if voice == "default":
                voice_name = "en-US-JennyNeural"
            else:
                voice_name = voice

            # Set the voice
            speech_config.speech_synthesis_voice_name = voice_name

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as output:
                output_path = output.name

            # Create an audio output config for MP3
            audio_config = speechsdk.audio.AudioOutputConfig(
                filename=output_path
            )

            # Create a speech synthesizer
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=audio_config
            )

            # Synthesize the text
            result = synthesizer.speak_text_async(text).get()

            # Check for success
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Read the file and return its contents
                with open(output_path, "rb") as f:
                    audio_data = f.read()

                # Clean up
                os.remove(output_path)

                return audio_data
            else:
                raise Exception(f"Speech synthesis failed: {result.reason}")

        except Exception as e:
            logger.error(f"Error with Azure TTS: {str(e)}")
            raise RuntimeError("Azure TTS synthesis failed") from e

    def _elevenlabs_tts(self, text: str, voice: str) -> bytes:
        """ElevenLabs TTS implementation

        Args:
            text: Text to convert
            voice: Voice identifier

        Returns:
            Audio data
        """
        try:
            # API endpoint
            url = "https://api.elevenlabs.io/v1/text-to-speech"

            # Map voice parameter to ElevenLabs voice IDs, or use default
            if voice == "default":
                voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default voice ID
            else:
                voice_id = voice

            # Add the voice ID to the URL
            url = f"{url}/{voice_id}"

            # Headers
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }

            # Data payload
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }

            # Make the request
            response = requests.post(url, json=data, headers=headers)

            # Check for success
            if response.status_code == 200:
                return response.content
            else:
                raise Exception(f"ElevenLabs API error: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error with ElevenLabs TTS: {str(e)}")
            raise RuntimeError("ElevenLabs TTS synthesis failed") from e
