"""Pluto AI Voice Adapter - Cloud Backend

Runs on cloud hosting (Render, Railway, Fly.io, etc.).
Receives audio/images from the XIAO ESP32-S3 Sense device.
Performs speech-to-text, AI inference, and text-to-speech.
Returns processed audio response to the device.

Features:
- Speech-to-text: Vosk (local, offline, keeps voice data private)
- AI inference: Claude API (cloud, via Anthropic)
- Text-to-speech: ElevenLabs API
- Web search: Brave Search API
- Persistent memory: User facts and preferences
"""

from __future__ import annotations

import json
import os
import re
import io
import wave
from pathlib import Path
from typing import Any
from datetime import datetime

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
from vosk import KaldiRecognizer, Model
import anthropic
from elevenlabs import ElevenLabs

# ============================================================================
# Configuration
# ============================================================================

app = Flask(__name__)
CORS(app)

# Pluto personality and authentication
ASSISTANT_NAME = "Pluto"
API_KEY = os.environ.get("PLUTO_API_KEY", "your-secret-key-change-in-production")
STOP_WORDS = {"stop", "quit", "exit", "goodbye", "shut down", "goodbye"}

# Memory storage
MEMORY_DIR = Path(os.environ.get("PLUTO_MEMORY_DIR", "./memory"))
MEMORY_DIR.mkdir(exist_ok=True)

# Vosk speech recognition (must be installed on backend server)
VOSK_MODEL_PATH = Path(
    os.environ.get("VOSK_MODEL_PATH", "./vosk-model-small-en-us-0.15")
)

# Anthropic Claude API
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"

# ElevenLabs Text-to-Speech
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice (or set your own)

# Brave Search API
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")

# System prompt for Jarvis-style personality
SYSTEM_PROMPT = """You are Pluto, a calm, concise AI voice assistant with dry wit.
Your personality is Jarvis-esque: efficient, intelligent, proactive but never naggy.
- Respond concisely by default; expand only when asked for detail.
- Address the user directly. Use their name if you know it.
- Be helpful, informative, and occasionally witty.
- Keep responses under 30 seconds of speech (~150 words).
- If asked for the time or date, be precise.
- If you don't know something, say so directly. Offer to search the web.
- Never pretend to have real-time capabilities you lack."""


# ============================================================================
# Memory Management
# ============================================================================

def get_memory_file(device_id: str) -> Path:
    """Get the memory file path for a specific device."""
    return MEMORY_DIR / f"{device_id}_memory.json"


def load_memory(device_id: str) -> dict[str, Any]:
    """Load remembered information for a device."""
    memory_file = get_memory_file(device_id)
    
    if not memory_file.exists():
        return {
            "device_id": device_id,
            "facts": [],
            "preferences": {},
            "context": [],
            "created_at": datetime.now().isoformat(),
        }

    try:
        return json.loads(memory_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "device_id": device_id,
            "facts": [],
            "preferences": {},
            "context": [],
            "created_at": datetime.now().isoformat(),
        }


def save_memory(memory: dict[str, Any]) -> None:
    """Save memory to persistent storage."""
    memory_file = get_memory_file(memory["device_id"])
    memory_file.write_text(
        json.dumps(memory, indent=2, ensure_ascii=True),
        encoding="utf-8"
    )


def remember(memory: dict[str, Any], fact: str) -> str:
    """Store a new fact in memory."""
    fact = fact.strip().rstrip(".")
    if not fact:
        return "I did not hear what you want me to remember."

    # Check for duplicates (case-insensitive)
    if fact.casefold() not in {item.casefold() for item in memory["facts"]}:
        memory["facts"].append(fact)
        save_memory(memory)
    
    return f"I will remember that {fact}."


# ============================================================================
# Speech Recognition
# ============================================================================

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio using Vosk (offline, local).
    
    Args:
        audio_bytes: Raw WAV audio data
        
    Returns:
        Transcribed text
    """
    if not VOSK_MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"Vosk model not found at {VOSK_MODEL_PATH}. "
            "Download from https://alphacephei.com/vosk/models and set VOSK_MODEL_PATH."
        )

    try:
        # Parse WAV header to get sample rate
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())

        # Initialize Vosk recognizer
        model = Model(str(VOSK_MODEL_PATH))
        recognizer = KaldiRecognizer(model, sample_rate)
        recognizer.AcceptWaveform(frames)
        
        result = json.loads(recognizer.FinalResult())
        return result.get("text", "").strip()
    
    except Exception as e:
        raise ValueError(f"Speech recognition failed: {str(e)}")


# ============================================================================
# Web Search
# ============================================================================

def search_web(query: str) -> str:
    """Search using Brave Search API."""
    if not BRAVE_API_KEY:
        return "Web search is not configured on this backend."
    
    try:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
            params={"q": query, "count": 5},
            timeout=10,
        )
        response.raise_for_status()
        
        results = response.json().get("web", [])
        if not results:
            return f"I found no results for '{query}'."
        
        # Summarize top 3 results
        summaries = []
        for i, result in enumerate(results[:3], 1):
            summaries.append(f"{i}. {result.get('title', 'No title')}: {result.get('description', 'No description')}")
        
        return " ".join(summaries)
    
    except requests.RequestException as e:
        return f"Web search failed: {str(e)}"


# ============================================================================
# AI Inference
# ============================================================================

def query_claude(prompt: str, memory: dict[str, Any]) -> str:
    """
    Send a prompt to Claude API for intelligent reasoning.
    
    Args:
        prompt: User query
        memory: Device memory including facts and context
        
    Returns:
        AI response
    """
    if not CLAUDE_API_KEY:
        return "Error: Claude API key not configured."
    
    try:
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        
        # Build context from memory
        context_text = ""
        if memory.get("facts"):
            context_text += f"\nUser facts I remember: {'; '.join(memory['facts'])}\n"
        
        # Build conversation history (last 5 turns)
        messages = []
        for turn in memory.get("context", [])[-5:]:
            messages.append({"role": "user", "content": turn.get("user", "")})
            messages.append({"role": "assistant", "content": turn.get("assistant", "")})
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        # Call Claude
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            system=SYSTEM_PROMPT + context_text,
            messages=messages,
        )
        
        reply = response.content[0].text.strip()
        
        # Update context history
        if len(memory["context"]) > 10:
            memory["context"] = memory["context"][-10:]
        memory["context"].append({"user": prompt, "assistant": reply})
        save_memory(memory)
        
        return reply
    
    except anthropic.APIError as e:
        return f"Error querying Claude: {str(e)}"


# ============================================================================
# Text-to-Speech
# ============================================================================

def synthesize_speech(text: str) -> bytes:
    """
    Convert text to speech using ElevenLabs API.
    
    Args:
        text: Text to synthesize
        
    Returns:
        Audio data as WAV bytes
    """
    if not ELEVENLABS_API_KEY:
        raise ValueError("ElevenLabs API key not configured.")
    
    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        
        audio_stream = client.generate(
            text=text,
            voice=ELEVENLABS_VOICE,
            model="eleven_monolingual_v1",
        )
        
        # Collect audio chunks
        audio_data = b"".join(audio_stream)
        return audio_data
    
    except Exception as e:
        raise ValueError(f"Text-to-speech failed: {str(e)}")


# ============================================================================
# Command Processing
# ============================================================================

def understand(text: str, memory: dict[str, Any]) -> tuple[str, bool]:
    """
    Process user input: handle local commands or defer to Claude.
    
    Returns:
        (reply_text, should_stop)
    """
    cleaned = text.strip()
    lowered = cleaned.casefold()

    # Stop command
    if lowered in STOP_WORDS:
        return "Goodbye.", True

    # Remember command
    remember_match = re.match(
        r"(?:please )?(?:remember|don't forget) (?:that )?(.+)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if remember_match:
        return remember(memory, remember_match.group(1)), False

    # Recall memory
    if "what do you remember" in lowered or lowered in {"my memories", "memory"}:
        if not memory["facts"]:
            return "I do not remember anything yet.", False
        facts = "; ".join(memory["facts"])
        return f"I remember: {facts}.", False

    # Help
    if lowered in {"help", "what can you do"}:
        return (
            "I can remember facts, answer questions with AI, search the web, tell you the time, "
            "and more. Just ask me something.",
            False,
        )

    # Time
    if "time" in lowered or "what time" in lowered:
        from datetime import datetime
        return datetime.now().strftime("The time is %I:%M %p."), False

    # Web search
    search_match = re.match(
        r"(?:search|find|look up|google) (?:for )?(.+)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if search_match:
        query = search_match.group(1)
        return search_web(query), False

    # Default: query Claude for intelligent response
    return query_claude(cleaned, memory), False


# ============================================================================
# API Endpoints
# ============================================================================

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "assistant": ASSISTANT_NAME}), 200


@app.route("/chat", methods=["POST"])
def chat():
    """
    Chat endpoint: accepts text, returns text response.
    
    Request JSON:
        {
            "device_id": "esp32-12345",
            "text": "What time is it?",
        }
    
    Response JSON:
        {
            "reply": "The time is 2:45 PM.",
            "should_stop": false
        }
    """
    # Verify API key
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_KEY}" and API_KEY != "your-secret-key-change-in-production":
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json()
        device_id = data.get("device_id", "unknown")
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Load device memory
        memory = load_memory(device_id)

        # Process the input
        reply, should_stop = understand(text, memory)

        return jsonify({
            "reply": reply,
            "should_stop": should_stop,
            "assistant": ASSISTANT_NAME,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/speech", methods=["POST"])
def speech():
    """
    Speech endpoint: accepts audio, returns text + audio response.
    
    Request:
        - Audio file (form data, key="audio")
        - device_id (form data)
    
    Response:
        - JSON with transcription + reply text
        - Audio file (WAV)
    """
    # Verify API key
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_KEY}" and API_KEY != "your-secret-key-change-in-production":
        return jsonify({"error": "Unauthorized"}), 401

    try:
        device_id = request.form.get("device_id", "unknown")
        
        # Get audio file
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files["audio"]
        audio_bytes = audio_file.read()

        # Transcribe audio
        text = transcribe_audio(audio_bytes)
        if not text:
            return jsonify({
                "error": "Could not transcribe audio",
                "transcription": ""
            }), 400

        # Load device memory
        memory = load_memory(device_id)

        # Process the input
        reply, should_stop = understand(text, memory)

        # Synthesize response to speech
        try:
            response_audio = synthesize_speech(reply)
            
            # Return JSON + audio
            return {
                "transcription": text,
                "reply": reply,
                "should_stop": should_stop,
                "assistant": ASSISTANT_NAME,
                "audio": response_audio  # Binary audio data
            }, 200
        
        except ValueError as tts_error:
            # If TTS fails, return text response only
            return jsonify({
                "transcription": text,
                "reply": reply,
                "should_stop": should_stop,
                "assistant": ASSISTANT_NAME,
                "error": f"Text-to-speech unavailable: {str(tts_error)}"
            }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/memory/<device_id>", methods=["GET"])
def get_memory(device_id: str):
    """Get the memory for a specific device."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_KEY}" and API_KEY != "your-secret-key-change-in-production":
        return jsonify({"error": "Unauthorized"}), 401

    memory = load_memory(device_id)
    return jsonify(memory), 200


@app.route("/memory/<device_id>", methods=["DELETE"])
def clear_memory(device_id: str):
    """Clear the memory for a specific device."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_KEY}" and API_KEY != "your-secret-key-change-in-production":
        return jsonify({"error": "Unauthorized"}), 401

    memory_file = get_memory_file(device_id)
    if memory_file.exists():
        memory_file.unlink()
    
    return jsonify({"status": "memory cleared"}), 200


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    
    print(f"\n{ASSISTANT_NAME} Cloud Backend")
    print("=" * 60)
    print(f"Starting on port {port}")
    print(f"API Key authentication: {'Enabled' if API_KEY != 'your-secret-key-change-in-production' else 'DISABLED (dev mode)'}")
    print(f"Vosk model: {VOSK_MODEL_PATH}")
    print(f"Claude model: {CLAUDE_MODEL}")
    print("=" * 60 + "\n")
    
    app.run(host="0.0.0.0", port=port, debug=debug)
