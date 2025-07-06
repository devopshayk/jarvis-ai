# 🤖 JARVIS AI - Your Personal Engineering Assistant

> **Hand Gesture Recognition + Voice AI + Computer Vision**

A cutting-edge Python AI assistant that combines hand gesture recognition, voice interaction, and computer vision to help you with electronic component identification and board detection.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🖐️ **Hand Gestures** | Interactive hand tracking with pinch gestures for object manipulation |
| 🎤 **Voice Assistant** | Speech-to-text and text-to-speech with ChatGPT integration |
| 🔍 **Board Detection** | Computer vision-based detection of electronic boards and components |
| 🖼️ **Image Search** | Automatic image search and display for electronic components |
| ⚡ **Real-time Processing** | Live camera feed with gesture and voice interaction |

## 📁 Project Structure

```
jarvis-ai/
├── 🤖 Hand_Recognition.py          # Hand gesture tracking & object manipulation
├── 🔍 jarvis_board_detector_pro.py # Electronic board detection
├── 🎤 new_speech_to_text.py        # Main voice assistant with ChatGPT
├── 🖼️ test_voice_search.py         # Image search for components
├── 📋 requirements.txt             # Python dependencies
└── 🛡️ .gitignore                  # Security & ignore rules
```

## 🚀 Quick Start

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Set Up API Keys
```bash
# OpenAI API Key (for ChatGPT)
export OPENAI_API_KEY="your_openai_api_key_here"

# ElevenLabs API Key (for text-to-speech)
export ELEVENLABS_API_KEY="your_elevenlabs_api_key_here"

# Google Cloud Speech API credentials
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/google_credentials.json"
```

### 3️⃣ Run the Assistant
```bash
python new_speech_to_text.py
```

## 🎮 Controls

### Voice Commands
- 🎯 **"Hey Jarvis"** - Wake up the assistant
- 📸 **"Show me a resistor"** - Request component images
- 🔍 **"What do you see?"** - Analyze camera view
- 👋 **"Goodbye"** - Exit the assistant

### Hand Gestures
- 🤏 **Pinch** - Grab and manipulate objects
- 🖱️ **Drag** - Move objects around
- 🔄 **Rotate** - Rotate objects with two hands
- 🗑️ **Trash Zone** - Delete objects by dragging to corner

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `q` | Quit application |
| `1`, `2` | Switch detection modes |
| `ESC` | Exit |
| `r` | Reset crop box |
| `d` | Toggle debug mode |
| `c` | Toggle calibration mode |

## 🔑 Getting API Keys

| Service | Link | Purpose |
|---------|------|---------|
| **OpenAI** | [platform.openai.com](https://platform.openai.com/) | ChatGPT integration |
| **ElevenLabs** | [elevenlabs.io](https://elevenlabs.io/) | Text-to-speech |
| **Google Cloud** | [cloud.google.com](https://cloud.google.com/) | Speech recognition |

## 🛡️ Security

- 🔒 **No hardcoded secrets** - All API keys use environment variables
- 🚫 **Git protection** - `.gitignore` prevents accidental commit of sensitive files
- ⚠️ **Warnings** - Clear alerts when API keys are missing

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| 📹 Camera not working | Check webcam connection and permissions |
| 🎤 Audio issues | Verify microphone and speakers |
| 🔑 API errors | Check environment variables are set |
| 📱 Permission errors | Grant camera/audio access |

## 🎯 Use Cases

- 🔧 **Electronics Education** - Learn about components visually
- 🛠️ **Circuit Design** - Identify and analyze electronic boards
- 🎓 **Engineering Projects** - Interactive component exploration
- 🏭 **Quality Control** - Automated board detection

## 📄 License

This project is for educational and personal use. Please respect the terms of service for the APIs used.

---

**Made with ❤️ for the engineering community** 