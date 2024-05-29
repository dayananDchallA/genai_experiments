from elevenlabs import stream
from dotenv import load_dotenv
import os
load_dotenv()
	
from elevenlabs.client import ElevenLabs

client = ElevenLabs(
  api_key=os.getenv('ELEVEN_API_KEY'), # Defaults to ELEVEN_API_KEY
)

response = client.voices.get_all()
audio = client.generate(text="Hello there!", voice=response.voices[0])

# Stream the audio directly 
stream(audio)



# Saving the audio for later 
with open('output.mp3', 'wb') as f:
    for i,chunk in enumerate(audio):
        if chunk: f.write(chunk) 