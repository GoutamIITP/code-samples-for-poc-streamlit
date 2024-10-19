import streamlit as st
from elevenlabs import ElevenLabs, VoiceSettings, Voice
import moviepy.editor as mp
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import io
import requests
import json

# Title of the app
st.title("Video Upload and Audio Transcription Application")

# Instruction text
st.write("Upload a video file to extract audio and transcribe it:")

def add_audio_to_video(video_path, audio_data, output_video_path):
    """
    Function to add the new audio to the existing video.
    video_path: Path to the original video.
    audio_data: The audio data (from ElevenLabs or TTS system).
    output_video_path: Path to save the final video.
    """
    # Load the video file
    video_clip = mp.VideoFileClip(video_path)
    
    # Load the audio data (BytesIO) and save it to a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(audio_data.getvalue())  # Write the actual byte data from BytesIO
        temp_audio_file_path = temp_audio_file.name
    
    # Load the new audio into MoviePy as an AudioFileClip
    audio_clip = mp.AudioFileClip(temp_audio_file_path)
    
    # Set the audio of the video clip to the new audio
    video_with_new_audio = video_clip.set_audio(audio_clip)
    
    # Save the final video
    video_with_new_audio.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    
    return output_video_path

def generator_to_bytes(generator):
    audio_bytes = io.BytesIO()
    
    # Iterate over the generator and write the data into BytesIO
    for chunk in generator:
        audio_bytes.write(chunk)
    
    # Ensure the pointer is at the start of the buffer before using it
    audio_bytes.seek(0)
    
    return audio_bytes

def gpt4o(data: str = "Hello, Azure OpenAI!", azure_openai_key: str = "22ec84421ec24230a3638d1b51e3a7dc", tokens: int = 50):
    azure_openai_endpoint = "https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
    headers = {
        "Content-Type": "application/json",  # Specifies that we are sending JSON data
        "api-key": azure_openai_key  # The API key for authentication
    }
    
    # Data to be sent to Azure OpenAI
    # Define the payload, which includes the message prompt and token limit.
    # **** This is where you can customize the message prompt and token limit. ****
    data = {
        "messages": [{"role": "user", "content": data}],  # The message we want the model to respond to
        "max_tokens": tokens # Limit the response length
    }
    
    # Making the POST request to the Azure OpenAI endpoint
    # Send the request to the Azure OpenAI endpoint using the defined headers and data.
    response = requests.post(azure_openai_endpoint, headers=headers, json=data)
    return response.json()

def correct_grammar(input_text, max_chunk_size=500):
    def process_chunk(chunk):
        prompt = f"""You are Provided with a Transcript of a Video so, Please Correct the Grammar and other Errors.
Your Response should strictly only contain the corrected transcript for better Parsing and Complete the Response:
{chunk}
        """
        tokens=int(len(prompt)+50)
        response = gpt4o(prompt, tokens=tokens)
        corrected_text = response["choices"][0]["message"]["content"].strip()
        return corrected_text
    
    # Split the input text into smaller chunks
    chunks = [input_text[i:i + max_chunk_size] for i in range(0, len(input_text), max_chunk_size)]
    
    corrected_chunks = []
    for chunk in chunks:
        corrected_chunks.append(process_chunk(chunk))
    
    return " ".join(corrected_chunks)

def tts(text:str, api_key):
    client = ElevenLabs(
        api_key=api_key,
    )
    audio = client.generate(
        text=text,
        voice=Voice(
            voice_id='nPczCjzI2devNBz1zQrb',
            settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
        )
    )
    return audio

# File uploader for video
video_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

def extract_audio_from_video(video_file):
    """
    Function to extract audio from the uploaded video file.
    Writes to a temporary file and returns the path to the audio file.
    """
    # Load video file from uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name

    # Use moviepy to extract audio from the video
    video_clip = mp.VideoFileClip(temp_video_path)
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    video_clip.audio.write_audiofile(audio_path)  # Save audio to a temporary file

    return audio_path, temp_video_path

def transcribe_audio(audio_path):
    """
    Function to transcribe audio to text using SpeechRecognition.
    Reads the audio from the provided file path.
    """
    recognizer = sr.Recognizer()
    
    # Use SpeechRecognition to transcribe the audio
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

        # Try to recognize speech from the audio
        try:
            transcription = recognizer.recognize_google(audio_data)
            return transcription
        except sr.UnknownValueError:
            return "Sorry, the audio could not be understood."
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

if st.button("Rectify"):
    with st.spinner("Extracting audio, transcribing, and generating new audio..."):
        # Extract the original audio from the uploaded video
        audio_path, temp_video_path = extract_audio_from_video(video_file)
        
        # Transcribe the audio
        transcription = transcribe_audio(audio_path)
        
        # Correct the transcription using GPT-4o
        out = correct_grammar(transcription)
        
        # Generate new audio from the corrected transcription using TTS
        out_audio = tts(out, "sk_7c181ae5fd9db167438ffdbf372df015bfb0ff9bbdbb3927")
        
        # Convert the generator object (out_audio) to bytes
        audio_data = generator_to_bytes(out_audio)
        
        # Create a temporary path for the final output video with new audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output_video:
            output_video_path = temp_output_video.name
        
        # Stitch the new audio with the uploaded video
        final_video_path = add_audio_to_video(temp_video_path, audio_data, output_video_path)
        
        # Display the new video with the corrected audio
        st.subheader("Transcription:")
        st.write(out)
        
        st.subheader("Video with New Audio:")
        st.video(final_video_path)

else:
    st.info("Please upload a video file.")
