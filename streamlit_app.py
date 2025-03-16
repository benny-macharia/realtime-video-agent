import streamlit as st
import asyncio
import base64
import io
import os
import sys
import time
import threading
import queue
import traceback

import cv2
import pyaudio
import PIL.Image
import numpy as np
import mss

from google import genai

# Import backports for Python < 3.11
if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

# Audio/video constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
MODEL = "models/gemini-2.0-flash-exp"
CONFIG = {"response_modalities": ["AUDIO"]}

# Initialize Google client
client = genai.Client(http_options={"api_version": "v1alpha"})
pya = pyaudio.PyAudio()

class StreamlitGeminiLive:
    def __init__(self, video_mode="camera"):
        self.video_mode = video_mode
        self.running = False
        self.session = None
        
        # Queues for communication
        self.audio_in_queue = queue.Queue()
        self.out_queue = queue.Queue(maxsize=5)
        self.response_queue = queue.Queue()
        
        # Tracking state
        self.current_frame = None
        self.audio_stream = None
        self.threads = []
        
        # Track when to send frames to the model
        self.last_frame_sent = 0
    
    def _get_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            return None, None
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([640, 480])  # Smaller for display
        
        # For display in Streamlit
        display_frame = np.array(img)
        
        # For sending to model
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        model_frame = {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}
        
        return display_frame, model_frame
    
    def _get_screen(self):
        try:
            sct = mss.mss()
            monitor = sct.monitors[0]
            
            screenshot = sct.grab(monitor)
            img = PIL.Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            img.thumbnail([640, 480])  
            
            display_frame = np.array(img)
            
            # For sending to model
            image_io = io.BytesIO()
            img.save(image_io, format="jpeg")
            image_io.seek(0)
            mime_type = "image/jpeg"
            image_bytes = image_io.read()
            model_frame = {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}
            
            return display_frame, model_frame
        except Exception as e:
            st.error(f"Screen capture error: {e}")
            return None, None
    
    def capture_video_thread(self):
        """Thread to capture video frames"""
        if self.video_mode == "camera":
            cap = cv2.VideoCapture(0)
            while self.running:
                display_frame, model_frame = self._get_frame(cap)
                if display_frame is not None:
                    self.current_frame = display_frame
                    
                    # Send to model only once per second. 
                    current_time = time.time()
                    if current_time - self.last_frame_sent >= 1.0:
                        if model_frame and not self.out_queue.full():
                            self.out_queue.put(model_frame)
                            self.last_frame_sent = current_time
                
                time.sleep(0.03)
            cap.release()
        elif self.video_mode == "screen":
            while self.running:
                display_frame, model_frame = self._get_screen()
                if display_frame is not None:
                    self.current_frame = display_frame
    
                    current_time = time.time()
                    if current_time - self.last_frame_sent >= 1.0:
                        if model_frame and not self.out_queue.full():
                            self.out_queue.put(model_frame)
                            self.last_frame_sent = current_time
                
                # Sleep for a short time to get smoother video (30 FPS approx)
                time.sleep(0.03)
    
    def capture_audio_thread(self):
        """Thread to capture audio input"""
        try:
            mic_info = pya.get_default_input_device_info()
            self.audio_stream = pya.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE,
            )
            
            while self.running:
                data = self.audio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                if not self.out_queue.full():
                    self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
                    
        except Exception as e:
            st.error(f"Audio capture error: {e}")
        finally:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
    
    def play_audio_thread(self):
        """Thread to play audio responses"""
        try:
            output_stream = pya.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
            )
            
            while self.running:
                try:
                    if not self.audio_in_queue.empty():
                        data = self.audio_in_queue.get(timeout=0.1)
                        output_stream.write(data)
                    else:
                        time.sleep(0.01)
                except queue.Empty:
                    time.sleep(0.01)
                    
        except Exception as e:
            st.error(f"Audio playback error: {e}")
        finally:
            output_stream.close()

    async def send_data_task(self):
        """Async task to send data to model"""
        while self.running:
            try:
                if not self.out_queue.empty():
                    data = self.out_queue.get(timeout=0.1)
                    await self.session.send(input=data)
                await asyncio.sleep(0.01)
            except queue.Empty:
                await asyncio.sleep(0.01)
            except Exception as e:
                st.error(f"Error sending data: {e}")
    
    async def receive_responses_task(self):
        """Async task to receive model responses"""
        while self.running:
            try:
                turn = self.session.receive()
                async for response in turn:
                    if data := response.data:
                        self.audio_in_queue.put(data)
                    if text := response.text:
                        self.response_queue.put(text)
                
                # Clear audio queue on turn complete
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()
                    
            except Exception as e:
                st.error(f"Error receiving response: {e}")
                await asyncio.sleep(1)
    
    async def session_manager(self):
        """Manages the Gemini session"""
        try:
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                self.session = session
                
                # System prompt
                system_prompt = ("You are a helpful assistant. Engage in a relaxed and natural conversation. "
                                "Do not repeatedly ask if the user has a problem unless it is explicitly mentioned. "
                                "Start your conversation by saying hello.")
                await self.session.send(input=system_prompt, end_of_turn=True)
                
                # Start tasks for sending/receiving data
                send_task = asyncio.create_task(self.send_data_task())
                receive_task = asyncio.create_task(self.receive_responses_task())
                
                # Wait until not running
                while self.running:
                    await asyncio.sleep(0.1)
                    
                # Clean up
                send_task.cancel()
                receive_task.cancel()
                
        except Exception as e:
            st.error(f"Session error: {e}")
            traceback.print_exc()
    
    def start(self):
        """Start all threads and async tasks"""
        if self.running:
            return
            
        self.running = True
        self.last_frame_sent = 0
        
        # Start capture threads
        if self.video_mode != "none":
            video_thread = threading.Thread(target=self.capture_video_thread)
            video_thread.daemon = True
            video_thread.start()
            self.threads.append(video_thread)
            
        audio_capture_thread = threading.Thread(target=self.capture_audio_thread)
        audio_capture_thread.daemon = True
        audio_capture_thread.start()
        self.threads.append(audio_capture_thread)
        
        audio_playback_thread = threading.Thread(target=self.play_audio_thread)
        audio_playback_thread.daemon = True
        audio_playback_thread.start()
        self.threads.append(audio_playback_thread)
        
        # Start async session in a separate thread
        def run_async_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.session_manager())
            
        session_thread = threading.Thread(target=run_async_loop)
        session_thread.daemon = True
        session_thread.start()
        self.threads.append(session_thread)
    
    def stop(self):
        """Stop all threads and async tasks"""
        self.running = False
        
        # Let threads terminate naturally
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
                
        self.threads = []
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None

def main():
    st.set_page_config(page_title="Gemini Live Assistant", layout="centered")
    
    # Init session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'app_running' not in st.session_state:
        st.session_state.app_running = False
    if 'app' not in st.session_state:
        st.session_state.app = None
    
    st.markdown("### Gemini Live Assistant")
    
    # Settings
    with st.expander("Settings", expanded=not st.session_state.app_running):
        # Use form to ensure consistent alignment
        with st.container():
            # Video mode
            video_mode = st.selectbox(
                "Video Input Mode",
                options=["camera", "screen", "none"],
                index=0,
                disabled=st.session_state.app_running
            )
            
            st.write("")
            
            # controls
            if not st.session_state.app_running:
                if st.button("Start Session", type="primary", use_container_width=True):
                    st.session_state.app = StreamlitGeminiLive(video_mode=video_mode)
                    st.session_state.app.start()
                    st.session_state.app_running = True
                    st.session_state.messages = []
                    st.rerun()
            else:
                if st.button("Stop Session", type="secondary", use_container_width=True):
                    if st.session_state.app:
                        st.session_state.app.stop()
                    st.session_state.app_running = False
                    st.session_state.app = None
                    st.rerun()
    
    # Video feed in a contained size
    if st.session_state.app_running and video_mode != "none":
        st.markdown("#### Video Feed")
        cols = st.columns([1, 2, 1])
        video_placeholder = cols[1].empty()
        
        app = st.session_state.app
        if app.current_frame is not None:
            video_placeholder.image(app.current_frame, channels="RGB", use_container_width=True)
    
    if st.session_state.app_running and st.session_state.app:
        if not st.session_state.app.response_queue.empty():
            response = st.session_state.app.response_queue.get()
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    if st.session_state.app_running:
        refresh_placeholder = st.empty()
        time.sleep(0.1)  
        st.rerun()

if __name__ == "__main__":
    main()