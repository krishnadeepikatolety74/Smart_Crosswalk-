import threading
import queue
import time
import pyttsx3

class AudioManager:
    def __init__(self):
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.last_spoken = {}
        
    def _worker(self):
        # Initialize engine in the worker thread for thread safety
        engine = pyttsx3.init()
        while True:
            msg, vol = self.q.get()
            if msg is None:
                break
            try:
                engine.setProperty('volume', vol / 100.0)
                engine.say(msg)
                engine.runAndWait()
            except Exception as e:
                print(f"Audio error: {e}")
            self.q.task_done()
            
    def speak(self, message, volume=100, cooldown=5):
        current_time = time.time()
        if message in self.last_spoken:
            if current_time - self.last_spoken[message] < cooldown:
                return
        self.last_spoken[message] = current_time
        
        # Keep queue small to avoid backlog of old alerts
        if self.q.qsize() < 2:
            self.q.put((message, int(volume)))

audio_manager = AudioManager()
