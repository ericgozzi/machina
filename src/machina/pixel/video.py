import ffmpeg
import numpy as np

class Video:


    def __init__(self, filepath: str):
        self.filepath = filepath
        self.stream = ffmpeg.input(filepath)  # keep stream inside object



    def resize(self, width: int, height: int):
        """
        Apply resize filter (does not export yet).
        """
        self.stream = self.stream.filter('scale', width, height)
        return self
    


    def crop(self, width: int, height: int, x: int = 0, y: int = 0):
        """
        Apply crop filter (does not export yet).
        """
        self.stream = self.stream.filter('crop', width, height, x, y)
        return self
    


    def convert(self, codec: str = "libx264", preset: str = "fast", crf: int = 23):
        """
        Set conversion parameters (does not export yet).
        """
        self.stream = ffmpeg.output(
            self.stream, 'pipe:', vcodec=codec, preset=preset, crf=crf, format='mp4'
        )
        return self
    


    def extract_audio(self, audio_codec: str = "mp3"):
        """
        Extract audio (does not export yet).
        """
        self.stream = ffmpeg.output(self.stream, 'pipe:', vn=True, acodec=audio_codec, format=audio_codec)
        return self
    


    
    def get_frames_every(self, interval: float) -> list:
        """
        Extract frames every `interval` seconds and return as a list of numpy arrays.
        
        :param interval: Time in seconds between frames
        :return: List of frames as numpy arrays (H x W x C)
        """
        # Probe video to get frame rate and duration
        probe = ffmpeg.probe(self.filepath)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        duration = float(probe['format']['duration'])
        
        # Calculate fps for ffmpeg filter to extract 1 frame every `interval` seconds
        fps = 1 / interval
        
        out, _ = (
            ffmpeg
            .input(self.filepath)
            .filter('fps', fps=fps)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True)
        )
        
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        # Convert raw bytes to numpy arrays
        frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        return frames
    



    def export(self, output_path: str):
        """
        Actually export the video to a file.
        """
        ffmpeg.output(self.stream, output_path).run()
        return output_path
    


