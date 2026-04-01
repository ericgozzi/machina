import os

import ffmpeg


class Video:
    def __init__(self, input_path: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file '{input_path}' not found")

        self._input_path = input_path
        # We start by getting the input object
        self._input = ffmpeg.input(input_path)
        # Separate the streams so we can filter them in parallel
        self._v = self._input.video
        self._a = self._input.audio

    def resize(self, width, height):
        """Resizes the video track."""
        self._v = self._v.filter("scale", width, height)
        return self

    def trim(self, start_time, end_time):
        """Trims both video and audio tracks to keep them in sync."""
        self._v = self._v.trim(start=start_time, end=end_time).setpts("PTS-STARTPTS")
        self._a = self._a.filter("atrim", start=start_time, end=end_time).filter(
            "asetpts", "PTS-STARTPTS"
        )
        return self

    def save(self, output_path: str, vcodec="libx264", acodec="aac", **kwargs):
        """Compiles the tracks and runs the FFmpeg process."""
        try:
            # Join the modified video and audio tracks back together
            output_stream = ffmpeg.output(
                self._v, self._a, output_path, vcodec=vcodec, acodec=acodec, **kwargs
            )

            ffmpeg.run(
                output_stream,
                overwrite_output=True,
                capture_stdout=True,
                capture_stderr=True,
            )
            return f"Success: {output_path} saved."

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            # It's often better to raise the error or log it than just return a string
            raise RuntimeError(f"FFmpeg Error: {error_msg}")
