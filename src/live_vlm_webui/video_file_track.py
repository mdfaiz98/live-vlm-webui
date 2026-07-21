"""
Video File Track for local file playback support

This module provides a VideoStreamTrack implementation for local video files,
following the same pattern as rtsp_track.py's RTSPVideoTrack - allowing
live-vlm-webui to process a pre-recorded video file through the same
pipeline as webcam/RTSP input.

Unlike RTSP (which is naturally paced by network delivery) or a live webcam
(paced by the camera's own frame rate), decoding a local file happens far
faster than real time, so this track has to actively pace frames to the
file's own timestamps - otherwise it would blast through the whole video
in a couple of seconds.

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import av
import asyncio
import logging
import time
from typing import Optional
from aiortc import VideoStreamTrack
from av import VideoFrame

av.logging.set_level(av.logging.FATAL)

logger = logging.getLogger(__name__)


class VideoFileTrack(VideoStreamTrack):
    """
    Video track that reads from a local video file and converts frames to
    aiortc VideoFrame, pacing playback to the file's real frame timing and
    optionally looping back to the start at end-of-file.

    Example:
        track = VideoFileTrack("/path/to/video.mp4", loop=True)
        frame = await track.recv()
    """

    def __init__(
        self,
        file_path: str,
        loop: bool = True,
        options: Optional[dict] = None,
    ):
        """
        Initialize video file track.

        Args:
            file_path: Absolute path to a local video file (mp4, mov, mkv, etc.)
            loop: Whether to restart from the beginning at end-of-file (default: True)
            options: Additional PyAV container options
        """
        super().__init__()
        self.file_path = file_path
        self.loop = loop
        self.container: Optional[av.container.InputContainer] = None
        self.stream: Optional[av.video.VideoStream] = None
        self._stopped = False
        self._frame_count = 0

        # Pacing state - tracks wall-clock vs. stream-time to avoid decoding
        # (and therefore "playing") the file faster than its real duration
        self._playback_start_wallclock: Optional[float] = None
        self._first_frame_pts: Optional[int] = None

        self.options = options or {}

        self._connect()

    def _connect(self):
        """Open the video file and locate its video stream."""
        try:
            logger.info(f"Opening video file: {self.file_path}")

            self.container = av.open(self.file_path, options=self.options)

            if not self.container.streams.video:
                raise ValueError("No video stream found in file")

            self.stream = self.container.streams.video[0]

            codec = self.stream.codec_context.name
            width = self.stream.width or "unknown"
            height = self.stream.height or "unknown"
            fps = self.stream.average_rate or "unknown"
            duration = (
                float(self.stream.duration * self.stream.time_base)
                if self.stream.duration
                else "unknown"
            )

            logger.info(
                f"Video file opened: {codec} {width}x{height} @{fps}fps, duration={duration}s"
            )

        except Exception as e:
            logger.error(f"Failed to open video file {self.file_path}: {e}")
            raise

    async def recv(self) -> VideoFrame:
        """
        Receive next frame from the video file, paced to real playback speed.

        Returns:
            VideoFrame: Next decoded video frame

        Raises:
            StopAsyncIteration: When the file ends and looping is disabled
        """
        if self._stopped:
            raise StopAsyncIteration

        loop_asyncio = asyncio.get_event_loop()
        frame = await loop_asyncio.run_in_executor(None, self._read_frame)

        if frame is None:
            if self.loop and not self._stopped:
                logger.info("Video file reached end, looping back to start")
                await loop_asyncio.run_in_executor(None, self._restart)
                self._playback_start_wallclock = None
                self._first_frame_pts = None
                frame = await loop_asyncio.run_in_executor(None, self._read_frame)
                if frame is None:
                    raise StopAsyncIteration
            else:
                raise StopAsyncIteration

        await self._pace_frame(frame)

        # Use aiortc's own monotonically-increasing clock for the outgoing
        # timestamp, rather than the file's raw internal pts. The file's pts
        # resets to near-zero every time we loop back to the start, which the
        # downstream VP8/H264 encoder correctly rejects if handed directly
        # ("pts is smaller than initial pts") - next_timestamp() keeps the
        # outbound stream's timeline continuous regardless of how many times
        # the underlying file has looped.
        frame.pts, frame.time_base = await self.next_timestamp()

        self._frame_count += 1
        if self._frame_count % 300 == 0:
            logger.debug(f"Video file: played {self._frame_count} frames")

        return frame

    async def _pace_frame(self, frame: VideoFrame):
        """
        Sleep as needed so frames are delivered at the file's real playback
        speed rather than as fast as they can be decoded.
        """
        if frame.pts is None or self.stream is None or self.stream.time_base is None:
            return  # no timing info available, just pass the frame through

        now = time.monotonic()

        if self._playback_start_wallclock is None:
            # First frame of this playback pass - anchor wall-clock to stream time
            self._playback_start_wallclock = now
            self._first_frame_pts = frame.pts
            return

        elapsed_stream_time = float((frame.pts - self._first_frame_pts) * self.stream.time_base)
        target_wallclock = self._playback_start_wallclock + elapsed_stream_time
        delay = target_wallclock - now

        if delay > 0:
            await asyncio.sleep(delay)

    def _read_frame(self) -> Optional[VideoFrame]:
        """Read and decode the next frame (blocking; run in executor)."""
        if not self.container or not self.stream:
            logger.error("Cannot read frame: container or stream not initialized")
            return None

        try:
            for packet in self.container.demux(self.stream):
                for frame in packet.decode():
                    if isinstance(frame, VideoFrame):
                        return frame
            return None  # end of file
        except av.error.EOFError:
            return None
        except Exception as e:
            logger.error(f"Error decoding video file frame: {e}")
            return None

    def _restart(self):
        """Seek back to the beginning of the file for looping."""
        try:
            if self.container:
                self.container.seek(0)
        except Exception as e:
            logger.warning(f"Error seeking to start of file, reopening instead: {e}")
            try:
                if self.container:
                    self.container.close()
            except Exception:
                pass
            self._connect()

    def stop(self):
        """Stop playback and release the file handle."""
        self._stopped = True

        if self.container:
            try:
                self.container.close()
                logger.info(f"Video file closed: {self._frame_count} frames played")
            except Exception as e:
                logger.warning(f"Error closing video file container: {e}")
            finally:
                self.container = None
                self.stream = None

        super().stop()

    @property
    def is_connected(self) -> bool:
        """Check if the video file is currently open."""
        return self.container is not None and not self._stopped

    def get_stats(self) -> dict:
        """Get statistics about the video file playback."""
        stats = {
            "file_path": self.file_path,
            "connected": self.is_connected,
            "frames_played": self._frame_count,
            "stopped": self._stopped,
            "loop": self.loop,
        }

        if self.stream:
            stats.update(
                {
                    "codec": self.stream.codec_context.name,
                    "width": self.stream.width,
                    "height": self.stream.height,
                    "fps": float(self.stream.average_rate) if self.stream.average_rate else None,
                }
            )

        return stats
