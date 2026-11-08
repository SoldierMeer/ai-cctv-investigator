"""Vision pipeline for AI CCTV Investigator.

Stage 1: YOLOv8 Person Detection, ByteTrack Tracking, and 3-second Laplacian Crop Sampling.
"""

from .detector import PersonDetector
from .tracker import PedestrianTracker
from .crop_sampler import TrackletCropSampler

__all__ = ["PersonDetector", "PedestrianTracker", "TrackletCropSampler"]
