## transformers audio_utils. 

# Copyright 2023 The HuggingFace Team. All rights reserved.
import datetime
import platform
import subprocess
from typing import Optional, Tuple, Union

import numpy as np


def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    Helper function to read an audio file through ffmpeg.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    format_for_conversion = "f32le"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    try:
        with subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as ffmpeg_process:
            output_stream = ffmpeg_process.communicate(bpayload)
    except FileNotFoundError as error:
        raise ValueError("ffmpeg was not found but is required to load audio files from filename") from error
    out_bytes = output_stream[0]
    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError(
            "Soundfile is either not in the correct format or is malformed. Ensure that the soundfile has "
            "a valid audio file extension (e.g. wav, flac or mp3) and is not corrupted. If reading from a remote "
            "URL, ensure that the URL is the full address to **download** the audio file."
        )
    return audio


def ffmpeg_microphone(
    sampling_rate: int,
    chunk_length_s: float,
    format_for_conversion: str = "f32le",
):
    """
    Helper function to read raw microphone data.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    if format_for_conversion == "s16le":
        size_of_sample = 2
    elif format_for_conversion == "f32le":
        size_of_sample = 4
    else:
        raise ValueError(f"Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`")

    system = platform.system()
    if system == "Linux":
        format_ = "alsa"
        input_ = "default"
    elif system == "Darwin":
        format_ = "avfoundation"
        input_ = ":0"
    elif system == "Windows":
        format_ = "dshow"
        input_ = _get_microphone_name()

    ffmpeg_command = [
        "ffmpeg",
        "-f",
        format_,
        "-i",
        input_,
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-fflags",
        "nobuffer",
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]
    chunk_len = int(round(sampling_rate * chunk_length_s)) * size_of_sample
    iterator = _ffmpeg_stream(ffmpeg_command, chunk_len)
    for item in iterator:
        yield item


def ffmpeg_microphone_live(
    sampling_rate: int,
    chunk_length_s: float,
    stream_chunk_s: Optional[int] = None,
    stride_length_s: Optional[Union[Tuple[float, float], float]] = None,
    format_for_conversion: str = "f32le",
):
    """
    Helper function to read audio from the microphone file through ffmpeg. This will output `partial` overlapping
    chunks starting from `stream_chunk_s` (if it is defined) until `chunk_length_s` is reached. It will make use of
    striding to avoid errors on the "sides" of the various chunks.

    Arguments:
        sampling_rate (`int`):
            The sampling_rate to use when reading the data from the microphone. Try using the model's sampling_rate to
            avoid resampling later.
        chunk_length_s (`float` or `int`):
            The length of the maximum chunk of audio to be sent returned. This includes the eventual striding.
        stream_chunk_s (`float` or `int`)
            The length of the minimal temporary audio to be returned.
        stride_length_s (`float` or `int` or `(float, float)`, *optional*, defaults to `None`)
            The length of the striding to be used. Stride is used to provide context to a model on the (left, right) of
            an audio sample but without using that part to actually make the prediction. Setting this does not change
            the length of the chunk.
        format_for_conversion (`str`, defalts to `f32le`)
            The name of the format of the audio samples to be returned by ffmpeg. The standard is `f32le`, `s16le`
            could also be used.
    Return:
        A generator yielding dictionaries of the following form

        `{"sampling_rate": int, "raw": np.array(), "partial" bool}` With optionnally a `"stride" (int, int)` key if
        `stride_length_s` is defined.

        `stride` and `raw` are all expressed in `samples`, and `partial` is a boolean saying if the current yield item
        is a whole chunk, or a partial temporary result to be later replaced by another larger chunk.


    """
    if stream_chunk_s is not None:
        chunk_s = stream_chunk_s
    else:
        chunk_s = chunk_length_s

    microphone = ffmpeg_microphone(sampling_rate, chunk_s, format_for_conversion=format_for_conversion)
    if format_for_conversion == "s16le":
        dtype = np.int16
        size_of_sample = 2
    elif format_for_conversion == "f32le":
        dtype = np.float32
        size_of_sample = 4
    else:
        raise ValueError(f"Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`")

    if stride_length_s is None:
        stride_length_s = chunk_length_s / 6
    chunk_len = int(round(sampling_rate * chunk_length_s)) * size_of_sample
    if isinstance(stride_length_s, (int, float)):
        stride_length_s = [stride_length_s, stride_length_s]

    stride_left = int(round(sampling_rate * stride_length_s[0])) * size_of_sample
    stride_right = int(round(sampling_rate * stride_length_s[1])) * size_of_sample
    audio_time = datetime.datetime.now()
    delta = datetime.timedelta(seconds=chunk_s)
    for item in chunk_bytes_iter(microphone, chunk_len, stride=(stride_left, stride_right), stream=True):
        # Put everything back in numpy scale
        item["raw"] = np.frombuffer(item["raw"], dtype=dtype)
        item["stride"] = (
            item["stride"][0] // size_of_sample,
            item["stride"][1] // size_of_sample,
        )
        item["sampling_rate"] = sampling_rate
        audio_time += delta
        if datetime.datetime.now() > audio_time + 10 * delta:
            # We're late !! SKIP
            continue
        yield item


def chunk_bytes_iter(iterator, chunk_len: int, stride: Tuple[int, int], stream: bool = False):
    """
    Reads raw bytes from an iterator and does chunks of length `chunk_len`. Optionally adds `stride` to each chunks to
    get overlaps. `stream` is used to return partial results even if a full `chunk_len` is not yet available.
    """
    acc = b""
    stride_left, stride_right = stride
    if stride_left + stride_right >= chunk_len:
        raise ValueError(
            f"Stride needs to be strictly smaller than chunk_len: ({stride_left}, {stride_right}) vs {chunk_len}"
        )
    _stride_left = 0
    for raw in iterator:
        acc += raw
        if stream and len(acc) < chunk_len:
            stride = (_stride_left, 0)
            yield {"raw": acc[:chunk_len], "stride": stride, "partial": True}
        else:
            while len(acc) >= chunk_len:
                # We are flushing the accumulator
                stride = (_stride_left, stride_right)
                item = {"raw": acc[:chunk_len], "stride": stride}
                if stream:
                    item["partial"] = False
                yield item
                _stride_left = stride_left
                acc = acc[chunk_len - stride_left - stride_right :]
    # Last chunk
    if len(acc) > stride_left:
        item = {"raw": acc, "stride": (_stride_left, 0)}
        if stream:
            item["partial"] = False
        yield item


def _ffmpeg_stream(ffmpeg_command, buflen: int):
    """
    Internal function to create the generator of data through ffmpeg
    """
    bufsize = 2**24  # 16Mo
    try:
        with subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, bufsize=bufsize) as ffmpeg_process:
            while True:
                raw = ffmpeg_process.stdout.read(buflen)
                if raw == b"":
                    break
                yield raw
    except FileNotFoundError as error:
        raise ValueError("ffmpeg was not found but is required to stream audio files from filename") from error


def _get_microphone_name():
    """
    Retrieve the microphone name in Windows .
    """
    command = ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", ""]

    try:
        ffmpeg_devices = subprocess.run(command, text=True, stderr=subprocess.PIPE, encoding="utf-8")
        microphone_lines = [line for line in ffmpeg_devices.stderr.splitlines() if "(audio)" in line]

        if microphone_lines:
            microphone_name = microphone_lines[0].split('"')[1]
            print(f"Using microphone: {microphone_name}")
            return f"audio={microphone_name}"
    except FileNotFoundError:
        print("ffmpeg was not found. Please install it or make sure it is in your system PATH.")

    return "default"


## from pitch tonic

class FormatTimeStamp:
    def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = "."):
        if seconds is not None:
            milliseconds = round(seconds * 1000.0)

            hours = milliseconds // 3_600_000
            milliseconds -= hours * 3_600_000

            minutes = milliseconds // 60_000
            milliseconds -= minutes * 60_000

            seconds = milliseconds // 1_000
            milliseconds -= seconds * 1_000

            hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
            return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
        else:
            # we have a malformed timestamp so just return it as is
            return seconds

# ### From Diarizers : https://github.com/BuildTonic/diarizers


### Batch Align 

import torch
from transformers.models.whisper.generation_whisper import _dynamic_time_warping as _dynamic_time_warping
from transformers.models.whisper.generation_whisper import _median_filter as _median_filter

from dataclasses import dataclass
import numpy as np

def _extract_token_timestamps(self, generate_outputs, alignment_heads, time_precision=0.02, num_frames=None):
    """
    Calculates token-level timestamps using the encoder-decoder cross-attentions and dynamic time-warping (DTW) to
    map each output token to a position in the input audio. If `num_frames` is specified, the encoder-decoder
    cross-attentions will be cropped before applying DTW.

    Returns:
        tensor containing the timestamps in seconds for each predicted token
    """
    # Create a list with `decoder_layers` elements, each a tensor of shape
    # (batch size, attention_heads, output length, input length).
    cross_attentions = []
    for i in range(self.config.decoder_layers):
        cross_attentions.append(torch.cat([x[i] for x in generate_outputs.cross_attentions], dim=2))

    # Select specific cross-attention layers and heads. This is a tensor
    # of shape (batch size, num selected, output length, input length).
    weights = torch.stack([cross_attentions[l][:, h] for l, h in alignment_heads])
    weights = weights.permute([1, 0, 2, 3])
    if num_frames is not None:
        weights = weights[..., : num_frames // 2]

    # Normalize and smoothen the weights.
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    weights = (weights - mean) / std
    weights = _median_filter(weights, self.config.median_filter_width)

    # Average the different cross-attention heads.
    matrix = weights.mean(dim=1)

    timestamps = torch.zeros_like(generate_outputs.sequences, dtype=torch.float32)

    # Perform dynamic time warping on each element of the batch.
    for batch_idx in range(timestamps.shape[0]):
        text_indices, time_indices = _dynamic_time_warping(-matrix[batch_idx].float().cpu().numpy())
        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        jump_times = time_indices[jumps] * time_precision
        timestamps[batch_idx, 1:] = torch.tensor(jump_times)

    return timestamps


@dataclass
class ASRAudioFile:
    file : str
    tensor : torch.Tensor
    rate : int

    def chunk(self,begin_ms, end_ms):
        """Get a chunk of the audio.

        Parameters
        ----------
        begin_ms : int
            Milliseconds of the start of the slice.
        end_ms : int
            Milliseconds of the end of the slice.

        Returns
        -------
        torch.Tensor
            The returned chunk to supply to the ASR engine.
        """

        data = self.tensor[int(round((begin_ms/1000)*self.rate)):
                           int(round((end_ms/1000)*self.rate))]

        return data

    def all(self):
        """Get the audio in its entirety

        Notes
        -----
        like `chunk()` but all of the audio
        """

        return self.tensor

# # Adapted from https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/tasks/segmentation/speaker_diarization.py
# # MIT License
# #
# # Copyright (c) 2020- CNRS
# #
# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:
# #
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.
# import math

# import numpy as np
# import torch

# # from ..models import SegmentationModel


# class Preprocess:
#     """Converts a HF dataset with the following features:
#         - "audio": Audio feature.
#         - "speakers": The list of audio speakers, with their order of appearance.
#         - "timestamps_start": A list of timestamps indicating the start of each speaker segment.
#        flake8>=3.8.3
#         - "timestamps_end": A list of timestamps indicating the end of each speaker segment.
#     to a preprocessed dataset ready to be used with the HF Trainer.
#     """

#     def __init__(
#         self,
#         config,
#     ):
#         """Preprocess init method.
#         Takes as input the dataset to process and the model to perform training with.
#         The preprocessing is done to fit the hyperparameters of the model.
#         Args:
#             input_dataset (dataset): Hugging Face Speaker Diarization dataset
#             model (SegmentationModel): A SegmentationModel from the diarizers library.
#         """
#         self.chunk_duration = config.chunk_duration
#         self.max_speakers_per_frame = config.max_speakers_per_frame
#         self.max_speakers_per_chunk = config.max_speakers_per_chunk
#         self.min_duration = config.min_duration
#         self.warm_up = config.warm_up

#         self.sample_rate = config.sample_rate
#         # model = SegmentationModel(config).to_pyannote_model()

#         # Get the number of frames associated to a chunk:
#         # _, self.num_frames_per_chunk, _ = model(torch.rand((1, int(self.chunk_duration * self.sample_rate)))).shape

#     def get_labels_in_file(self, file):
#         """Get speakers in file.
#         Args:
#             file (_type_): dataset row from input dataset.

#         Returns:
#             file_labels (list): a list of all speakers in the audio file.
#         """

#         file_labels = []
#         for i in range(len(file["speakers"][0])):
#             if file["speakers"][0][i] not in file_labels:
#                 file_labels.append(file["speakers"][0][i])

#         return file_labels

#     def get_segments_in_file(self, file, labels):
#         """Get segments in file.

#         Args:
#             file (_type_): dataset row from input dataset.
#             labels (_type_):  a list of all speakers in the audio file.

#         Returns:
#             annotations (numpy array): _description_
#         """

#         file_annotations = []

#         for i in range(len(file["timestamps_start"][0])):
#             start_segment = file["timestamps_start"][0][i]
#             end_segment = file["timestamps_end"][0][i]
#             label = labels.index(file["speakers"][0][i])
#             file_annotations.append((start_segment, end_segment, label))

#         dtype = [("start", "<f4"), ("end", "<f4"), ("labels", "i1")]

#         annotations = np.array(file_annotations, dtype)

#         return annotations

#     def get_chunk(self, file, start_time):
#         """Method used to get an audio chunk from an audio file given a start_time.

#         Args:
#             file (dict): dataset row containing the "audio" feature.
#             start_time (float): start time (in seconds) of the audio_chunk to extract.

#         Returns:
#             waveform (array): audio chunk
#             y (numpy array): target array.
#             labels (list): list of speakers in chunk.
#         """

#         sample_rate = file["audio"][0]["sampling_rate"]

#         assert sample_rate == self.sample_rate

#         end_time = start_time + self.chunk_duration
#         start_frame = math.floor(start_time * sample_rate)
#         num_frames_waveform = math.floor(self.chunk_duration * sample_rate)
#         end_frame = start_frame + num_frames_waveform

#         waveform = file["audio"][0]["array"][start_frame:end_frame]

#         labels = self.get_labels_in_file(file)

#         file_segments = self.get_segments_in_file(file, labels)

#         chunk_segments = file_segments[(file_segments["start"] < end_time) & (file_segments["end"] > start_time)]

#         # compute frame resolution:
#         resolution = self.chunk_duration / self.num_frames_per_chunk

#         # discretize chunk annotations at model output resolution
#         start = np.maximum(chunk_segments["start"], start_time) - start_time
#         start_idx = np.floor(start / resolution).astype(int)
#         end = np.minimum(chunk_segments["end"], end_time) - start_time
#         end_idx = np.ceil(end / resolution).astype(int)

#         # get list and number of labels for current scope
#         labels = list(np.unique(chunk_segments["labels"]))
#         num_labels = len(labels)
#         # initial frame-level targets
#         y = np.zeros((self.num_frames_per_chunk, num_labels), dtype=np.uint8)

#         # map labels to indices
#         mapping = {label: idx for idx, label in enumerate(labels)}

#         for start, end, label in zip(start_idx, end_idx, chunk_segments["labels"]):
#             mapped_label = mapping[label]
#             y[start:end, mapped_label] = 1

#         return waveform, y, labels

#     def get_start_positions(self, file, overlap, random=False):
#         """Get start positions from the audio_chunks in the input audio file.

#         Args:
#             file (dict): dataset row containing the "audio" feature.
#             overlap (float, optional):  Overlap between successive start positions.
#             random (bool, optional):  Whether or not to randomly select chunks in the audio file. Defaults to False.

#         Returns:
#             start_positions: Numpy array containing the start positions of the audio chunks in file.
#         """

#         sample_rate = file["audio"][0]["sampling_rate"]

#         assert sample_rate == self.sample_rate

#         file_duration = len(file["audio"][0]["array"]) / sample_rate
#         start_positions = np.arange(0, file_duration - self.chunk_duration, self.chunk_duration * (1 - overlap))

#         if random:
#             nb_samples = int(file_duration / self.chunk_duration)
#             start_positions = np.random.uniform(0, file_duration, nb_samples)

#         return start_positions

#     def __call__(self, file, random=False, overlap=0.0):
#         """Chunk an audio file into short segments of duration self.chunk_duration

#         Args:
#             file (dict): dataset row containing the "audio" feature.
#             random (bool, optional): Whether or not to randomly select chunks in the audio file. Defaults to False.
#             overlap (float, optional):  Overlap between successive chunks. Defaults to 0.0.

#         Returns:
#             new_batch: new batch containing for each chunk the corresponding waveform, labels and number of speakers.
#         """

#         new_batch = {"waveforms": [], "labels": [], "nb_speakers": []}

#         if random:
#             start_positions = self.get_start_positions(file, overlap, random=True)
#         else:
#             start_positions = self.get_start_positions(file, overlap)

#         for start_time in start_positions:
#             waveform, target, label = self.get_chunk(file, start_time)

#             new_batch["waveforms"].append(waveform)
#             new_batch["labels"].append(target)
#             new_batch["nb_speakers"].append(label)

#         return new_batch

