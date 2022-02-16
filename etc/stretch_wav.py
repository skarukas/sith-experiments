from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
import os
from os.path import join
import shutil
import tqdm
import sys


def stretch_wav(in_fname, out_fname, speed):
    with WavReader(in_fname) as reader:
        with WavWriter(out_fname, reader.channels, reader.samplerate) as writer:
            tsm = phasevocoder(reader.channels, speed=speed)
            tsm.run(reader, writer)


def create_stretched_copy(data_dir, out_dir, scale):
    for root, dirs, files in os.walk(data_dir):
        for name in dirs:
            sub_dir = root.replace(data_dir, "")
            root_out = join(out_dir, scale, sub_dir)
            os.makedirs(join(root_out, name), exist_ok=True)
        
        for name in tqdm.tqdm(files, desc=root):
            sub_dir = root.replace(data_dir, "")
            root_out = join(out_dir, scale, sub_dir)
            in_fname = join(root, name)
            out_fname = join(root_out, name)
            if os.path.exists(out_fname):
                continue
            if name.endswith(".wav"):
                stretch_wav(in_fname, out_fname, float(scale))
            else:
                shutil.copy(in_fname, out_fname)

if __name__ == "__main__":
    data_dir = "data/SpeechCommands/speech_commands_v0.02/"
    out_dir = "data/SpeechCommands/speech_commands/"
    scale = sys.argv[1]
    print(f"Outputting files to {join(out_dir, scale)}")
    create_stretched_copy(data_dir, out_dir, scale)