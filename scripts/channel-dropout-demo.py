"""
Apply random channel dropout on log-scale Mel-spectra for some utterance and plot spectrograms.
"""
import argparse
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import lidbox.features.audio as audio_feat


frame_length_ms = 25
frame_step_ms = 10
num_mel_bins = 64
channel_dropout_rate = 0.5
img_scale = 0.2
dropout_samples = 4


def tf_print(*args):
    return tf.print(*args, summarize=-1, output_stream=sys.stdout)

def compute_mag_spec(signal, sample_rate):
    return audio_feat.spectrograms(
            [signal],
            sample_rate,
            frame_length_ms,
            frame_step_ms,
            power=1.0)

def compute_logmel_spec(signal, sample_rate):
    mag_spec = compute_mag_spec(signal, sample_rate)
    pow_spec = tf.math.pow(mag_spec, 2.0)
    mel_spec = audio_feat.melspectrograms(
            pow_spec,
            sample_rate,
            num_mel_bins=num_mel_bins,
            fmin=20.0,
            fmax=8000.0)
    return tf.math.log(mel_spec + 1e-6)

def channels_mean_centered(t):
    tf.debugging.assert_rank(t, 2)
    return t - tf.math.reduce_mean(t, axis=0, keepdims=True)

def write_ax(name, t, uttid, output_dir, cmap="viridis"):
    img = tf.transpose(t).numpy()
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(img_scale * img.shape[0], img_scale * img.shape[1]))
    ax.imshow(img, cmap=cmap, norm=mcolors.Normalize(), aspect="equal")
    ax.set_xlim(0, img.shape[1])
    ax.set_axis_off()
    ax.invert_yaxis()
    img_path = os.path.join(output_dir, "{}-{}.png".format(uttid, name))
    fig.savefig(img_path, bbox_inches="tight", pad_inches=0, dpi=200)
    print("wrote img of shape {} as '{}'".format(img.shape, img_path))
    plt.close(fig)

def main(uttid, signal_path, output_dir):
    assert os.path.exists(signal_path), signal_path
    assert os.path.isdir(output_dir), output_dir
    print("loading uttid {} from '{}'".format(uttid, signal_path))
    wav = tf.audio.decode_wav(tf.io.read_file(signal_path))
    signal = tf.math.reduce_mean(wav.audio, axis=1)
    tf_print("loaded mono signal of shape", tf.shape(signal), "and sample rate", wav.sample_rate)
    logmel_spec = compute_logmel_spec(signal, wav.sample_rate)
    logmel_spec = channels_mean_centered(tf.squeeze(logmel_spec, 0))
    tf_print("computed log-mel spectrogram of shape", tf.shape(logmel_spec))
    write_ax("logmel-spectrogram", logmel_spec, uttid, output_dir)
    logmel_spec = tf.expand_dims(logmel_spec, 0)
    dropout = tf.keras.layers.SpatialDropout1D(channel_dropout_rate)
    for i in range(dropout_samples):
        logmel_spec_dropout = dropout(logmel_spec, training=True)
        logmel_spec_dropout = tf.squeeze(logmel_spec_dropout, 0)
        write_ax("logmel-spectrogram-channeldropout50-{}".format(i + 1), logmel_spec_dropout, uttid, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("uttid")
    parser.add_argument("signal_path")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    main(args.uttid, args.signal_path, args.output_dir)
