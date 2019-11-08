import librosa
import json
import math
import numpy as np
import os
import scipy.signal
import torch
import random
import logging
import epitran
import scipy.io.wavfile as wav

from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from utils.audio import load_audio, get_audio_length, audio_with_sox, augment_audio_with_sox, load_randomly_augmented_audio
from bpemb import BPEmb
from python_speech_features import mfcc
from python_speech_features import logfbank

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

ipa_map = {'en': 'eng-Latn', 'es': 'spa-Latn', 'fr': 'fra-Latn', 'pl': 'pol-Latn', 'ru': 'rus-Cyrl', 'ro': 'ron-Latn', 'sv': 'swe-Latn', 'hu':'hun-Latn'}

class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


class SpectrogramParser(AudioParser):
    def __init__(self, audio_conf, normalize=False, augment=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        super(SpectrogramParser, self).__init__()
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.normalize = normalize
        self.augment = augment
        self.noiseInjector = NoiseInjection(audio_conf['noise_dir'], self.sample_rate,
                                            audio_conf['noise_levels']) if audio_conf.get(
            'noise_dir') is not None else None
        self.noise_prob = audio_conf.get('noise_prob')

    def parse_audio(self, audio_path):
        if self.augment:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        else:
            y = load_audio(audio_path)

        if self.noiseInjector:
            logging.info("inject noise")
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)

        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)

        # Short-time Fourier transform (STFT)
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, phase = librosa.magphase(D)

        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)

        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError

class LogFBankDataset(Dataset):
    def __init__(self, vocab, args, audio_conf, manifest_filepath_list, normalize=False, augment=False, input_type="char", is_train=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:
        /path/to/audio.wav,/path/to/audio.txt
        ...
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        self.max_size = 0
        self.ids_list = []
        for i in range(len(manifest_filepath_list)):
            manifest_filepath = manifest_filepath_list[i]
            with open(manifest_filepath) as f:
                ids = f.readlines()

            ids = [x.strip().split(',') for x in ids]
            self.ids_list.append(ids)
            self.max_size = max(len(ids), self.max_size)

        self.max_size = self.max_size * len(manifest_filepath_list)
        print("max_size:", self.max_size)

        print("input_type:", input_type)
        self.input_type = input_type
        self.manifest_filepath_list = manifest_filepath_list
        self.normalize = normalize
        self.vocab = vocab

        if self.input_type == "bpe":
            self.bpeemb_list = []
            for i in range(len(self.lang_list)):
                lang = self.lang_list[i].replace("<","").replace(">","").lower()
                self.bpeemb_list.append(BPEmb(lang=lang, vs=1000))
        super(LogFBankDataset, self).__init__()

    def __getitem__(self, index):
        # lang_id = random.randint(0, len(self.ids_list)-1)
        lang_id = index % len(self.manifest_filepath_list)
        sample_id = index // len(self.manifest_filepath_list)
        ids = self.ids_list[lang_id]
        sample = ids[sample_id % len(ids)]
        # print(lang_id, sample_id)
        audio_path, transcript_path = sample[0], sample[1]
        feat = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript_path)
        return feat, transcript

    def parse_audio(self, path):
        (rate,sig) = wav.read(path)
        fbank_feat = logfbank(sig,rate,nfilt=80)
        fbank_feat = torch.FloatTensor(np.transpose(fbank_feat, (1, 0)))
        # print(fbank_feat.size())
        if self.normalize:
            mean = fbank_feat.mean()
            std = fbank_feat.std()
            fbank_feat.add_(-mean)
            fbank_feat.div_(std)
        return fbank_feat

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = " " + transcript_file.read().replace('\n', '').lower()

        # if self.input_type == "bpe":
        #     bpes = self.bpeemb_list[lang_id].encode(transcript)
        #     transcript = []
        #     for i in range(len(bpes)):
        #         transcript = transcript + [bpes[i]]

        transcript = list(filter(None, [self.vocab.label2id.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return self.max_size

class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, vocab, args, audio_conf, manifest_filepath_list, normalize=False, augment=False, input_type="char", is_train=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:
        /path/to/audio.wav,/path/to/audio.txt
        ...
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        self.max_size = 0
        self.ids_list = []
        self.is_train = is_train
        self.args = args
        self.vocab = vocab

        for i in range(len(manifest_filepath_list)):
            manifest_filepath = manifest_filepath_list[i]
            with open(manifest_filepath) as f:
                ids = f.readlines()

            ids = [x.strip().split(',') for x in ids]
            self.ids_list.append(ids)
            self.max_size = max(len(ids), self.max_size)

        self.max_size = self.max_size * len(manifest_filepath_list)
        if self.is_train:
            if len(manifest_filepath_list) == 1:
                self.max_size = self.max_size
            else:
                self.max_size = 30000
        else:
            self.max_size = self.max_size
        print("max_size:", self.max_size)

        print("input_type:", input_type)
        self.input_type = input_type
        self.manifest_filepath_list = manifest_filepath_list

        # if self.input_type == "bpe":
        #     self.bpeemb_list = []
        #     for i in range(len(self.lang_list)):
        #         lang = self.lang_list[i].replace("<","").replace(">","").lower()
        #         self.bpeemb_list.append(BPEmb(lang=lang, vs=1000))
        # elif self.input_type == "ipa":
        #     self.ipa_list = []
        #     for i in range(len(self.lang_list)):
        #         lang = ipa_map[self.lang_list[i].replace("<","").replace(">","").lower()]
        #         self.ipa_list.append(epitran.Epitran(lang))

        super(SpectrogramDataset, self).__init__(
            audio_conf, normalize, augment)

    def __getitem__(self, index):
        if self.is_train:
            manifest_id = index % len(self.manifest_filepath_list)
            sample_id = index // len(self.manifest_filepath_list)
            ids = self.ids_list[manifest_id]
            sample = ids[sample_id % len(ids)]

            audio_path, transcript_path = sample[0], sample[1]
            spect = self.parse_audio(audio_path)[:,:self.args.src_max_len]
            transcript = self.parse_transcript(transcript_path)
        else: # valid or test
            ids = self.ids_list[0]
            sample = ids[index % len(ids)]
 
            audio_path, transcript_path = sample[0], sample[1]
            spect = self.parse_audio(audio_path)[:,:self.args.src_max_len]
            transcript = self.parse_transcript(transcript_path)
        return spect, transcript

    def parse_transcript(self, transcript_path):
        if self.input_type == "char":
            with open(transcript_path, 'r', encoding='utf8') as transcript_file:
                cur_transcript = " " + transcript_file.read().replace('\n', '').lower()
        elif self.input_type == "ipa":
            cur_transcript = np.load(transcript_path)
        # elif self.input_type == "bpe":
        #     with open(transcript_path, 'r', encoding='utf8') as transcript_file:
        #         cur_transcript = " " + transcript_file.read().replace('\n', '').lower()
        #     bpes = self.bpeemb_list[lang_id].encode(cur_transcript)
        #     cur_transcript = []
        #     for i in range(len(bpes)):
        #         cur_transcript = cur_transcript + [bpes[i]]

        transcript = list(filter(None, [self.vocab.label2id.get(x) for x in list(cur_transcript)]))
        return transcript

    def __len__(self):
        return self.max_size

class CPT2LogFBankDataset(Dataset):
    def __init__(self, tokenizer, args, audio_conf, manifest_filepath_list, normalize=False, augment=False, input_type="char", is_train=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:
        /path/to/audio.wav,/path/to/audio.txt
        ...
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        self.max_size = 0
        self.ids_list = []
        for i in range(len(manifest_filepath_list)):
            manifest_filepath = manifest_filepath_list[i]
            with open(manifest_filepath) as f:
                ids = f.readlines()

            ids = [x.strip().split(',') for x in ids]
            self.ids_list.append(ids)
            self.max_size = max(len(ids), self.max_size)

        self.max_size = self.max_size * len(manifest_filepath_list)
        print("max_size:", self.max_size)

        print("input_type:", input_type)
        self.input_type = input_type
        self.manifest_filepath_list = manifest_filepath_list
        self.normalize = normalize
        self.tokenizer = tokenizer

        if self.input_type == "bpe":
            self.bpeemb_list = []
            for i in range(len(self.lang_list)):
                lang = self.lang_list[i].replace("<","").replace(">","").lower()
                self.bpeemb_list.append(BPEmb(lang=lang, vs=1000))
        super(LogFBankDataset, self).__init__()

    def __getitem__(self, index):
        # lang_id = random.randint(0, len(self.ids_list)-1)
        lang_id = index % len(self.manifest_filepath_list)
        sample_id = index // len(self.manifest_filepath_list)
        ids = self.ids_list[lang_id]
        sample = ids[sample_id % len(ids)]
        # print(lang_id, sample_id)
        audio_path, transcript_path = sample[0], sample[1]
        feat = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript_path)
        return feat, transcript

    def parse_audio(self, path):
        (rate,sig) = wav.read(path)
        fbank_feat = logfbank(sig,rate,nfilt=80)
        fbank_feat = torch.FloatTensor(np.transpose(fbank_feat, (1, 0)))
        # print(fbank_feat.size())
        if self.normalize:
            mean = fbank_feat.mean()
            std = fbank_feat.std()
            fbank_feat.add_(-mean)
            fbank_feat.div_(std)
        return fbank_feat

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            cur_transcript = " " + transcript_file.read().replace('\n', '').lower()
        bpes = self.tokenizer.encode(cur_transcript)
        return transcript

    def __len__(self):
        return self.max_size

class CPT2SpectogramDataset(Dataset, SpectrogramParser):
    def __init__(self, tokenizer, args, audio_conf, manifest_filepath_list, normalize=False, augment=False, is_train=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:
        /path/to/audio.wav,/path/to/audio.txt
        ...
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        self.max_size = 0
        self.ids_list = []
        self.is_train = is_train
        self.args = args
        self.tokenizer = tokenizer

        for i in range(len(manifest_filepath_list)):
            manifest_filepath = manifest_filepath_list[i]
            with open(manifest_filepath) as f:
                ids = f.readlines()

            ids = [x.strip().split(',') for x in ids]
            self.ids_list.append(ids)
            self.max_size = max(len(ids), self.max_size)

        self.max_size = self.max_size * len(manifest_filepath_list)
        if self.is_train:
            if len(manifest_filepath_list) == 1:
                self.max_size = self.max_size
            else:
                self.max_size = 30000
        else:
            self.max_size = self.max_size
        print("max_size:", self.max_size)

        print("input_type:", input_type)
        self.input_type = input_type
        self.manifest_filepath_list = manifest_filepath_list

        # if self.input_type == "bpe":
        #     self.bpeemb_list = []
        #     for i in range(len(self.lang_list)):
        #         lang = self.lang_list[i].replace("<","").replace(">","").lower()
        #         self.bpeemb_list.append(BPEmb(lang=lang, vs=1000))
        # elif self.input_type == "ipa":
        #     self.ipa_list = []
        #     for i in range(len(self.lang_list)):
        #         lang = ipa_map[self.lang_list[i].replace("<","").replace(">","").lower()]
        #         self.ipa_list.append(epitran.Epitran(lang))

        super(CPT2SpectogramDataset, self).__init__(
            audio_conf, normalize, augment)

    def __getitem__(self, index):
        if self.is_train:
            manifest_id = index % len(self.manifest_filepath_list)
            sample_id = index // len(self.manifest_filepath_list)
            ids = self.ids_list[manifest_id]
            sample = ids[sample_id % len(ids)]

            audio_path, transcript_path = sample[0], sample[1]
            spect = self.parse_audio(audio_path)[:,:self.args.src_max_len]
            transcript = self.parse_transcript(transcript_path)
        else: # valid or test
            ids = self.ids_list[0]
            sample = ids[index % len(ids)]
 
            audio_path, transcript_path = sample[0], sample[1]
            spect = self.parse_audio(audio_path)[:,:self.args.src_max_len]
            transcript = self.parse_transcript(transcript_path)
        return spect, transcript

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            cur_transcript = " " + transcript_file.read().replace('\n', '').lower()
        bpes = self.tokenizer.encode(cur_transcript)
        
        print('cur_transcript', cur_transcript)
        print('bpes', bpes)
        print('dec_transcript', self.tokenizer.decode(bpes))

        return bpes

    def __len__(self):
        return self.max_size

class NoiseInjection(object):
    def __init__(self,
                 path=None,
                 sample_rate=16000,
                 noise_levels=(0, 0.5)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        if not os.path.exists(path):
            print("Directory doesn't exist: {}".format(path))
            raise IOError
        self.paths = path is not None and librosa.util.find_files(path)
        self.sample_rate = sample_rate
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    def inject_noise_sample(self, data, noise_path, noise_level):
        noise_len = get_audio_length(noise_path)
        data_len = len(data) / self.sample_rate
        noise_start = np.random.rand() * (noise_len - data_len)
        noise_end = noise_start + data_len
        noise_dst = audio_with_sox(
            noise_path, self.sample_rate, noise_start, noise_end)
        assert len(data) == len(noise_dst)
        noise_energy = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
        data_energy = np.sqrt(data.dot(data) / data.size)
        data += noise_level * noise_dst * data_energy / noise_energy
        return data

class AudioDataLoader(DataLoader):
    def __init__(self, vocab, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.pad_token_id = vocab.PAD_ID
    
        def _collate_fn(batch):
            def func(p):
                return p[0].size(1)

            def func_trg(p):
                return len(p[1])

            def func_trg_transcript(p):
                return len(p[2])

            # descending sorted
            batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)

            max_seq_len = max(batch, key=func)[0].size(1)
            freq_size = max(batch, key=func)[0].size(0)
            max_trg_len = len(max(batch, key=func_trg)[1])

            inputs = torch.zeros(len(batch), 1, freq_size, max_seq_len)
            input_sizes = torch.IntTensor(len(batch))
            input_percentages = torch.FloatTensor(len(batch))

            targets = torch.full((len(batch), max_trg_len), self.pad_token_id).long()
            target_sizes = torch.IntTensor(len(batch))

            for x in range(len(batch)):
                sample = batch[x]
                input_data = sample[0]
                target = sample[1]
                seq_length = input_data.size(1)
                input_sizes[x] = seq_length
                inputs[x][0].narrow(1, 0, seq_length).copy_(input_data)
                input_percentages[x] = seq_length / float(max_seq_len)
                target_sizes[x] = len(target)
                targets[x][:len(target)] = torch.IntTensor(target)

            # print(">", targets[0], langs, lang_names)
            # print(target_transcripts)
            return inputs, targets, input_percentages, input_sizes, target_sizes

        self.collate_fn = _collate_fn

class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size]
                     for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)