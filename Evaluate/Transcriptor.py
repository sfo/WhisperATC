import io
import json
from abc import ABC, abstractmethod

import assemblyai as aai
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import whisper
from datasets import Audio, load_dataset
from deepgram import DeepgramClient, PrerecordedOptions
from numpy.typing import NDArray
from overrides import override
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm


class Sample:
    nato = "alpha bravo charlie delta echo foxtrot golf hotel india juliett kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu"
    terminology = "climb climbing descend descending passing feet knots degrees direct maintain identified ILS VFR IFR contact frequency turn right left heading altitude flight level cleared squawk approach runway established report affirm negative wilco roger radio radar"

    def __init__(self, record: dict) -> None:
        super().__init__()
        self._prompt = " ".join(
            [
                "Air Traffic Control Communications",
                record["info"].replace("\n", " ") if "info" in record else "",
                self.nato,
                self.terminology,
            ]
        )

        # TODO - maybe move this to a model's preprocessing steps
        self._audio = record["audio"]
        self._audio_array = np.float32(whisper.pad_or_trim(self._audio["array"]))

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def audio(self) -> NDArray:
        return self._audio_array

    @property
    def bytes(self) -> io.BytesIO:
        audio = self._audio["array"]
        sampling_rate = self._audio["sampling_rate"]
        bytes_wav = bytes()
        byte_io = io.BytesIO(bytes_wav)
        wavfile.write(byte_io, sampling_rate, audio)
        return byte_io


class Transcript(ABC):
    def __init__(self, raw) -> None:
        super().__init__()
        self._raw = raw

    @property
    def raw(self) -> dict:
        return self._raw

    @property
    @abstractmethod
    def transcript(self) -> str:
        pass

    @property
    @abstractmethod
    def timestamps(self) -> list[dict]:
        pass


class WhisperTranscript(Transcript):
    @property
    @override
    def transcript(self) -> str:
        return self._raw["text"]

    @property
    @override
    def timestamps(self) -> list[dict]:
        segments = self._raw["segments"]
        assert len(segments) == 1
        return segments["words"]


class DeepgramTranscript(Transcript):
    @property
    @override
    def transcript(self) -> str:
        channels = json.loads(self._raw)["results"]["channels"]
        assert len(channels) == 1
        alternatives = channels[0]["alternatives"]
        assert len(alternatives) == 1
        return alternatives[0]["transcript"]

    @property
    @override
    def timestamps(self) -> list[dict]:
        channels = json.loads(self._raw)["results"]["channels"]
        assert len(channels) == 1
        alternatives = channels[0]["alternatives"]
        assert len(alternatives) == 1
        return alternatives[0]["words"]


class AssemblyAITranscript(Transcript):
    @property
    @override
    def transcript(self) -> str:
        return self._raw["text"]

    @property
    @override
    def timestamps(self) -> list[dict]:
        return self._raw["words"]


class Model(ABC):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def transcribe(self, sample: Sample, with_prompt: bool) -> Transcript:
        pass

    @property
    @abstractmethod
    def supports_prompt(self) -> bool:
        pass


class AssemblyAIModel(Model):
    def __init__(self, api_key: str) -> None:
        super().__init__("assemblyai")
        aai.settings.api_key = api_key
        self._transcriber = aai.Transcriber()

    @override
    def transcribe(self, sample: Sample, with_prompt: bool) -> Transcript:
        config = aai.TranscriptionConfig(
            language_code="en",
            speech_model=aai.SpeechModel.best,
        )
        if with_prompt:
            config.set_word_boost(
                words=sample.prompt.split(" "),
                boost=aai.WordBoost.high,
            )
        response = self._transcriber.transcribe(sample.bytes, config=config)
        return AssemblyAITranscript(response.json_response)

    @property
    @override
    def supports_prompt(self) -> bool:
        return True


class DeepgramModel(Model):
    def __init__(self, variant: str, api_key: str) -> None:
        super().__init__(f"deepgram-{variant}")
        self._variant = variant
        self._client = DeepgramClient(api_key)

    @override
    def transcribe(self, sample: Sample, with_prompt: bool) -> Transcript:
        return self._request_transcript(
            sample.bytes.read(), self._model_options(sample, with_prompt)
        )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def _request_transcript(
        self, audio: bytes, options: PrerecordedOptions
    ) -> Transcript:
        transcript = self._client.listen.rest.v("1").transcribe_file(  # type: ignore
            source={
                "buffer": audio,
            },
            options=options,
        )
        return DeepgramTranscript(transcript.to_json())  # type: ignore

    @abstractmethod
    def _model_options(self, sample: Sample, with_prompt: bool) -> PrerecordedOptions:
        pass

    @property
    @override
    def supports_prompt(self) -> bool:
        return True


class DeepgramNova2ATC(DeepgramModel):
    def __init__(self, api_key: str) -> None:
        super().__init__("nova-2-atc", api_key)

    @override
    def _model_options(self, sample: Sample, with_prompt: bool) -> PrerecordedOptions:
        if with_prompt:
            return PrerecordedOptions(
                model=self._variant,
                smart_format=True,
                keywords=sample.prompt,  # FIXME check this (e.g. delimiter, etc.)
            )

        return PrerecordedOptions(
            model=self._variant,
            smart_format=True,
        )


class DeepgramNova3(DeepgramModel):
    def __init__(self, api_key: str) -> None:
        super().__init__("nova-3", api_key)

    @override
    def _model_options(self, sample: Sample, with_prompt: bool) -> PrerecordedOptions:
        if with_prompt:
            return PrerecordedOptions(
                model=self._variant,
                smart_format=True,
                keyterm=sample.prompt.split(
                    " "
                ),  # FIXME check this (e.g. delimiter, etc.)
            )

        return PrerecordedOptions(
            model=self._variant,
            smart_format=True,
        )


class WhisperModel(Model):
    def __init__(self, variant: str) -> None:
        super().__init__(f"whisper-{variant}")
        self._variant = variant
        self._base_model = self._load_model()

    @abstractmethod
    def _load_model(self) -> whisper.Whisper:
        pass

    @override
    def transcribe(self, sample: Sample, with_prompt: bool) -> Transcript:
        options = dict(
            language="en",
            fp16=False,
            word_timestamps=True,
        )
        if with_prompt:
            options["prompt"] = sample.prompt
        result = whisper.transcribe(self._base_model, sample.audio, **options)
        return WhisperTranscript(result)

    @property
    @override
    def supports_prompt(self) -> bool:
        return True


class VanillaWhisperModel(WhisperModel):
    @override
    def _load_model(self) -> whisper.Whisper:
        return whisper.load_model(self._variant)


class WhisperATCModel(WhisperModel):
    from model_loader import load_model

    @override
    def _load_model(self) -> whisper.Whisper:
        whisper_base_model = "-".join(self._variant.split("-")[:2])
        return load_model(whisper_base_model, f"jlvdoorn/{self._name}")


class Transcriptor:
    def __init__(self, model: Model, dts: str, spl: str) -> None:
        super().__init__()
        self._model = model
        self._dataset = load_dataset(dts).cast_column(
            "audio", Audio(sampling_rate=16000)
        )
        self._spl = spl
        self._output_file = (
            dts.split("/")[-1] + "-" + spl + "-" + model.name + ".pickle"
        )

    def _transcribe(self, s: str, i: int):
        sample = Sample(self._dataset[s][i])  # type: ignore
        transcript_clean = self._model.transcribe(sample, False)
        transcript_prmpt = (
            self._model.transcribe(sample, True)
            if self._model.supports_prompt
            else None
        )
        return transcript_clean, transcript_prmpt

    def transcribe(self) -> None:
        df = pd.DataFrame()

        print("Starting inference...")

        for s in tqdm(self._spl.split("+")):
            for i in tqdm(range(len(self._dataset[s]))):  # type: ignore
                transcript_clean, transcript_prmpt = self._transcribe(s, i)

                series = pd.Series(
                    {
                        "split": s,
                        "hyp-prmpt": (
                            transcript_prmpt.transcript if transcript_prmpt else None
                        ),
                        "hyp-clean": transcript_clean.transcript,
                        "ref": self._dataset[s][i]["text"],  # type: ignore
                        "words-prmpt": (
                            transcript_prmpt.timestamps if transcript_prmpt else None
                        ),
                        "words-clean": transcript_clean.timestamps,
                    }
                )
                df = pd.concat((df, series.to_frame().T), ignore_index=True)
                df.to_pickle(self._output_file)
