from faster_whisper import WhisperModel
import string

class WhisperTranscriptor:
    def __init__(self, model_size="large-v3"):
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(audio, beam_size=5, vad_filter=True, word_timestamps=True)
        transcribed_text = ""
        words_timestamps = []
        for segment in segments:
            transcribed_text += segment.text
            for word in segment.words:
                clean_word = word.word.rstrip(string.punctuation)
                words_timestamps.append([str(word.start), str(word.end), clean_word])

        return transcribed_text.strip(), words_timestamps

if __name__ == "__main__":
    tran = WhisperTranscriptor()
    print(tran.transcribe("../downloads/test.wav"))
    # import torch
    #
    # print(torch.cuda.is_available())
    # print(torch.version.cuda)
    # print(torch.backends.cudnn.version())
