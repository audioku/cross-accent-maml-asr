class Vocab(object):
    def __init__(self):
        self.PAD_TOKEN = "<PAD>"
        self.SOS_TOKEN = "<SOS>"
        self.EOS_TOKEN = "<EOS>"
        self.OOV_TOKEN = "<OOV>"

        self.special_token_list = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.OOV_TOKEN]
        self.token2id, self.id2token = {}, []
        self.label2id, self.id2label = {}, []

        for token in self.special_token_list:
            self.add_token(token)
            self.add_label(token)

    def add_token(self, token):
        if token not in self.id2token:
            idx = len(self.token2id)
            self.id2token.append(token)
            self.token2id[token] = idx

    def add_label(self, label):
        if label not in self.id2label:
            idx = len(self.label2id)
            self.id2label.append(label)
            self.label2id[label] = idx