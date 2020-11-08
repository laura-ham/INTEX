from utils import s3

class Dataset:
    def __init__(self, config):
        self.config = config
        self.s3agent = s3.S3Agent(config)

