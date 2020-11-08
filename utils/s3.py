import logging
import boto3
import pickle
import botocore
import fastparquet
import pickle
import s3fs
import json
import pandas as pd

logger = logging.getLogger(__name__)


class S3Agent:
    def __init__(self, config):
        self.bucket = config["bucket"]
        self.config = config
        if "s3_endpoint" in config:
            self.endpoint_url = config["s3_endpoint"]
            self.filesystem = s3fs.S3FileSystem(
                anon=config["s3fs_anon"],
                client_kwargs={"endpoint_url": self.endpoint_url},
            )
        else:
            self.filesystem = s3fs.S3FileSystem(anon=config["s3fs_anon"])

    def get_subfolders(self, path):
        folder_list = self.filesystem.ls(self.bucket)
        return folder_list

    def load_parquet(self, path=""):
        s3_path = f"{self.bucket}/{path}"
        return fastparquet.ParquetFile(
            s3_path, open_with=self.filesystem.open
        ).to_pandas()

    def upload_parquet(self, df, path=""):
        s3_path = f"{self.bucket}/{path}"
        fastparquet.write(
            s3_path,
            df,
            compression="gzip",
            open_with=self.filesystem.open,
            write_index=False,
        )

    def load_object(self, directory, filename, file_extension, pickled=False, use_json=False):
        s3_object = self._object_reference(directory, filename, file_extension)
        if pickled:
            loaded_object = pickle.loads(s3_object.get()["Body"].read())
        elif use_json:
            loaded_object = json.loads(s3_object.get()["Body"].read())

        else:
            loaded_object = s3_object.get()["Body"].read()
        return loaded_object

    def store_object(self, obj, directory, filename, file_extension, pickled=False):
        latest_ref = self._object_reference(directory, filename, file_extension)
        if pickled:
            obj = pickle.dumps(obj)
        response = latest_ref.put(Body=obj)
        return response

    def is_object_available(self, directory, filename, file_extension):
        s3_reference = self._object_reference(directory, filename, file_extension)
        try:
            s3_reference.load()
        except botocore.exceptions.ClientError:
            return False
        return True

    def _s3_resource(self):
        if "s3_endpoint" in self.config:
            s3_resource = boto3.resource("s3", endpoint_url=self.config["s3_endpoint"])
        else:
            s3_resource = boto3.resource("s3")
        return s3_resource

    def _object_reference(self, directory, filename, file_extension):
        if "s3_endpoint" in self.config:
            s3 = boto3.client("s3", endpoint_url=self.config["s3_endpoint"])
        s3_resource = self._s3_resource()
        path = file_path(directory, filename, file_extension)
        s3_object = s3_resource.Object(self.bucket, path)
        return s3_object


def file_path(directory, filename, file_extension):
    if directory is not None:
        return f"{directory}/{filename}.{file_extension}"
    else:
        return f"{filename}.{file_extension}"