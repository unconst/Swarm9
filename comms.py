# The MIT License (MIT)
# © 2025 Swarm9
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import json
import asyncio
import botocore
from typing import List, Dict
from dotenv import load_dotenv
from aiobotocore.session import get_session

load_dotenv(override=True)

R2_BUCKET_ID = None
def bucket():
    global R2_BUCKET_ID
    if R2_BUCKET_ID is not None:
        return R2_BUCKET_ID
    try:
        R2_BUCKET_ID = os.getenv("R2_BUCKET_ID")
        if not R2_BUCKET_ID:
            return None
        return R2_BUCKET_ID
    except Exception as e:
        return None

R2_ACCOUNT_ID = None
def load_r2_account_id():
    global R2_ACCOUNT_ID
    if R2_ACCOUNT_ID is not None:
        return R2_ACCOUNT_ID
    try:
        R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
        if not R2_ACCOUNT_ID:
            return None
        return R2_ACCOUNT_ID
    except Exception as e:
        return None

R2_ENDPOINT_URL = None
def load_r2_endpoint_url():
    global R2_ENDPOINT_URL
    account_id = load_r2_account_id()
    if not account_id:
        return None
    try:
        R2_ENDPOINT_URL = f"https://{account_id}.r2.cloudflarestorage.com"
        return R2_ENDPOINT_URL
    except Exception as e:
        return None

R2_WRITE_ACCESS_KEY_ID = None
def load_r2_write_access_key_id():
    global R2_WRITE_ACCESS_KEY_ID
    if R2_WRITE_ACCESS_KEY_ID is not None:
        return R2_WRITE_ACCESS_KEY_ID
    try:
        R2_WRITE_ACCESS_KEY_ID = os.getenv("R2_WRITE_ACCESS_KEY_ID")
        if not R2_WRITE_ACCESS_KEY_ID:
            return None
        return R2_WRITE_ACCESS_KEY_ID
    except Exception as e:
        return None

R2_WRITE_SECRET_ACCESS_KEY = None
def load_r2_write_secret_access_key():
    global R2_WRITE_SECRET_ACCESS_KEY
    if R2_WRITE_SECRET_ACCESS_KEY is not None:
        return R2_WRITE_SECRET_ACCESS_KEY
    try:
        R2_WRITE_SECRET_ACCESS_KEY = os.getenv("R2_WRITE_SECRET_ACCESS_KEY")
        if not R2_WRITE_SECRET_ACCESS_KEY:
            return None
        return R2_WRITE_SECRET_ACCESS_KEY
    except Exception as e:
        return None

# R2 configuration details.
CLIENT_CONFIG = botocore.config.Config(max_pool_connections=256)
session = get_session()

# Helper: compute local file path.
def get_local_path(bucket: str, filename: str ) -> str:
    return os.path.join(os.path.expanduser("~/storage"), bucket, filename)

async def exists_locally( bucket:str, filename: str ) -> bool:
    local_path = get_local_path( bucket, filename )
    return os.path.exists(local_path)

# Deletes the local file from ~/storage/<bucket>/<hotkey>-<window>-<name>.json
async def delete_locally( bucket:str, filename: str ) -> None:
    local_path = get_local_path( bucket, filename )
    if os.path.exists(local_path):
        try:
            await asyncio.to_thread(os.remove, local_path)
        except Exception as e:
            pass
    else:
        pass
    
# Loads the file as JSON from ~/storage/<bucket>/<window>-<name>.json
async def load( bucket:str, filename: str) -> dict:
    local_path = get_local_path(bucket, filename)
    try:
        def load_json(path):
            with open(path, "r") as f:
                return json.load(f)
        data = await asyncio.to_thread(load_json, local_path)
        return data
    except Exception as e:
        return None

async def download(bucket:str, filename: str) -> Dict:
    local_path = get_local_path(bucket, filename)
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
    except Exception as e:
        return None
    try:
        endpoint_url = load_r2_endpoint_url()
        access_key = load_r2_write_access_key_id()
        secret_key = load_r2_write_secret_access_key()
        if not endpoint_url or not access_key or not secret_key:
            return None

        async with session.create_client(
            "s3",
            endpoint_url=endpoint_url,
            region_name="enam",  # Region can be arbitrary for R2.
            config=CLIENT_CONFIG,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        ) as s3_client:
            response = await s3_client.get_object(Bucket=bucket, Key=filename)
            last_modified = response.get("LastModified")
            async with response['Body'] as stream:
                data_bytes = await stream.read()
            try:
                data = json.loads(data_bytes.decode("utf-8"))
            except json.JSONDecodeError as e:
                return None
            try:
                with open(local_path, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                return None
        return data
    except Exception as e:
        return None

# Uploads the data dictionary as a JSON file to the bucket as filename <window>-<name>.json
async def upload( bucket:str, filename: str, data: dict ) -> None:
    try:
        json_bytes = json.dumps(data, indent=2).encode("utf-8")
    except Exception as e:
        return

    try:
        endpoint_url = load_r2_endpoint_url()
        access_key = load_r2_write_access_key_id()
        secret_key = load_r2_write_secret_access_key()
        if not endpoint_url or not access_key or not secret_key:
            return

        async with session.create_client(
            "s3",
            endpoint_url=endpoint_url,
            region_name="enam",
            config=CLIENT_CONFIG,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        ) as s3_client:
            await s3_client.put_object(
                Bucket=bucket,
                Key=filename,
                Body=json_bytes
            )
    except Exception as e:
        return

    try:
        local_path = get_local_path(bucket, filename)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        return


# Checks if the file exists on the bucket (remote storage)
async def exists( bucket:str, filename: str ) -> bool:
    try:
        endpoint_url = load_r2_endpoint_url()
        access_key = load_r2_write_access_key_id()
        secret_key = load_r2_write_secret_access_key()
        if not endpoint_url or not access_key or not secret_key:
            return False
        async with session.create_client(
            "s3",
            endpoint_url=endpoint_url,
            region_name="enam",
            config=CLIENT_CONFIG,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        ) as s3_client:
            await s3_client.head_object(Bucket=bucket, Key=filename)
            return True
    except botocore.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in ["404", "NoSuchKey", "NotFound"]:
            return False
        else:
            return False
    except Exception as e:
        return False
    
    
async def timestamp( bucket:str, filename: str ):
    try:
        endpoint_url = load_r2_endpoint_url()
        access_key = load_r2_write_access_key_id()
        secret_key = load_r2_write_secret_access_key()
        if not endpoint_url or not access_key or not secret_key:
            return None

        async with session.create_client(
            "s3",
            endpoint_url=endpoint_url,
            region_name="enam",
            config=CLIENT_CONFIG,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        ) as s3_client:
            response = await s3_client.head_object(Bucket=bucket, Key=filename)
            last_modified = response.get("LastModified")
            if last_modified is not None:
                return last_modified
            else:
                return None
    except Exception as e:
        return None

async def list( bucket:str, prefix: str ) -> List[str]:
    endpoint_url = load_r2_endpoint_url()
    access_key = load_r2_write_access_key_id()
    secret_key = load_r2_write_secret_access_key()

    if not endpoint_url or not access_key or not secret_key:
        return []

    matching_keys = []    
    try:
        async with session.create_client(
            "s3",
            endpoint_url=endpoint_url,
            region_name="enam",
            config=CLIENT_CONFIG,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        ) as s3_client:
            paginator = s3_client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj.get("Key", "")
                        matching_keys.append(key)
    except Exception as e:
        pass
    return matching_keys


