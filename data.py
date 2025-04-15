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

import torch
import random
import aiohttp
import asyncio
import requests
import urllib.parse
from PIL import Image
from io import BytesIO
from async_lru import alru_cache
import torchvision.transforms as transforms

MAX_ROWS_PER_REQUEST = 10

@alru_cache(maxsize=10000)
async def get_split_row_count(dataset, config="default", split="train"):
    """
    Uses the HuggingFace datasets-server API to retrieve the number of examples in a given dataset/config/split.
    """
    encoded_dataset = urllib.parse.quote(dataset)
    info_url = f"https://datasets-server.huggingface.co/info?dataset={encoded_dataset}"
    async with aiohttp.ClientSession() as session:
        async with session.get(info_url) as response:
            response.raise_for_status()
            data = await response.json()
    try:
        return data["dataset_info"][config]["splits"][split]["num_examples"]
    except KeyError:
        raise ValueError(f"Split '{split}' not found in dataset '{dataset}' with config '{config}'.")

async def fetch_batch(session, dataset, config, split, offset, length):
    url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }
    async with session.get(url, params=params) as response:
        response.raise_for_status()
        data = await response.json()
        return data["rows"]

async def get_samples(
    dataset="PrimeIntellect/verifiable-coding-problems", 
    config="default", 
    split="train", 
    sample_size=1,
    seed="42"
):
    """
    Fast parallel sampling from HuggingFace datasets-server.
    """
    total_rows = await get_split_row_count(dataset, config, split)
    sample_size = min(sample_size, total_rows)
    random.seed(seed)

    # Create randomized batch offsets
    num_batches = (sample_size + MAX_ROWS_PER_REQUEST - 1) // MAX_ROWS_PER_REQUEST
    offsets = sorted(random.sample(range(0, total_rows - MAX_ROWS_PER_REQUEST), num_batches))
    batch_lengths = [min(MAX_ROWS_PER_REQUEST, sample_size - i * MAX_ROWS_PER_REQUEST) for i in range(num_batches)]

    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_batch(session, dataset, config, split, offset, length)
            for offset, length in zip(offsets, batch_lengths)
        ]
        results = await asyncio.gather(*tasks)

    # Flatten and truncate to exact sample_size
    all_samples = [item for batch in results for item in batch]
    return all_samples[:sample_size]


async def get_batch(
        batch_size: int = 2,
        seed:str = "42"
    ):
    dd = await get_samples("ylecun/mnist", config='mnist', sample_size=batch_size, seed = seed )
    # Define transform to convert images to tensors and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    # Process each image
    tensors = []
    labels = []
    for sample in dd:
        # Get image from URL
        response = requests.get(sample['row']['image']['src'])
        img = Image.open(BytesIO(response.content))
        
        # Convert to tensor and normalize
        img_tensor = transform(img)
        tensors.append(img_tensor)
        labels.append(sample['row']['label'])

    # Stack into batches
    images = torch.stack(tensors)
    labels = torch.tensor(labels)
    return images, labels