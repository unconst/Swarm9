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

import sys
import data
import time
import copy
import comms
import torch
import random
import asyncio
import datetime
import argparse
import traceback
import torch.nn as nn
import bittensor as bt
import torch.nn.functional as F
from rich.console import Console
from async_lru import alru_cache
from typing import List, Dict, Tuple, Optional, Union

# Subnet identifier.
NETUID = 10
# Number of blocks per window.
WINDOW_SIZE = 2
# Enums
FORWARD = 'activations'
BACKWARD = 'gradients'
STATE = 'state'

# Build a simple MNIST sequential model.
model_layers = [
    nn.Flatten(),
    nn.Linear(28 * 28, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
]

# Create a sequential model from the layers
model = nn.Sequential(*model_layers)

def get_layers(rank: int, worldsize: int) -> List[nn.Module]:
    """
    Split the full model's layers into partitions for pipeline parallelism.
    Each pipeline stage (indexed by rank) is assigned a contiguous block of layers.
    """
    total_layers = len(model_layers)
    # Determine how many layers go into each partition.
    layers_per_partition = total_layers // worldsize
    remainder = total_layers % worldsize  # extra layers to distribute
    
    # Calculate the start and end indices for this rank.
    start_idx = rank * layers_per_partition + min(rank, remainder)
    end_idx = start_idx + layers_per_partition + (1 if rank < remainder else 0)
    
    return model_layers[start_idx:end_idx]

def get_rank(layer_idx: int, worldsize: int) -> int:
    """
    Given a layer index and world size, determine which rank owns this layer.
    This is the inverse function of get_layers().
    """
    total_layers = len(model_layers)
    layers_per_partition = total_layers // worldsize
    remainder = total_layers % worldsize

    # For each rank, calculate its start_idx until we find the rank
    # whose range contains layer_idx
    for rank in range(worldsize):
        start_idx = rank * layers_per_partition + min(rank, remainder)
        end_idx = start_idx + layers_per_partition + (1 if rank < remainder else 0)
        if start_idx <= layer_idx < end_idx:
            return rank
            
    raise ValueError(f"Layer index {layer_idx} is out of range")

class Pipe:
    @staticmethod
    def get_config():
        """
        Configures command line arguments for the miner script.
        """
        parser = argparse.ArgumentParser(description='Miner script')
        parser.add_argument('--netuid', type=int, default=NETUID)
        parser.add_argument('--bucket', type=str, default=comms.bucket())
        parser.add_argument('--bs', type=int, default=2)
        parser.add_argument('--lr', type=float, default=0.001)
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        return bt.config(parser)

    def __init__(self, config):
        """
        Creates runner objects.
        """
        self.console = Console()
        self.config = Pipe.get_config() if config is None else config
        self.console.print(f"[bold yellow]\nPipe: {self.config}[/bold yellow]")
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.async_subtensor(config=self.config)
        self.criterion = nn.CrossEntropyLoss()
        self.world = [self.wallet.hotkey.ss58_address]
        self.layers = get_layers(0, worldsize=len(self.world))
        self.optimizers = {}
        self.console.print("\n[bold orange]Layers:[/bold orange]")
        for idx, layer in enumerate(self.layers):
            optim = None
            if any(p.requires_grad for p in layer.parameters()):
                optim = torch.optim.Adam(layer.parameters(), lr=self.config.lr)
            self.optimizers[idx] = optim
            self.console.print(f"[bold orange]\tlayer={layer}, optimizer=({True if optim != None else False}) [/bold orange]")

    async def initialize(self):
        """
        Initializes runner objects, gets your chain uid, and commits your R2 bucket information to the chain.
        """
        try:
            self.console.print(f"[bold green]\nInitialization...[bold green]")
            await self.subtensor.initialize()
            self.console.print(f"[bold green]\t>> Subtensor:[/bold green] [green]{self.subtensor}[/green]")
            self.metagraph = await self.subtensor.metagraph(self.config.netuid)
            self.console.print(f"[bold green]\t>> Metagraph:[/bold green] [green]{self.metagraph}[/green]")
            if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
                self.console.print(f"[bold red]\tERROR: Hotkey {self.wallet.hotkey.ss58_address} is not registered on subnet {self.config.netuid}![/bold red]")
                sys.exit()
            self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            self.console.print(f"[bold green]\t>> UID:[/bold green] [green]{self.uid}[/green]")
            self.bucket = await self.subtensor.get_commitment(netuid=self.config.netuid, uid=self.uid)
            if self.bucket != self.config.bucket:
                self.console.print(f"[misty_rose3]\tBucket mismatch detected. Updating commitment to: {self.config.bucket}[/misty_rose3]")
                await self.subtensor.set_commitment(wallet=self.wallet, netuid=self.config.netuid, data=self.config.bucket)
            self.console.print(f"[bold green]\t>> Bucket:[/bold green] [green]{self.bucket}[/green]")
            self.weights = [0 for _ in self.metagraph.hotkeys]
            self.console.print(f"[bold gold]\nWeights:[/bold gold] \n\t[green]{self.weights}[/green]")
        except Exception as e:
            self.console.print(f"[bold red]\tInitialization failed: {str(e)}\n{traceback.format_exc()}[/bold red]")
            sys.exit()
            
    @alru_cache(maxsize=10000)
    async def timestamp( self, window:int ) -> datetime.datetime:
        """
        Gets the block time at this window.
        """
        try:
            time =  await self.subtensor.get_timestamp( block = window * WINDOW_SIZE )
            return time
        except:
            return datetime.datetime.now()

    @alru_cache(maxsize=128)
    async def get_buckets(self, window: int) -> Dict[ str, str ]:
        """
        Retrieves bucket information for all peers at a specific window.
        Args - window (int): The window number to get bucket information for.
        Returns - buckets (Dict[str, str]): Maps hotkey to R2 bucket.
        """
        commitments = await self.subtensor.get_all_commitments(netuid=self.config.netuid, block=window * WINDOW_SIZE)
        buckets = {k: v for k, v in commitments.items() if isinstance(v, str) and len(v) == 32 and v.isalnum()}
        return buckets

    async def window(self) -> int:
        """
        Gets the current block window.
        Returns - window (int): block window
        """
        current_block = await self.subtensor.get_current_block()
        window_value = int(current_block / WINDOW_SIZE)
        return window_value
    
    async def apply_gradient(self, layer_idx: int ):
        # Apply gradients using optimizer
        if self.optimizers[layer_idx] is not None:
            self.optimizers[layer_idx].step()
            self.optimizers[layer_idx].zero_grad()
            self.console.print(f'\t\t>> [green]Ran[/green] gradient for layer = {layer_idx}', style='tan')
        else:
            self.console.print(f'\t\t>> [red]Empty[/red] gradient to apply for layer = {layer_idx}', style='tan')
    
    async def forward( self, layer_idx: int, inputs: torch.Tensor, require_grads: bool ) -> torch.Tensor:
        with torch.no_grad() if not require_grads else torch.enable_grad():
            # Enable grad tracking only for the last layer (for loss calculation)
            inputs.requires_grad_(require_grads)
            # Forward pass through the layer
            outputs: torch.Tensor = self.layers[layer_idx](inputs)
        self.console.print(f'\t\t>> [green]Ran[/green] forward for layer={layer_idx}, grads={require_grads}', style='tan')
        return outputs
        
    async def get_batch( self, window:int ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get batch of images and labels for MNIST
        seed = await self.subtensor.get_block_hash(window * WINDOW_SIZE)
        batch = await data.get_batch(batch_size=self.config.bs, seed=seed)
        self.console.print(f'\t\t>> [green]Pulled[/green] batch for window={window}', style='tan')
        return batch
    
    async def download( self, layer_idx:int, window:int, key: str) -> torch.Tensor:
        # Get the rank of the miner with this layer.
        input_rank = get_rank(layer_idx=layer_idx, worldsize=len(self.world))
        # Download activations from the previous layer from the previous window.
        fname = f'{self.world[input_rank]}-{key}-{layer_idx}-{window}.json'
        data_inputs = await comms.download(
            bucket = self.buckets[self.world[input_rank]],
            filename=fname
        )
        # Convert to tensor
        if data_inputs is None:
            self.console.print(f'\t\t>> [red]Empty[/red] {key} from window={window}, layer={layer_idx}, fname={fname}', style='tan')
            return None
        else:
            if key == STATE:
                layer = copy.deepcopy(self.layers[layer_idx])
                for name, param in layer.named_parameters():
                    param.data = torch.tensor(data_inputs[name])
                return layer
            else:
                tensor = torch.tensor([data_inputs[k] for k in sorted(data_inputs.keys())], dtype=torch.float32)
            self.console.print(f'\t\t>> [green]Downloaded[/green] {key} from window={window}, layer={layer_idx}, fname={fname}', style='tan')
            return tensor
    
    async def upload(self, layer_idx:int, window:int, key: str, tensor: Union[torch.Tensor, nn.Module] ):
        # Upload the gradients.
        fname: str = f'{self.wallet.hotkey.ss58_address}-{key}-{layer_idx}-{window}.json'
        if key == STATE:
            uploads = { name: param.data.detach().tolist() for name, param in tensor.named_parameters() }
        else:
            uploads = {str(i): tensor[i].detach().tolist() for i in range(len(tensor))}
        await comms.upload(
            bucket = self.bucket,
            filename=fname,
            data=uploads
        )
        self.console.print(f'\t\t>> [green]Uploaded[/green] {key} for window={window}, layer={layer_idx}, file={fname}', style='tan')
    
    
    async def mine(self, window:int):
        """
        Performs the mining operation for the current window.
        Executes forward or backward pass for the assigned layer(s) in the pipeline.
        
        Args:
            window (int): Current block window
        """
        self.console.print(
            f"\n[bold cyan]Mining[/bold cyan] window = [yellow]{window}[/yellow]"
        )          
        # Process each layer assigned to this node
        for layer_idx, layer in enumerate(self.layers):
            self.console.print(
                "\n\t"
                f"layer=[green]{layer_idx}[/green], "
                f"layer=[white]{layer}[/white]"
            )
            
            # Layer 0 continuosly loads pages and uploads them.
            if layer_idx == 0:
                # Pull next inputs and upload them.
                inputs, labels = await self.get_batch( window )
                activations = await self.forward( layer_idx, inputs, require_grads = False )
                await self.upload( layer_idx, window, FORWARD, activations )
                # Pull gradients for layer above me from the previous window.
                grads = await self.download( layer_idx + 1, window - 1, BACKWARD)
                if grads is None: continue # Nothing to do.
                # Pull inputs from the window associated with this grad window. 
                matching_window = window - (len(model_layers) - layer_idx)
                inputs, labels = await self.get_batch( matching_window )
                # Run the input forward again and run the backward.
                logits = await self.forward( layer_idx, inputs, require_grads = True )
                if logits is None: continue
                logits.backward(grads)
                # Apply the gradient update.
                await self.apply_gradient( layer_idx )

            # Last layer continuosly loads latest activations and does backward.
            elif layer_idx == len(model_layers) - 1:
                # Pull activations from the layer below me at the previous window.
                activations = await self.download( layer_idx - 1, window - 1, FORWARD)
                if activations is None: continue # Nothing to do.   
                # Get the inputs from the paired window.
                _, labels = await self.get_batch( window - len(model_layers) ) 
                # Run the forward to get the uploads.
                logits = await self.forward( layer_idx, activations, require_grads = True )
                # Calculate loss
                loss = self.criterion(logits, labels)
                loss.backward()
                # Upload the gradients.
                await self.upload( layer_idx, window, BACKWARD, activations.grad )
                # Apply gradients using optimizer
                await self.apply_gradient( layer_idx )
     
            else:
                # Pull activations from the layer below me at the previous window.
                inputs = await self.download( layer_idx - 1, window - 1, FORWARD)
                if inputs is None: continue # Nothing to do. 
                # Run the forward to get the uploads.
                activations = await self.forward( layer_idx, inputs, require_grads = False )
                # Upload the activations.
                await self.upload( layer_idx, window, FORWARD, activations )
                # Pull gradients for layer above me from the previous window.
                grads = await self.download( layer_idx + 1, window - 1, BACKWARD)
                if grads is None: continue # Nothing to do.
                # Pull activations for this batch.
                matching_window = window - (len(model_layers) - layer_idx)
                activations = await self.download( layer_idx - 1, matching_window, FORWARD)
                if activations is None: continue # Nothing to do.
                # Run the input forward again and run the backward.
                logits = await self.forward( layer_idx, activations, require_grads = True )
                if logits is None: continue
                logits.backward(grads)
                # Upload the gradients.
                await self.upload( layer_idx, window, BACKWARD, activations.grad )
                # Apply gradients using optimizer
                await self.apply_gradient( layer_idx )
                
    
    async def upload_state(self, window:int):
        self.console.print(
            f"\n[bold cyan]Uploading[/bold cyan] window = [yellow]{window}[/yellow]\n"
        ) 
        # Get the randomness seed.
        seed = await self.subtensor.get_block_hash(window * WINDOW_SIZE)
        # Seet the randomness with the block hash and get a layer.
        random.seed( seed )
        layer_idx = random.choice(list( range( len(model_layers)) ))
        # Upload the state.
        await self.upload( layer_idx, window, STATE, self.layers[ layer_idx ] )

    async def check_timestamp( self, layer_idx, rank, window ) -> bool:
        # Get the forward filetimestamp
        ftime = await comms.timestamp(
            bucket = self.buckets[self.world[rank]],
            filename = f'{self.wallet.hotkey.ss58_address}-{FORWARD}-{layer_idx}-{window}.json'
        )
        # Get the time when the block was produced.
        wtime = await self.timestamp( window + 1 )
        # Check the timestamp is before the window time.
        if ftime is None or ftime > wtime:
            self.console.print(f"[bold gold]\t>> [red]Late[/red] {FORWARD} for window={window} rank={rank}, layer={layer_idx}, {ftime} > {wtime}", style='tan' )
            return False
        self.console.print(f"[bold gold]\t>> [green]OnTime[/green] {FORWARD} for window={window} rank = {rank}, layer = {layer_idx}, {ftime} > {wtime}", style='tan' )
        return True
    
    async def check_forward( self, layer_idx, rank, window ) -> bool:
        # The final layer doesn't have a forward.
        if layer_idx == len(model_layers): return True
            
        # Get inputs either from dataset or from previous miner.
        if layer_idx == 0:
            # Inputs are from the dataset.
            inputs, _ = await self.get_batch( window ) 
        else:
            inputs = await self.download( layer_idx - 1, window - 1, FORWARD)
        if inputs is None: return True
        
        # Download the activations for this window.
        acts_to_check = await self.download( layer_idx, window, FORWARD )
        if acts_to_check is None: return False

        # Download the layer for this window.
        layer = await self.download( layer_idx, window, STATE )
        if layer is None: return False

        # Check that the activations match.
        if not torch.allclose( layer(inputs), acts_to_check, rtol=1e-4, atol=1e-4):
            # Activations didnt match
            self.console.print(f"[bold red]\t>> Activations don't match [/bold red]", style='tan' )
            return False
        else:
            # Activations match.
            self.console.print(f"[bold green]\t>> Activations match [/bold green]", style='tan' )
            return True
                
    async def validate(self, window:int):
        self.console.print(
            f"\n[bold cyan]Validating[/bold cyan] window = [yellow]{window}[/yellow]\n"
        ) 
        # Get the randomness seed.
        seed = await self.subtensor.get_block_hash(window * WINDOW_SIZE)
        # Seet the randomness with the block hash and get a layer.
        random.seed( seed )
        layer_idx = random.choice(list( range( len(model_layers)) ))
        # Now get the miner rank for this layer. 
        rank: int = get_rank(layer_idx = layer_idx, worldsize = len(self.world)) 
        # Get the uid we are evalling.
        uid = self.metagraph.hotkeys.index( self.world[rank] )
        # If the timestamp is valid check the forward is legitimate.
        if (await self.check_timestamp( layer_idx, rank, window )) and (await self.check_forward( layer_idx, rank, window )):
            # Increase weights for this miner.
            self.weights[uid] = 0.01 + 0.99 * self.weights[uid]
        else:
            # Decrease weights.
            self.weights[uid] = 0.99 * self.weights[uid]
        self.console.print(f"[bold gold]\nWeights:[/bold gold] \n\t[green]{self.weights}[/green]")
        # Set weights every 360 blocks.
        if window % int(360/WINDOW_SIZE) == 0:
            await self.subtensor.set_weights(
                wallet = self.wallet,
                netuid = self.config.netuid,
                uids = self.metagraph.uids,
                weights = self.weights,
                wait_for_inclusion = False
            )
            self.console.print(f"[bold green]\nSet weights on the chain.[/boldgreen]")
    
    async def run(self):
        """
        Main runner loop.
        """
        # Run forever.
        while True:
            # Update global state.
            window = await self.window()
            self.buckets = await self.get_buckets( window )      
            self.metagraph = await self.subtensor.metagraph( self.config.netuid )  
            # Upload one layer to my bucket based on the window the corresponding bloch hash 
            await self.upload_state( window )
            # Perform forward and backward duties if I am a miner.
            await self.mine( window )
            # Pick the last uploaded model state and check that it computed activations properly.
            await self.validate( window - 1 )
            # Wait for window to end.
            while await self.window() == window:
                time.sleep(2)
                
# Main entry point.
async def main():
    runner = Pipe(Pipe.get_config())
    await runner.initialize()
    await runner.run()

# Run with asyncio.
if __name__ == "__main__":
    asyncio.run(main())