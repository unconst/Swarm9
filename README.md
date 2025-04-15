# Swarm9

This codebase implements incentivized pipeline parallelism for a distributed NN on Bittensor.

## 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Create .venv
```bash
uv venv
source .venv/bin/activate
```

## 3. Install Dependencies
```bash
uv sync
```

## 4. Set Up Environment Variables
```bash
touch .env
```
Open the file and add your environment variables for R2.
```
# Replace these values with your own R2 credentials in the .env
R2_BUCKET_ID = "your_bucket_id"
R2_ACCOUNT_ID = "your_account_id" 
R2_GRADIENTS_BUCKET_NAME = "your_bucket_name"
R2_WRITE_ACCESS_KEY_ID = "your_access_key_id"
R2_WRITE_SECRET_ACCESS_KEY = "your_secret_access_key"
```

## 5. Register on network
```bash
btcli subnet register --wallet.name ... --wallet.hotkey ... --netuid ...
```

## 6. Run miner and validator
```bash
python run.py --wallet.name ... --wallet.hotkey ... --netuid ...
```
