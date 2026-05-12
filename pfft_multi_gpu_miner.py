#!/usr/bin/env python3
"""
PFFT Multi-GPU Miner Bot — NVIDIA CUDA (all GPUs, multiple wallets)

Spawns one mining process per GPU. Each GPU rotates through its assigned
wallets after each successful mint.

Wallet distribution (stride pattern):
  10 wallets + 4 GPUs:
    GPU 0 → wallets [0, 4, 8]
    GPU 1 → wallets [1, 5, 9]
    GPU 2 → wallets [2, 6]
    GPU 3 → wallets [3, 7]

Wallet config in .env:
  PRIVATE_KEY_0=...   (wallet 0)
  PRIVATE_KEY_1=...   (wallet 1)
  ...
  Or just PRIVATE_KEY=... for a single wallet on all GPUs.

Usage:
  cp .env.example .env
  python3 pfft_multi_gpu_miner.py

Optional env:
  GPU_BLOCKS=65535
  GPU_THREADS=256
  GPU_BATCHES_PER_STATUS=32
  GPUS=0,1,2           # specific GPUs to use (default: all available)
"""

from __future__ import annotations

import multiprocessing
import os
import secrets
import signal
import sys
import time
from pathlib import Path

# Load .env file if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

CONTRACT = "0xEFAd2Eab7172dDEbE5Ce7a41f5Ddf8fCcE4Ca0CB"
CHAIN_ID = 1
RPC = os.environ.get("ETH_RPC", "https://ethereum-rpc.publicnode.com")
GAS_LIMIT = int(os.environ.get("GAS_LIMIT", "200000"))
PAUSE_BETWEEN_ROUNDS = int(os.environ.get("PAUSE_BETWEEN_ROUNDS", "3"))

GPU_BLOCKS = int(os.environ.get("GPU_BLOCKS", "65535"))
GPU_THREADS = int(os.environ.get("GPU_THREADS", "256"))
GPU_BATCHES_PER_STATUS = int(os.environ.get("GPU_BATCHES_PER_STATUS", "32"))


CUDA_SOURCE = r'''
#include <stdint.h>

#define ROL64(a, offset) (((a) << (offset)) ^ ((a) >> (64 - (offset))))

__device__ __constant__ uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ __forceinline__ uint64_t load64_le(const unsigned char *x) {
    uint64_t r = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r |= ((uint64_t)x[i]) << (8 * i);
    }
    return r;
}

__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    return ((x & 0x00000000000000ffULL) << 56) |
           ((x & 0x000000000000ff00ULL) << 40) |
           ((x & 0x0000000000ff0000ULL) << 24) |
           ((x & 0x00000000ff000000ULL) << 8)  |
           ((x & 0x000000ff00000000ULL) >> 8)  |
           ((x & 0x0000ff0000000000ULL) >> 24) |
           ((x & 0x00ff000000000000ULL) >> 40) |
           ((x & 0xff00000000000000ULL) >> 56);
}

__device__ void keccakf(uint64_t st[25]) {
    const int piln[24] = {
        10, 7, 11, 17, 18, 3, 5, 16,
        8, 21, 24, 4, 15, 23, 19, 13,
        12, 2, 20, 14, 22, 9, 6, 1
    };
    const int rotc[24] = {
        1, 3, 6, 10, 15, 21, 28, 36,
        45, 55, 2, 14, 27, 41, 56, 8,
        25, 43, 62, 18, 39, 61, 20, 44
    };

    for (int round = 0; round < 24; round++) {
        uint64_t bc[5];

        #pragma unroll
        for (int i = 0; i < 5; i++) {
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        }

        #pragma unroll
        for (int i = 0; i < 5; i++) {
            uint64_t t = bc[(i + 4) % 5] ^ ROL64(bc[(i + 1) % 5], 1);
            st[i] ^= t;
            st[i + 5] ^= t;
            st[i + 10] ^= t;
            st[i + 15] ^= t;
            st[i + 20] ^= t;
        }

        uint64_t t = st[1];
        #pragma unroll
        for (int i = 0; i < 24; i++) {
            int j = piln[i];
            uint64_t tmp = st[j];
            st[j] = ROL64(t, rotc[i]);
            t = tmp;
        }

        #pragma unroll
        for (int j = 0; j < 25; j += 5) {
            uint64_t row0 = st[j + 0];
            uint64_t row1 = st[j + 1];
            uint64_t row2 = st[j + 2];
            uint64_t row3 = st[j + 3];
            uint64_t row4 = st[j + 4];
            st[j + 0] = row0 ^ ((~row1) & row2);
            st[j + 1] = row1 ^ ((~row2) & row3);
            st[j + 2] = row2 ^ ((~row3) & row4);
            st[j + 3] = row3 ^ ((~row4) & row0);
            st[j + 4] = row4 ^ ((~row0) & row1);
        }

        st[0] ^= RC[round];
    }
}

__device__ __forceinline__ unsigned char digest_byte(uint64_t st[25], int idx) {
    uint64_t lane = st[idx / 8];
    return (unsigned char)((lane >> (8 * (idx % 8))) & 0xff);
}

__device__ bool digest_le_target(uint64_t st[25], const unsigned char *target) {
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        unsigned char d = digest_byte(st, i);
        unsigned char t = target[i];
        if (d < t) return true;
        if (d > t) return false;
    }
    return true;
}

extern "C" __global__ void mine_kernel(
    const unsigned char *challenge,
    const unsigned char *target,
    unsigned long long start_nonce,
    unsigned long long *nonce_out,
    int *found
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long nonce = start_nonce + idx;

    if (found[0] != 0) return;

    uint64_t st[25];
    #pragma unroll
    for (int i = 0; i < 25; i++) st[i] = 0ULL;

    st[0] = load64_le(challenge + 0);
    st[1] = load64_le(challenge + 8);
    st[2] = load64_le(challenge + 16);
    st[3] = load64_le(challenge + 24);
    st[4] = 0ULL;
    st[5] = 0ULL;
    st[6] = 0ULL;
    st[7] = bswap64(nonce);

    st[8] ^= 0x0000000000000001ULL;
    st[16] ^= 0x8000000000000000ULL;

    keccakf(st);

    if (digest_le_target(st, target)) {
        if (atomicCAS(found, 0, 1) == 0) {
            nonce_out[0] = nonce;
        }
    }
}
'''


ABI = [
    {
        "inputs": [],
        "name": "currentPowHexZeros",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "totalMinted",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "MAX_SUPPLY",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "requested", "type": "uint256"}],
        "name": "calculateActualMint",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "user", "type": "address"}],
        "name": "currentPowChallenge",
        "outputs": [{"type": "bytes32"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "user", "type": "address"},
            {"name": "powNonce", "type": "uint256"},
        ],
        "name": "isValidPow",
        "outputs": [{"type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "powNonce", "type": "uint256"}],
        "name": "freeMint",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"name": "user", "type": "address"}],
        "name": "mintedByAddress",
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]


def get_all_private_keys() -> list[str]:
    """Collect all available private keys from env (PRIVATE_KEY_0, _1, ... and PRIVATE_KEY)."""
    keys = []
    for i in range(100):
        k = os.environ.get(f"PRIVATE_KEY_{i}", "").strip()
        if k and k != "your_private_key_here":
            keys.append(k)
        elif i > 10 and not k:
            break
    if not keys:
        k = os.environ.get("PRIVATE_KEY", "").strip()
        if k and k != "your_private_key_here":
            keys.append(k)
    return keys


def get_available_gpus() -> list[int]:
    """Get list of GPU IDs to use."""
    gpus_env = os.environ.get("GPUS", "")
    if gpus_env:
        return [int(x.strip()) for x in gpus_env.split(",") if x.strip()]
    try:
        import pycuda.driver as cuda
        cuda.init()
        count = cuda.Device.count()
        return list(range(count))
    except Exception:
        return [0]


def gpu_worker(
    gpu_id: int,
    worker_index: int,
    total_workers: int,
    wallet_keys: list[str],
    stop_event: multiprocessing.Event,
):
    """Worker process for a single GPU with wallet rotation.

    Args:
        gpu_id: CUDA device index
        worker_index: sequential index (0..N-1) for nonce partitioning
        total_workers: total number of GPU workers
        wallet_keys: list of hex private keys this worker rotates through
        stop_event: shared event to signal graceful shutdown
    """
    import numpy as np
    import pycuda.driver as cuda

    cuda.init()
    device = cuda.Device(gpu_id)
    ctx = device.make_context()

    try:
        from pycuda.compiler import SourceModule
        from eth_account import Account
        from web3 import Web3

        module = SourceModule(CUDA_SOURCE, no_extern_c=True)
        kernel = module.get_function("mine_kernel")

        prefix = f"[GPU {gpu_id}]"
        print(f"{prefix} ✅ {device.name()} | Compute {device.compute_capability()}")

        w3 = Web3(Web3.HTTPProvider(RPC, request_kwargs={"timeout": 30}))
        if not w3.is_connected():
            print(f"{prefix} ❌ Cannot connect to RPC")
            return

        contract = w3.eth.contract(
            address=w3.to_checksum_address(CONTRACT), abi=ABI
        )

        wallets = []
        for k in wallet_keys:
            if not k.startswith("0x"):
                k = "0x" + k
            wallets.append(Account.from_key(k))

        print(f"{prefix} 🔑 {len(wallets)} wallet(s) assigned:")
        for i, wlt in enumerate(wallets):
            print(f"{prefix}    [{i}] {wlt.address}")

        # Nonce partitioning: divide uint64 space across all workers
        nonce_space = 2**64
        slice_size = nonce_space // total_workers
        my_nonce_start = worker_index * slice_size
        my_nonce_end = (worker_index + 1) * slice_size if worker_index < total_workers - 1 else nonce_space
        print(
            f"{prefix} 🔢 Nonce range: 0x{my_nonce_start:016x} - 0x{(my_nonce_end - 1):016x}"
        )

        total_mints = 0
        total_pfft = 0.0
        round_num = 0
        global_start = time.time()
        wallet_idx = 0
        capped_wallets: set[int] = set()

        while not stop_event.is_set():
            if len(capped_wallets) >= len(wallets):
                print(f"{prefix} 🏁 All {len(wallets)} wallet(s) reached cap!")
                break

            if wallet_idx in capped_wallets:
                wallet_idx = (wallet_idx + 1) % len(wallets)
                continue

            wallet = wallets[wallet_idx]
            round_num += 1
            print(f"\n{prefix} {'─' * 50}")
            print(f"{prefix} Round #{round_num} | Wallet [{wallet_idx}] {wallet.address}")
            print(f"{prefix} {'─' * 50}")

            try:
                hex_zeros = contract.functions.currentPowHexZeros().call()
                total_minted = contract.functions.totalMinted().call()
                max_supply = contract.functions.MAX_SUPPLY().call()
                next_mint = contract.functions.calculateActualMint(
                    w3.to_wei(1000, "ether")
                ).call()
                wallet_minted = contract.functions.mintedByAddress(wallet.address).call()
                wallet_bal = contract.functions.balanceOf(wallet.address).call()
                target = (2**256 - 1) >> (hex_zeros * 4)
                progress = total_minted * 10000 / max_supply / 100

                print(
                    f"{prefix} Supply: {total_minted / 1e18:,.0f} "
                    f"({progress:.1f}%) | "
                    f"Next: ~{next_mint / 1e18:,.2f} PFFT | "
                    f"Diff: {hex_zeros * 4}-bit"
                )
                print(
                    f"{prefix} Wallet minted: {wallet_minted / 1e18:,.2f} / "
                    f"10,000 PFFT | Balance: {wallet_bal / 1e18:,.2f} PFFT"
                )

                if total_minted >= max_supply:
                    print(f"{prefix} 🏁 Max supply reached!")
                    break
                if wallet_minted >= 10_000 * 10**18:
                    print(f"{prefix} 🏁 Wallet [{wallet_idx}] cap reached, rotating...")
                    capped_wallets.add(wallet_idx)
                    wallet_idx = (wallet_idx + 1) % len(wallets)
                    continue
            except Exception as exc:
                print(f"{prefix} ⚠️  Status error: {exc}, retrying in 15s...")
                time.sleep(15)
                continue

            eth_bal = w3.eth.get_balance(wallet.address) / 1e18
            if eth_bal < 0.00005:
                print(f"{prefix} ⚠️  Wallet [{wallet_idx}] low ETH ({eth_bal:.6f}), rotating...")
                wallet_idx = (wallet_idx + 1) % len(wallets)
                continue

            challenge = contract.functions.currentPowChallenge(wallet.address).call()
            if not isinstance(challenge, bytes):
                challenge = challenge.to_bytes(32, "big")

            print(f"{prefix} ⛏️  Mining ({hex_zeros * 4}-bit difficulty)...")
            nonce = _gpu_mine(
                np, cuda, kernel, challenge, target, prefix,
                stop_event, my_nonce_start, my_nonce_end,
            )
            if nonce is None:
                if stop_event.is_set():
                    print(f"{prefix} Stopped by signal")
                else:
                    print(f"{prefix} Nonce range exhausted, re-randomizing...")
                continue

            try:
                valid = contract.functions.isValidPow(wallet.address, nonce).call()
                if not valid:
                    print(f"{prefix} ⚠️  Nonce invalid on-chain, restarting...")
                    continue
            except Exception as exc:
                print(f"{prefix} ⚠️  Verify error: {exc}, submitting anyway...")

            if _submit_mint(w3, wallet, contract, nonce, prefix):
                total_mints += 1
                earned = next_mint / 1e18
                total_pfft += earned
                print(
                    f"{prefix} 💰 +{earned:,.2f} PFFT | "
                    f"Total: {total_pfft:,.2f} PFFT from {total_mints} mints"
                )

            elapsed = time.time() - global_start
            print(
                f"\n{prefix} 📈 Session: {total_mints} mints | "
                f"{total_pfft:,.2f} PFFT | {elapsed / 60:.1f} min"
            )

            # Rotate to next wallet after each round
            wallet_idx = (wallet_idx + 1) % len(wallets)

            print(f"{prefix} ⏳ {PAUSE_BETWEEN_ROUNDS}s cooldown...")
            for _ in range(PAUSE_BETWEEN_ROUNDS * 10):
                if stop_event.is_set():
                    break
                time.sleep(0.1)

        print(f"\n{prefix} {'=' * 50}")
        print(f"{prefix} Session Summary")
        print(f"{prefix} Mints: {total_mints}")
        print(f"{prefix} PFFT earned: {total_pfft:,.2f}")
        print(f"{prefix} Runtime: {(time.time() - global_start) / 60:.1f} min")
        print(f"{prefix} {'=' * 50}")

    finally:
        ctx.pop()
        ctx.detach()


def _gpu_mine(
    np, cuda, kernel,
    challenge: bytes,
    target: int,
    prefix: str,
    stop_event: multiprocessing.Event,
    nonce_range_start: int,
    nonce_range_end: int,
):
    """Run GPU mining loop with partitioned nonce range.

    Returns nonce (int) or None if stopped/exhausted.
    """
    challenge_np = np.frombuffer(challenge, dtype=np.uint8).copy()
    target_np = np.frombuffer(target.to_bytes(32, "big"), dtype=np.uint8).copy()
    found_np = np.zeros(1, dtype=np.int32)
    nonce_np = np.zeros(1, dtype=np.uint64)

    challenge_gpu = cuda.mem_alloc(challenge_np.nbytes)
    target_gpu = cuda.mem_alloc(target_np.nbytes)
    found_gpu = cuda.mem_alloc(found_np.nbytes)
    nonce_gpu = cuda.mem_alloc(nonce_np.nbytes)

    cuda.memcpy_htod(challenge_gpu, challenge_np)
    cuda.memcpy_htod(target_gpu, target_np)

    batch_size = GPU_BLOCKS * GPU_THREADS
    range_size = nonce_range_end - nonce_range_start

    usable = range_size - (batch_size * GPU_BATCHES_PER_STATUS)
    if usable > 0:
        start_nonce = nonce_range_start + secrets.randbelow(usable)
    else:
        start_nonce = nonce_range_start

    total_hashes = 0
    start_time = time.time()
    last_report = start_time

    print(f"{prefix} 🎲 Start nonce: {start_nonce} (range 0x{nonce_range_start:016x}..)")

    while not stop_event.is_set():
        if start_nonce >= nonce_range_end:
            start_nonce = nonce_range_start

        found_np[0] = 0
        nonce_np[0] = 0
        cuda.memcpy_htod(found_gpu, found_np)
        cuda.memcpy_htod(nonce_gpu, nonce_np)

        for _ in range(GPU_BATCHES_PER_STATUS):
            if stop_event.is_set():
                return None

            kernel(
                challenge_gpu,
                target_gpu,
                np.uint64(start_nonce & 0xFFFFFFFFFFFFFFFF),
                nonce_gpu,
                found_gpu,
                block=(GPU_THREADS, 1, 1),
                grid=(GPU_BLOCKS, 1),
            )
            cuda.Context.synchronize()
            cuda.memcpy_dtoh(found_np, found_gpu)

            total_hashes += batch_size
            if found_np[0]:
                cuda.memcpy_dtoh(nonce_np, nonce_gpu)
                elapsed = time.time() - start_time
                rate = total_hashes / elapsed if elapsed > 0 else 0
                nonce = int(nonce_np[0])
                print(
                    f"\n{prefix} ✅ FOUND nonce={nonce} | "
                    f"{total_hashes:,} checked | {rate:,.0f} H/s"
                )
                return nonce

            start_nonce += batch_size

        now = time.time()
        if now - last_report >= 5:
            elapsed = now - start_time
            rate = total_hashes / elapsed if elapsed > 0 else 0
            print(
                f"{prefix} ⚡ {rate:,.0f} H/s | "
                f"checked {total_hashes:,} | nonce {start_nonce:,}"
            )
            last_report = now

    return None


def _submit_mint(w3, wallet, contract, nonce: int, prefix: str) -> bool:
    """Submit freeMint transaction."""
    try:
        fn = contract.functions.freeMint(nonce)
        tx = fn.build_transaction(
            {
                "from": wallet.address,
                "nonce": w3.eth.get_transaction_count(wallet.address),
                "chainId": CHAIN_ID,
                "gas": GAS_LIMIT,
            }
        )
        if "maxFeePerGas" not in tx and "maxPriorityFeePerGas" not in tx:
            tx["gasPrice"] = w3.eth.gas_price

        signed = wallet.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"{prefix} 📤 TX: https://etherscan.io/tx/0x{tx_hash.hex()}")

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        if receipt.status == 1:
            print(f"{prefix} ✅ MINT OK | Block {receipt.blockNumber} | Gas {receipt.gasUsed}")
            return True

        print(f"{prefix} ❌ REVERTED | Gas {receipt.gasUsed}")
        return False
    except Exception as exc:
        print(f"{prefix} ❌ TX error: {exc}")
        return False


def main():
    multiprocessing.set_start_method("spawn", force=True)

    print("=" * 60)
    print("  🚀 PFFT Multi-GPU Miner Bot — NVIDIA CUDA")
    print(f"  Contract: {CONTRACT}")
    print(f"  RPC: {RPC}")
    print(f"  Grid: {GPU_BLOCKS} blocks x {GPU_THREADS} threads per GPU")
    print("=" * 60)

    gpu_ids = get_available_gpus()
    print(f"\n  🎮 Detected {len(gpu_ids)} GPU(s): {gpu_ids}")

    all_keys = get_all_private_keys()
    if not all_keys:
        print("\n❌ No valid private keys found!")
        print("   Set PRIVATE_KEY_0, PRIVATE_KEY_1, ... in .env")
        print("   Or set PRIVATE_KEY to use same wallet on all GPUs")
        sys.exit(1)

    total_workers = len(gpu_ids)
    print(f"  🔑 Found {len(all_keys)} wallet(s), distributing across {total_workers} GPU(s)")

    # Distribute wallets to GPUs using stride: GPU0 gets [0,4,8], GPU1 gets [1,5,9], etc.
    worker_wallet_map: list[tuple[int, int, list[str]]] = []
    for idx, gpu_id in enumerate(gpu_ids):
        my_keys = [all_keys[i] for i in range(idx, len(all_keys), total_workers)]
        my_wallet_indices = list(range(idx, len(all_keys), total_workers))
        print(f"     GPU {gpu_id} (worker #{idx}) → wallets {my_wallet_indices}")
        worker_wallet_map.append((gpu_id, idx, my_keys))

    stop_event = multiprocessing.Event()

    def _signal_handler(sig, frame):
        print("\n\n  ⚠️  Stopping all GPU workers...")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print(f"\n  ✅ Starting {len(worker_wallet_map)} worker(s)...\n")

    processes = []
    for gpu_id, worker_idx, keys in worker_wallet_map:
        p = multiprocessing.Process(
            target=gpu_worker,
            args=(gpu_id, worker_idx, total_workers, keys, stop_event),
            name=f"gpu-{gpu_id}",
        )
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Force stopping all GPU workers...")
        stop_event.set()
        for p in processes:
            p.join(timeout=10)
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        print("  Done.")


if __name__ == "__main__":
    main()
