## Usage

Benchmark 3-member etcd cluster to get its read and write performance.

## Instructions

1. Start 3-member etcd cluster on 3 machines
2. Update `$leader` and `$servers` in the script
3. Run the script in a separate machine

## Caveat

1. Set environment variable `GOMAXPROCS` as the number of available cores to maximize CPU resources for both etcd member and bench process.
2. Set the number of open files per process as 10000 for amounts of client connections for both etcd member and benchmark process.
