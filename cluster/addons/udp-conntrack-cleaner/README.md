# UDP Conntrack Cleaner

## Description
This addon provides a mitigation for **Issue #136213** (UDP Blackhole). 

In scenarios where a UDP client persists traffic during a network partition, the Linux kernel `conntrack` table may mark the flow as `[UNREPLIED]` and persist this state even after connectivity is restored. This prevents the traffic from being re-routed correctly by `kube-proxy`.

## How it works
This DaemonSet:
1. Runs on every node with `hostNetwork: true`.
2. Grants `NET_ADMIN` capabilities to interact with the kernel network stack.
3. Periodically (every 60s) flushes UDP conntrack entries marked as `UNREPLIED`.

## Usage
Apply this manifest if you experience UDP connectivity drops after network flaps.

```bash
kubectl apply -f udp-conntrack-cleaner.yaml