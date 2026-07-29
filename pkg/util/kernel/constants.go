/*
Copyright 2023 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kernel

// IPLocalReservedPortsNamespacedKernelVersion is the kernel version in which net.ipv4.ip_local_reserved_ports was namespaced(netns).
// (ref: https://github.com/torvalds/linux/commit/122ff243f5f104194750ecbc76d5946dd1eec934)
const IPLocalReservedPortsNamespacedKernelVersion = "3.16"

// IPVSConnReuseModeMinSupportedKernelVersion is the minium kernel version supporting net.ipv4.vs.conn_reuse_mode.
// (ref: https://github.com/torvalds/linux/commit/d752c364571743d696c2a54a449ce77550c35ac5)
const IPVSConnReuseModeMinSupportedKernelVersion = "4.1"

// TCPKeepAliveTimeNamespacedKernelVersion is the kernel version in which net.ipv4.tcp_keepalive_time was namespaced(netns).
// (ref: https://github.com/torvalds/linux/commit/13b287e8d1cad951634389f85b8c9b816bd3bb1e)
const TCPKeepAliveTimeNamespacedKernelVersion = "4.5"

// TCPKeepAliveIntervalNamespacedKernelVersion is the kernel version in which net.ipv4.tcp_keepalive_intvl was namespaced(netns).
// (ref: https://github.com/torvalds/linux/commit/b840d15d39128d08ed4486085e5507d2617b9ae1)
const TCPKeepAliveIntervalNamespacedKernelVersion = "4.5"

// TCPKeepAliveProbesNamespacedKernelVersion is the kernel version in which net.ipv4.tcp_keepalive_probes was namespaced(netns).
// (ref: https://github.com/torvalds/linux/commit/9bd6861bd4326e3afd3f14a9ec8a723771fb20bb)
const TCPKeepAliveProbesNamespacedKernelVersion = "4.5"

// TCPFinTimeoutNamespacedKernelVersion is the kernel version in which net.ipv4.tcp_fin_timeout was namespaced(netns).
// (ref: https://github.com/torvalds/linux/commit/1e579caa18b96f9eb18f4f5416658cd15f37c062)
const TCPFinTimeoutNamespacedKernelVersion = "4.6"

// IPVSConnReuseModeFixedKernelVersion is the kernel version in which net.ipv4.vs.conn_reuse_mode was fixed.
// (ref: https://github.com/torvalds/linux/commit/35dfb013149f74c2be1ff9c78f14e6a3cd1539d1)
const IPVSConnReuseModeFixedKernelVersion = "5.9"

const TmpfsNoswapSupportKernelVersion = "6.4"

// NFTablesKubeProxyKernelVersion is the lowest kernel version kube-proxy supports using
// nftables mode with by default. This is not directly related to any specific kernel
// commit; see https://issues.k8s.io/122743#issuecomment-1893922424
const NFTablesKubeProxyKernelVersion = "5.13"

// TCPReceiveMemoryNamespacedKernelVersion is the kernel version in which net.ipv4.tcp_rmem was namespaced(netns).
// (ref: https://github.com/torvalds/linux/commit/356d1833b638bd465672aefeb71def3ab93fc17d)
const TCPReceiveMemoryNamespacedKernelVersion = "4.15"

// TCPTransmitMemoryNamespacedKernelVersion is the kernel version in which net.ipv4.tcp_wmem was namespaced(netns).
// (ref: https://github.com/torvalds/linux/commit/356d1833b638bd465672aefeb71def3ab93fc17d)
const TCPTransmitMemoryNamespacedKernelVersion = "4.15"

// MemoryQoSMinKernelVersion is the minimum kernel version for safe memory.high throttling.
// Kernels older than 5.9 have a livelock bug where processes can get stuck in an infinite
// reclaim loop at the memory.high boundary.
// (ref: https://github.com/torvalds/linux/commit/b3ff92916af3156df27716bb080a407e4caf9085)
const MemoryQoSMinKernelVersion = "5.9"

// TCPSlowStartAfterIdleNamespacedKernelVersion is the kernel version in which net.ipv4.tcp_slow_start_after_idle was namespaced(netns).
// (ref: https://github.com/torvalds/linux/commit/b510f0d23a47c3d1f074fe583e7867dc4918fe02)
const TCPSlowStartAfterIdleNamespacedKernelVersion = "4.15"

// TCPNotSentLowatNamespacedKernelVersion is the kernel version in which net.ipv4.tcp_notsent_lowat was namespaced(netns).
// (ref: https://github.com/torvalds/linux/commit/4979f2d9f7262b9b180bc83de8d70f7a7721c085)
const TCPNotSentLowatNamespacedKernelVersion = "4.6"
