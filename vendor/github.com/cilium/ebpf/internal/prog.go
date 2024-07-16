package internal

// EmptyBPFContext is the smallest-possible BPF input context to be used for
// invoking `Program.{Run,Benchmark,Test}`.
//
// Programs require a context input buffer of at least 15 bytes. Looking in
// net/bpf/test_run.c, bpf_test_init() requires that the input is at least
// ETH_HLEN (14) bytes. As of Linux commit fd18942 ("bpf: Don't redirect packets
// with invalid pkt_len"), it also requires the skb to be non-empty after
// removing the Layer 2 header.
var EmptyBPFContext = make([]byte, 15)
