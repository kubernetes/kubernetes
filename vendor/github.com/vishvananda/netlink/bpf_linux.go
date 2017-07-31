package netlink

/*
#include <asm/types.h>
#include <asm/unistd.h>
#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

static int load_simple_bpf(int prog_type, int ret) {
#ifdef __NR_bpf
	// { return ret; }
	__u64 __attribute__((aligned(8))) insns[] = {
		0x00000000000000b7ull | ((__u64)ret<<32),
		0x0000000000000095ull,
	};
	__u8 __attribute__((aligned(8))) license[] = "ASL2";
	// Copied from a header file since libc is notoriously slow to update.
	// The call will succeed or fail and that will be our indication on
	// whether or not it is supported.
	struct {
		__u32 prog_type;
		__u32 insn_cnt;
		__u64 insns;
		__u64 license;
		__u32 log_level;
		__u32 log_size;
		__u64 log_buf;
		__u32 kern_version;
	} __attribute__((aligned(8))) attr = {
		.prog_type = prog_type,
		.insn_cnt = 2,
		.insns = (uintptr_t)&insns,
		.license = (uintptr_t)&license,
	};
	return syscall(__NR_bpf, 5, &attr, sizeof(attr));
#else
	errno = EINVAL;
	return -1;
#endif
}
*/
import "C"

type BpfProgType C.int

const (
	BPF_PROG_TYPE_UNSPEC BpfProgType = iota
	BPF_PROG_TYPE_SOCKET_FILTER
	BPF_PROG_TYPE_KPROBE
	BPF_PROG_TYPE_SCHED_CLS
	BPF_PROG_TYPE_SCHED_ACT
	BPF_PROG_TYPE_TRACEPOINT
	BPF_PROG_TYPE_XDP
)

// loadSimpleBpf loads a trivial bpf program for testing purposes
func loadSimpleBpf(progType BpfProgType, ret int) (int, error) {
	fd, err := C.load_simple_bpf(C.int(progType), C.int(ret))
	return int(fd), err
}
