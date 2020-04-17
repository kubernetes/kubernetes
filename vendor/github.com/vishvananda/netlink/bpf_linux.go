package netlink

import (
	"unsafe"

	"golang.org/x/sys/unix"
)

type BpfProgType uint32

const (
	BPF_PROG_TYPE_UNSPEC BpfProgType = iota
	BPF_PROG_TYPE_SOCKET_FILTER
	BPF_PROG_TYPE_KPROBE
	BPF_PROG_TYPE_SCHED_CLS
	BPF_PROG_TYPE_SCHED_ACT
	BPF_PROG_TYPE_TRACEPOINT
	BPF_PROG_TYPE_XDP
)

type BPFAttr struct {
	ProgType    uint32
	InsnCnt     uint32
	Insns       uintptr
	License     uintptr
	LogLevel    uint32
	LogSize     uint32
	LogBuf      uintptr
	KernVersion uint32
}

// loadSimpleBpf loads a trivial bpf program for testing purposes.
func loadSimpleBpf(progType BpfProgType, ret uint32) (int, error) {
	insns := []uint64{
		0x00000000000000b7 | (uint64(ret) << 32),
		0x0000000000000095,
	}
	license := []byte{'A', 'S', 'L', '2', '\x00'}
	attr := BPFAttr{
		ProgType: uint32(progType),
		InsnCnt:  uint32(len(insns)),
		Insns:    uintptr(unsafe.Pointer(&insns[0])),
		License:  uintptr(unsafe.Pointer(&license[0])),
	}
	fd, _, errno := unix.Syscall(unix.SYS_BPF,
		5, /* bpf cmd */
		uintptr(unsafe.Pointer(&attr)),
		unsafe.Sizeof(attr))
	if errno != 0 {
		return 0, errno
	}
	return int(fd), nil
}
