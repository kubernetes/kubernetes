package sys

import (
	"runtime"
	"syscall"
	"unsafe"

	"github.com/cilium/ebpf/internal/unix"
)

// ENOTSUPP is a Linux internal error code that has leaked into UAPI.
//
// It is not the same as ENOTSUP or EOPNOTSUPP.
var ENOTSUPP = syscall.Errno(524)

// BPF wraps SYS_BPF.
//
// Any pointers contained in attr must use the Pointer type from this package.
func BPF(cmd Cmd, attr unsafe.Pointer, size uintptr) (uintptr, error) {
	// Prevent the Go profiler from repeatedly interrupting the verifier,
	// which could otherwise lead to a livelock due to receiving EAGAIN.
	if cmd == BPF_PROG_LOAD || cmd == BPF_PROG_RUN {
		maskProfilerSignal()
		defer unmaskProfilerSignal()
	}

	for {
		r1, _, errNo := unix.Syscall(unix.SYS_BPF, uintptr(cmd), uintptr(attr), size)
		runtime.KeepAlive(attr)

		// As of ~4.20 the verifier can be interrupted by a signal,
		// and returns EAGAIN in that case.
		if errNo == unix.EAGAIN && cmd == BPF_PROG_LOAD {
			continue
		}

		var err error
		if errNo != 0 {
			err = wrappedErrno{errNo}
		}

		return r1, err
	}
}

// Info is implemented by all structs that can be passed to the ObjInfo syscall.
//
//	MapInfo
//	ProgInfo
//	LinkInfo
//	BtfInfo
type Info interface {
	info() (unsafe.Pointer, uint32)
}

var _ Info = (*MapInfo)(nil)

func (i *MapInfo) info() (unsafe.Pointer, uint32) {
	return unsafe.Pointer(i), uint32(unsafe.Sizeof(*i))
}

var _ Info = (*ProgInfo)(nil)

func (i *ProgInfo) info() (unsafe.Pointer, uint32) {
	return unsafe.Pointer(i), uint32(unsafe.Sizeof(*i))
}

var _ Info = (*LinkInfo)(nil)

func (i *LinkInfo) info() (unsafe.Pointer, uint32) {
	return unsafe.Pointer(i), uint32(unsafe.Sizeof(*i))
}

var _ Info = (*BtfInfo)(nil)

func (i *BtfInfo) info() (unsafe.Pointer, uint32) {
	return unsafe.Pointer(i), uint32(unsafe.Sizeof(*i))
}

// ObjInfo retrieves information about a BPF Fd.
//
// info may be one of MapInfo, ProgInfo, LinkInfo and BtfInfo.
func ObjInfo(fd *FD, info Info) error {
	ptr, len := info.info()
	err := ObjGetInfoByFd(&ObjGetInfoByFdAttr{
		BpfFd:   fd.Uint(),
		InfoLen: len,
		Info:    NewPointer(ptr),
	})
	runtime.KeepAlive(fd)
	return err
}

// BPFObjName is a null-terminated string made up of
// 'A-Za-z0-9_' characters.
type ObjName [unix.BPF_OBJ_NAME_LEN]byte

// NewObjName truncates the result if it is too long.
func NewObjName(name string) ObjName {
	var result ObjName
	copy(result[:unix.BPF_OBJ_NAME_LEN-1], name)
	return result
}

// LogLevel controls the verbosity of the kernel's eBPF program verifier.
type LogLevel uint32

const (
	BPF_LOG_LEVEL1 LogLevel = 1 << iota
	BPF_LOG_LEVEL2
	BPF_LOG_STATS
)

// LinkID uniquely identifies a bpf_link.
type LinkID uint32

// BTFID uniquely identifies a BTF blob loaded into the kernel.
type BTFID uint32

// MapFlags control map behaviour.
type MapFlags uint32

//go:generate stringer -type MapFlags

const (
	BPF_F_NO_PREALLOC MapFlags = 1 << iota
	BPF_F_NO_COMMON_LRU
	BPF_F_NUMA_NODE
	BPF_F_RDONLY
	BPF_F_WRONLY
	BPF_F_STACK_BUILD_ID
	BPF_F_ZERO_SEED
	BPF_F_RDONLY_PROG
	BPF_F_WRONLY_PROG
	BPF_F_CLONE
	BPF_F_MMAPABLE
	BPF_F_PRESERVE_ELEMS
	BPF_F_INNER_MAP
)

// wrappedErrno wraps syscall.Errno to prevent direct comparisons with
// syscall.E* or unix.E* constants.
//
// You should never export an error of this type.
type wrappedErrno struct {
	syscall.Errno
}

func (we wrappedErrno) Unwrap() error {
	return we.Errno
}

func (we wrappedErrno) Error() string {
	if we.Errno == ENOTSUPP {
		return "operation not supported"
	}
	return we.Errno.Error()
}

type syscallError struct {
	error
	errno syscall.Errno
}

func Error(err error, errno syscall.Errno) error {
	return &syscallError{err, errno}
}

func (se *syscallError) Is(target error) bool {
	return target == se.error
}

func (se *syscallError) Unwrap() error {
	return se.errno
}
