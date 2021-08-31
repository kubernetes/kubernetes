package internal

import (
	"fmt"
	"path/filepath"
	"runtime"
	"syscall"
	"unsafe"

	"github.com/cilium/ebpf/internal/unix"
)

//go:generate stringer -output syscall_string.go -type=BPFCmd

// BPFCmd identifies a subcommand of the bpf syscall.
type BPFCmd int

// Well known BPF commands.
const (
	BPF_MAP_CREATE BPFCmd = iota
	BPF_MAP_LOOKUP_ELEM
	BPF_MAP_UPDATE_ELEM
	BPF_MAP_DELETE_ELEM
	BPF_MAP_GET_NEXT_KEY
	BPF_PROG_LOAD
	BPF_OBJ_PIN
	BPF_OBJ_GET
	BPF_PROG_ATTACH
	BPF_PROG_DETACH
	BPF_PROG_TEST_RUN
	BPF_PROG_GET_NEXT_ID
	BPF_MAP_GET_NEXT_ID
	BPF_PROG_GET_FD_BY_ID
	BPF_MAP_GET_FD_BY_ID
	BPF_OBJ_GET_INFO_BY_FD
	BPF_PROG_QUERY
	BPF_RAW_TRACEPOINT_OPEN
	BPF_BTF_LOAD
	BPF_BTF_GET_FD_BY_ID
	BPF_TASK_FD_QUERY
	BPF_MAP_LOOKUP_AND_DELETE_ELEM
	BPF_MAP_FREEZE
	BPF_BTF_GET_NEXT_ID
	BPF_MAP_LOOKUP_BATCH
	BPF_MAP_LOOKUP_AND_DELETE_BATCH
	BPF_MAP_UPDATE_BATCH
	BPF_MAP_DELETE_BATCH
	BPF_LINK_CREATE
	BPF_LINK_UPDATE
	BPF_LINK_GET_FD_BY_ID
	BPF_LINK_GET_NEXT_ID
	BPF_ENABLE_STATS
	BPF_ITER_CREATE
)

// BPF wraps SYS_BPF.
//
// Any pointers contained in attr must use the Pointer type from this package.
func BPF(cmd BPFCmd, attr unsafe.Pointer, size uintptr) (uintptr, error) {
	r1, _, errNo := unix.Syscall(unix.SYS_BPF, uintptr(cmd), uintptr(attr), size)
	runtime.KeepAlive(attr)

	var err error
	if errNo != 0 {
		err = wrappedErrno{errNo}
	}

	return r1, err
}

type BPFProgAttachAttr struct {
	TargetFd     uint32
	AttachBpfFd  uint32
	AttachType   uint32
	AttachFlags  uint32
	ReplaceBpfFd uint32
}

func BPFProgAttach(attr *BPFProgAttachAttr) error {
	_, err := BPF(BPF_PROG_ATTACH, unsafe.Pointer(attr), unsafe.Sizeof(*attr))
	return err
}

type BPFProgDetachAttr struct {
	TargetFd    uint32
	AttachBpfFd uint32
	AttachType  uint32
}

func BPFProgDetach(attr *BPFProgDetachAttr) error {
	_, err := BPF(BPF_PROG_DETACH, unsafe.Pointer(attr), unsafe.Sizeof(*attr))
	return err
}

type BPFEnableStatsAttr struct {
	StatsType uint32
}

func BPFEnableStats(attr *BPFEnableStatsAttr) (*FD, error) {
	ptr, err := BPF(BPF_ENABLE_STATS, unsafe.Pointer(attr), unsafe.Sizeof(*attr))
	if err != nil {
		return nil, fmt.Errorf("enable stats: %w", err)
	}
	return NewFD(uint32(ptr)), nil

}

type bpfObjAttr struct {
	fileName  Pointer
	fd        uint32
	fileFlags uint32
}

const bpfFSType = 0xcafe4a11

// BPFObjPin wraps BPF_OBJ_PIN.
func BPFObjPin(fileName string, fd *FD) error {
	dirName := filepath.Dir(fileName)
	var statfs unix.Statfs_t
	if err := unix.Statfs(dirName, &statfs); err != nil {
		return err
	}
	if uint64(statfs.Type) != bpfFSType {
		return fmt.Errorf("%s is not on a bpf filesystem", fileName)
	}

	value, err := fd.Value()
	if err != nil {
		return err
	}

	attr := bpfObjAttr{
		fileName: NewStringPointer(fileName),
		fd:       value,
	}
	_, err = BPF(BPF_OBJ_PIN, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	if err != nil {
		return fmt.Errorf("pin object %s: %w", fileName, err)
	}
	return nil
}

// BPFObjGet wraps BPF_OBJ_GET.
func BPFObjGet(fileName string, flags uint32) (*FD, error) {
	attr := bpfObjAttr{
		fileName:  NewStringPointer(fileName),
		fileFlags: flags,
	}
	ptr, err := BPF(BPF_OBJ_GET, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	if err != nil {
		return nil, fmt.Errorf("get object %s: %w", fileName, err)
	}
	return NewFD(uint32(ptr)), nil
}

type bpfObjGetInfoByFDAttr struct {
	fd      uint32
	infoLen uint32
	info    Pointer
}

// BPFObjGetInfoByFD wraps BPF_OBJ_GET_INFO_BY_FD.
//
// Available from 4.13.
func BPFObjGetInfoByFD(fd *FD, info unsafe.Pointer, size uintptr) error {
	value, err := fd.Value()
	if err != nil {
		return err
	}

	attr := bpfObjGetInfoByFDAttr{
		fd:      value,
		infoLen: uint32(size),
		info:    NewPointer(info),
	}
	_, err = BPF(BPF_OBJ_GET_INFO_BY_FD, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	if err != nil {
		return fmt.Errorf("fd %v: %w", fd, err)
	}
	return nil
}

// BPFObjName is a null-terminated string made up of
// 'A-Za-z0-9_' characters.
type BPFObjName [unix.BPF_OBJ_NAME_LEN]byte

// NewBPFObjName truncates the result if it is too long.
func NewBPFObjName(name string) BPFObjName {
	var result BPFObjName
	copy(result[:unix.BPF_OBJ_NAME_LEN-1], name)
	return result
}

type BPFMapCreateAttr struct {
	MapType        uint32
	KeySize        uint32
	ValueSize      uint32
	MaxEntries     uint32
	Flags          uint32
	InnerMapFd     uint32     // since 4.12 56f668dfe00d
	NumaNode       uint32     // since 4.14 96eabe7a40aa
	MapName        BPFObjName // since 4.15 ad5b177bd73f
	MapIfIndex     uint32
	BTFFd          uint32
	BTFKeyTypeID   uint32
	BTFValueTypeID uint32
}

func BPFMapCreate(attr *BPFMapCreateAttr) (*FD, error) {
	fd, err := BPF(BPF_MAP_CREATE, unsafe.Pointer(attr), unsafe.Sizeof(*attr))
	if err != nil {
		return nil, err
	}

	return NewFD(uint32(fd)), nil
}

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

type syscallError struct {
	error
	errno syscall.Errno
}

func SyscallError(err error, errno syscall.Errno) error {
	return &syscallError{err, errno}
}

func (se *syscallError) Is(target error) bool {
	return target == se.error
}

func (se *syscallError) Unwrap() error {
	return se.errno
}
