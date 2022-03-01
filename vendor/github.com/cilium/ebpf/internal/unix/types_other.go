// +build !linux

package unix

import (
	"fmt"
	"runtime"
	"syscall"
)

var errNonLinux = fmt.Errorf("unsupported platform %s/%s", runtime.GOOS, runtime.GOARCH)

const (
	ENOENT = syscall.ENOENT
	EEXIST = syscall.EEXIST
	EAGAIN = syscall.EAGAIN
	ENOSPC = syscall.ENOSPC
	EINVAL = syscall.EINVAL
	EINTR  = syscall.EINTR
	EPERM  = syscall.EPERM
	ESRCH  = syscall.ESRCH
	ENODEV = syscall.ENODEV
	EBADF  = syscall.Errno(0)
	// ENOTSUPP is not the same as ENOTSUP or EOPNOTSUP
	ENOTSUPP = syscall.Errno(0x20c)

	BPF_F_NO_PREALLOC        = 0
	BPF_F_NUMA_NODE          = 0
	BPF_F_RDONLY             = 0
	BPF_F_WRONLY             = 0
	BPF_F_RDONLY_PROG        = 0
	BPF_F_WRONLY_PROG        = 0
	BPF_F_SLEEPABLE          = 0
	BPF_F_MMAPABLE           = 0
	BPF_F_INNER_MAP          = 0
	BPF_OBJ_NAME_LEN         = 0x10
	BPF_TAG_SIZE             = 0x8
	SYS_BPF                  = 321
	F_DUPFD_CLOEXEC          = 0x406
	EPOLLIN                  = 0x1
	EPOLL_CTL_ADD            = 0x1
	EPOLL_CLOEXEC            = 0x80000
	O_CLOEXEC                = 0x80000
	O_NONBLOCK               = 0x800
	PROT_READ                = 0x1
	PROT_WRITE               = 0x2
	MAP_SHARED               = 0x1
	PERF_ATTR_SIZE_VER1      = 0
	PERF_TYPE_SOFTWARE       = 0x1
	PERF_TYPE_TRACEPOINT     = 0
	PERF_COUNT_SW_BPF_OUTPUT = 0xa
	PERF_EVENT_IOC_DISABLE   = 0
	PERF_EVENT_IOC_ENABLE    = 0
	PERF_EVENT_IOC_SET_BPF   = 0
	PerfBitWatermark         = 0x4000
	PERF_SAMPLE_RAW          = 0x400
	PERF_FLAG_FD_CLOEXEC     = 0x8
	RLIM_INFINITY            = 0x7fffffffffffffff
	RLIMIT_MEMLOCK           = 8
	BPF_STATS_RUN_TIME       = 0
	PERF_RECORD_LOST         = 2
	PERF_RECORD_SAMPLE       = 9
	AT_FDCWD                 = -0x2
	RENAME_NOREPLACE         = 0x1
)

// Statfs_t is a wrapper
type Statfs_t struct {
	Type    int64
	Bsize   int64
	Blocks  uint64
	Bfree   uint64
	Bavail  uint64
	Files   uint64
	Ffree   uint64
	Fsid    [2]int32
	Namelen int64
	Frsize  int64
	Flags   int64
	Spare   [4]int64
}

// Rlimit is a wrapper
type Rlimit struct {
	Cur uint64
	Max uint64
}

// Setrlimit is a wrapper
func Setrlimit(resource int, rlim *Rlimit) (err error) {
	return errNonLinux
}

// Syscall is a wrapper
func Syscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err syscall.Errno) {
	return 0, 0, syscall.Errno(1)
}

// FcntlInt is a wrapper
func FcntlInt(fd uintptr, cmd, arg int) (int, error) {
	return -1, errNonLinux
}

// IoctlSetInt is a wrapper
func IoctlSetInt(fd int, req uint, value int) error {
	return errNonLinux
}

// Statfs is a wrapper
func Statfs(path string, buf *Statfs_t) error {
	return errNonLinux
}

// Close is a wrapper
func Close(fd int) (err error) {
	return errNonLinux
}

// EpollEvent is a wrapper
type EpollEvent struct {
	Events uint32
	Fd     int32
	Pad    int32
}

// EpollWait is a wrapper
func EpollWait(epfd int, events []EpollEvent, msec int) (n int, err error) {
	return 0, errNonLinux
}

// EpollCtl is a wrapper
func EpollCtl(epfd int, op int, fd int, event *EpollEvent) (err error) {
	return errNonLinux
}

// Eventfd is a wrapper
func Eventfd(initval uint, flags int) (fd int, err error) {
	return 0, errNonLinux
}

// Write is a wrapper
func Write(fd int, p []byte) (n int, err error) {
	return 0, errNonLinux
}

// EpollCreate1 is a wrapper
func EpollCreate1(flag int) (fd int, err error) {
	return 0, errNonLinux
}

// PerfEventMmapPage is a wrapper
type PerfEventMmapPage struct {
	Version        uint32
	Compat_version uint32
	Lock           uint32
	Index          uint32
	Offset         int64
	Time_enabled   uint64
	Time_running   uint64
	Capabilities   uint64
	Pmc_width      uint16
	Time_shift     uint16
	Time_mult      uint32
	Time_offset    uint64
	Time_zero      uint64
	Size           uint32

	Data_head   uint64
	Data_tail   uint64
	Data_offset uint64
	Data_size   uint64
	Aux_head    uint64
	Aux_tail    uint64
	Aux_offset  uint64
	Aux_size    uint64
}

// SetNonblock is a wrapper
func SetNonblock(fd int, nonblocking bool) (err error) {
	return errNonLinux
}

// Mmap is a wrapper
func Mmap(fd int, offset int64, length int, prot int, flags int) (data []byte, err error) {
	return []byte{}, errNonLinux
}

// Munmap is a wrapper
func Munmap(b []byte) (err error) {
	return errNonLinux
}

// PerfEventAttr is a wrapper
type PerfEventAttr struct {
	Type               uint32
	Size               uint32
	Config             uint64
	Sample             uint64
	Sample_type        uint64
	Read_format        uint64
	Bits               uint64
	Wakeup             uint32
	Bp_type            uint32
	Ext1               uint64
	Ext2               uint64
	Branch_sample_type uint64
	Sample_regs_user   uint64
	Sample_stack_user  uint32
	Clockid            int32
	Sample_regs_intr   uint64
	Aux_watermark      uint32
	Sample_max_stack   uint16
}

// PerfEventOpen is a wrapper
func PerfEventOpen(attr *PerfEventAttr, pid int, cpu int, groupFd int, flags int) (fd int, err error) {
	return 0, errNonLinux
}

// Utsname is a wrapper
type Utsname struct {
	Release [65]byte
	Version [65]byte
}

// Uname is a wrapper
func Uname(buf *Utsname) (err error) {
	return errNonLinux
}

// Getpid is a wrapper
func Getpid() int {
	return -1
}

// Gettid is a wrapper
func Gettid() int {
	return -1
}

// Tgkill is a wrapper
func Tgkill(tgid int, tid int, sig syscall.Signal) (err error) {
	return errNonLinux
}

// BytePtrFromString is a wrapper
func BytePtrFromString(s string) (*byte, error) {
	return nil, errNonLinux
}

// ByteSliceToString is a wrapper
func ByteSliceToString(s []byte) string {
	return ""
}

// Renameat2 is a wrapper
func Renameat2(olddirfd int, oldpath string, newdirfd int, newpath string, flags uint) error {
	return errNonLinux
}

func KernelRelease() (string, error) {
	return "", errNonLinux
}
