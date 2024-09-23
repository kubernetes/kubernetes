//go:build !linux

package unix

import (
	"fmt"
	"runtime"
	"syscall"
)

var errNonLinux = fmt.Errorf("unsupported platform %s/%s", runtime.GOOS, runtime.GOARCH)

// Errnos are distinct and non-zero.
const (
	ENOENT syscall.Errno = iota + 1
	EEXIST
	EAGAIN
	ENOSPC
	EINVAL
	EINTR
	EPERM
	ESRCH
	ENODEV
	EBADF
	E2BIG
	EFAULT
	EACCES
	EILSEQ
	EOPNOTSUPP
)

// Constants are distinct to avoid breaking switch statements.
const (
	BPF_F_NO_PREALLOC = iota
	BPF_F_NUMA_NODE
	BPF_F_RDONLY
	BPF_F_WRONLY
	BPF_F_RDONLY_PROG
	BPF_F_WRONLY_PROG
	BPF_F_SLEEPABLE
	BPF_F_MMAPABLE
	BPF_F_INNER_MAP
	BPF_F_KPROBE_MULTI_RETURN
	BPF_F_XDP_HAS_FRAGS
	BPF_OBJ_NAME_LEN
	BPF_TAG_SIZE
	BPF_RINGBUF_BUSY_BIT
	BPF_RINGBUF_DISCARD_BIT
	BPF_RINGBUF_HDR_SZ
	SYS_BPF
	F_DUPFD_CLOEXEC
	EPOLLIN
	EPOLL_CTL_ADD
	EPOLL_CLOEXEC
	O_CLOEXEC
	O_NONBLOCK
	PROT_NONE
	PROT_READ
	PROT_WRITE
	MAP_ANON
	MAP_SHARED
	MAP_PRIVATE
	PERF_ATTR_SIZE_VER1
	PERF_TYPE_SOFTWARE
	PERF_TYPE_TRACEPOINT
	PERF_COUNT_SW_BPF_OUTPUT
	PERF_EVENT_IOC_DISABLE
	PERF_EVENT_IOC_ENABLE
	PERF_EVENT_IOC_SET_BPF
	PerfBitWatermark
	PerfBitWriteBackward
	PERF_SAMPLE_RAW
	PERF_FLAG_FD_CLOEXEC
	RLIM_INFINITY
	RLIMIT_MEMLOCK
	BPF_STATS_RUN_TIME
	PERF_RECORD_LOST
	PERF_RECORD_SAMPLE
	AT_FDCWD
	RENAME_NOREPLACE
	SO_ATTACH_BPF
	SO_DETACH_BPF
	SOL_SOCKET
	SIGPROF
	SIG_BLOCK
	SIG_UNBLOCK
	EM_NONE
	EM_BPF
	BPF_FS_MAGIC
	TRACEFS_MAGIC
	DEBUGFS_MAGIC
)

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

type Stat_t struct {
	Dev     uint64
	Ino     uint64
	Nlink   uint64
	Mode    uint32
	Uid     uint32
	Gid     uint32
	_       int32
	Rdev    uint64
	Size    int64
	Blksize int64
	Blocks  int64
}

type Rlimit struct {
	Cur uint64
	Max uint64
}

type Signal int

type Sigset_t struct {
	Val [4]uint64
}

func Syscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err syscall.Errno) {
	return 0, 0, syscall.ENOTSUP
}

func PthreadSigmask(how int, set, oldset *Sigset_t) error {
	return errNonLinux
}

func FcntlInt(fd uintptr, cmd, arg int) (int, error) {
	return -1, errNonLinux
}

func IoctlSetInt(fd int, req uint, value int) error {
	return errNonLinux
}

func Statfs(path string, buf *Statfs_t) error {
	return errNonLinux
}

func Close(fd int) (err error) {
	return errNonLinux
}

type EpollEvent struct {
	Events uint32
	Fd     int32
	Pad    int32
}

func EpollWait(epfd int, events []EpollEvent, msec int) (n int, err error) {
	return 0, errNonLinux
}

func EpollCtl(epfd int, op int, fd int, event *EpollEvent) (err error) {
	return errNonLinux
}

func Eventfd(initval uint, flags int) (fd int, err error) {
	return 0, errNonLinux
}

func Write(fd int, p []byte) (n int, err error) {
	return 0, errNonLinux
}

func EpollCreate1(flag int) (fd int, err error) {
	return 0, errNonLinux
}

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

func SetNonblock(fd int, nonblocking bool) (err error) {
	return errNonLinux
}

func Mmap(fd int, offset int64, length int, prot int, flags int) (data []byte, err error) {
	return []byte{}, errNonLinux
}

func Munmap(b []byte) (err error) {
	return errNonLinux
}

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

func PerfEventOpen(attr *PerfEventAttr, pid int, cpu int, groupFd int, flags int) (fd int, err error) {
	return 0, errNonLinux
}

type Utsname struct {
	Release [65]byte
	Version [65]byte
}

func Uname(buf *Utsname) (err error) {
	return errNonLinux
}

func Getpid() int {
	return -1
}

func Gettid() int {
	return -1
}

func Tgkill(tgid int, tid int, sig syscall.Signal) (err error) {
	return errNonLinux
}

func BytePtrFromString(s string) (*byte, error) {
	return nil, errNonLinux
}

func ByteSliceToString(s []byte) string {
	return ""
}

func Renameat2(olddirfd int, oldpath string, newdirfd int, newpath string, flags uint) error {
	return errNonLinux
}

func Prlimit(pid, resource int, new, old *Rlimit) error {
	return errNonLinux
}

func Open(path string, mode int, perm uint32) (int, error) {
	return -1, errNonLinux
}

func Fstat(fd int, stat *Stat_t) error {
	return errNonLinux
}

func SetsockoptInt(fd, level, opt, value int) error {
	return errNonLinux
}
