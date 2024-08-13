//go:build linux
// +build linux

package unix

import (
	"syscall"

	linux "golang.org/x/sys/unix"
)

const (
	ENOENT  = linux.ENOENT
	EEXIST  = linux.EEXIST
	EAGAIN  = linux.EAGAIN
	ENOSPC  = linux.ENOSPC
	EINVAL  = linux.EINVAL
	EPOLLIN = linux.EPOLLIN
	EINTR   = linux.EINTR
	EPERM   = linux.EPERM
	ESRCH   = linux.ESRCH
	ENODEV  = linux.ENODEV
	EBADF   = linux.EBADF
	E2BIG   = linux.E2BIG
	EFAULT  = linux.EFAULT
	EACCES  = linux.EACCES
	// ENOTSUPP is not the same as ENOTSUP or EOPNOTSUP
	ENOTSUPP = syscall.Errno(0x20c)

	BPF_F_NO_PREALLOC        = linux.BPF_F_NO_PREALLOC
	BPF_F_NUMA_NODE          = linux.BPF_F_NUMA_NODE
	BPF_F_RDONLY             = linux.BPF_F_RDONLY
	BPF_F_WRONLY             = linux.BPF_F_WRONLY
	BPF_F_RDONLY_PROG        = linux.BPF_F_RDONLY_PROG
	BPF_F_WRONLY_PROG        = linux.BPF_F_WRONLY_PROG
	BPF_F_SLEEPABLE          = linux.BPF_F_SLEEPABLE
	BPF_F_MMAPABLE           = linux.BPF_F_MMAPABLE
	BPF_F_INNER_MAP          = linux.BPF_F_INNER_MAP
	BPF_OBJ_NAME_LEN         = linux.BPF_OBJ_NAME_LEN
	BPF_TAG_SIZE             = linux.BPF_TAG_SIZE
	BPF_RINGBUF_BUSY_BIT     = linux.BPF_RINGBUF_BUSY_BIT
	BPF_RINGBUF_DISCARD_BIT  = linux.BPF_RINGBUF_DISCARD_BIT
	BPF_RINGBUF_HDR_SZ       = linux.BPF_RINGBUF_HDR_SZ
	SYS_BPF                  = linux.SYS_BPF
	F_DUPFD_CLOEXEC          = linux.F_DUPFD_CLOEXEC
	EPOLL_CTL_ADD            = linux.EPOLL_CTL_ADD
	EPOLL_CLOEXEC            = linux.EPOLL_CLOEXEC
	O_CLOEXEC                = linux.O_CLOEXEC
	O_NONBLOCK               = linux.O_NONBLOCK
	PROT_READ                = linux.PROT_READ
	PROT_WRITE               = linux.PROT_WRITE
	MAP_SHARED               = linux.MAP_SHARED
	PERF_ATTR_SIZE_VER1      = linux.PERF_ATTR_SIZE_VER1
	PERF_TYPE_SOFTWARE       = linux.PERF_TYPE_SOFTWARE
	PERF_TYPE_TRACEPOINT     = linux.PERF_TYPE_TRACEPOINT
	PERF_COUNT_SW_BPF_OUTPUT = linux.PERF_COUNT_SW_BPF_OUTPUT
	PERF_EVENT_IOC_DISABLE   = linux.PERF_EVENT_IOC_DISABLE
	PERF_EVENT_IOC_ENABLE    = linux.PERF_EVENT_IOC_ENABLE
	PERF_EVENT_IOC_SET_BPF   = linux.PERF_EVENT_IOC_SET_BPF
	PerfBitWatermark         = linux.PerfBitWatermark
	PERF_SAMPLE_RAW          = linux.PERF_SAMPLE_RAW
	PERF_FLAG_FD_CLOEXEC     = linux.PERF_FLAG_FD_CLOEXEC
	RLIM_INFINITY            = linux.RLIM_INFINITY
	RLIMIT_MEMLOCK           = linux.RLIMIT_MEMLOCK
	BPF_STATS_RUN_TIME       = linux.BPF_STATS_RUN_TIME
	PERF_RECORD_LOST         = linux.PERF_RECORD_LOST
	PERF_RECORD_SAMPLE       = linux.PERF_RECORD_SAMPLE
	AT_FDCWD                 = linux.AT_FDCWD
	RENAME_NOREPLACE         = linux.RENAME_NOREPLACE
	SO_ATTACH_BPF            = linux.SO_ATTACH_BPF
	SO_DETACH_BPF            = linux.SO_DETACH_BPF
	SOL_SOCKET               = linux.SOL_SOCKET
)

// Statfs_t is a wrapper
type Statfs_t = linux.Statfs_t

type Stat_t = linux.Stat_t

// Rlimit is a wrapper
type Rlimit = linux.Rlimit

// Syscall is a wrapper
func Syscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err syscall.Errno) {
	return linux.Syscall(trap, a1, a2, a3)
}

// FcntlInt is a wrapper
func FcntlInt(fd uintptr, cmd, arg int) (int, error) {
	return linux.FcntlInt(fd, cmd, arg)
}

// IoctlSetInt is a wrapper
func IoctlSetInt(fd int, req uint, value int) error {
	return linux.IoctlSetInt(fd, req, value)
}

// Statfs is a wrapper
func Statfs(path string, buf *Statfs_t) (err error) {
	return linux.Statfs(path, buf)
}

// Close is a wrapper
func Close(fd int) (err error) {
	return linux.Close(fd)
}

// EpollEvent is a wrapper
type EpollEvent = linux.EpollEvent

// EpollWait is a wrapper
func EpollWait(epfd int, events []EpollEvent, msec int) (n int, err error) {
	return linux.EpollWait(epfd, events, msec)
}

// EpollCtl is a wrapper
func EpollCtl(epfd int, op int, fd int, event *EpollEvent) (err error) {
	return linux.EpollCtl(epfd, op, fd, event)
}

// Eventfd is a wrapper
func Eventfd(initval uint, flags int) (fd int, err error) {
	return linux.Eventfd(initval, flags)
}

// Write is a wrapper
func Write(fd int, p []byte) (n int, err error) {
	return linux.Write(fd, p)
}

// EpollCreate1 is a wrapper
func EpollCreate1(flag int) (fd int, err error) {
	return linux.EpollCreate1(flag)
}

// PerfEventMmapPage is a wrapper
type PerfEventMmapPage linux.PerfEventMmapPage

// SetNonblock is a wrapper
func SetNonblock(fd int, nonblocking bool) (err error) {
	return linux.SetNonblock(fd, nonblocking)
}

// Mmap is a wrapper
func Mmap(fd int, offset int64, length int, prot int, flags int) (data []byte, err error) {
	return linux.Mmap(fd, offset, length, prot, flags)
}

// Munmap is a wrapper
func Munmap(b []byte) (err error) {
	return linux.Munmap(b)
}

// PerfEventAttr is a wrapper
type PerfEventAttr = linux.PerfEventAttr

// PerfEventOpen is a wrapper
func PerfEventOpen(attr *PerfEventAttr, pid int, cpu int, groupFd int, flags int) (fd int, err error) {
	return linux.PerfEventOpen(attr, pid, cpu, groupFd, flags)
}

// Utsname is a wrapper
type Utsname = linux.Utsname

// Uname is a wrapper
func Uname(buf *Utsname) (err error) {
	return linux.Uname(buf)
}

// Getpid is a wrapper
func Getpid() int {
	return linux.Getpid()
}

// Gettid is a wrapper
func Gettid() int {
	return linux.Gettid()
}

// Tgkill is a wrapper
func Tgkill(tgid int, tid int, sig syscall.Signal) (err error) {
	return linux.Tgkill(tgid, tid, sig)
}

// BytePtrFromString is a wrapper
func BytePtrFromString(s string) (*byte, error) {
	return linux.BytePtrFromString(s)
}

// ByteSliceToString is a wrapper
func ByteSliceToString(s []byte) string {
	return linux.ByteSliceToString(s)
}

// Renameat2 is a wrapper
func Renameat2(olddirfd int, oldpath string, newdirfd int, newpath string, flags uint) error {
	return linux.Renameat2(olddirfd, oldpath, newdirfd, newpath, flags)
}

func Prlimit(pid, resource int, new, old *Rlimit) error {
	return linux.Prlimit(pid, resource, new, old)
}

func Open(path string, mode int, perm uint32) (int, error) {
	return linux.Open(path, mode, perm)
}

func Fstat(fd int, stat *Stat_t) error {
	return linux.Fstat(fd, stat)
}
