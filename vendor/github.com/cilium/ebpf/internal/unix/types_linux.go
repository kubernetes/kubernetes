// +build linux

package unix

import (
	"syscall"

	linux "golang.org/x/sys/unix"
)

const (
	ENOENT                   = linux.ENOENT
	EEXIST                   = linux.EEXIST
	EAGAIN                   = linux.EAGAIN
	ENOSPC                   = linux.ENOSPC
	EINVAL                   = linux.EINVAL
	EPOLLIN                  = linux.EPOLLIN
	EINTR                    = linux.EINTR
	EPERM                    = linux.EPERM
	ESRCH                    = linux.ESRCH
	ENODEV                   = linux.ENODEV
	BPF_F_RDONLY_PROG        = linux.BPF_F_RDONLY_PROG
	BPF_F_WRONLY_PROG        = linux.BPF_F_WRONLY_PROG
	BPF_OBJ_NAME_LEN         = linux.BPF_OBJ_NAME_LEN
	BPF_TAG_SIZE             = linux.BPF_TAG_SIZE
	SYS_BPF                  = linux.SYS_BPF
	F_DUPFD_CLOEXEC          = linux.F_DUPFD_CLOEXEC
	EPOLL_CTL_ADD            = linux.EPOLL_CTL_ADD
	EPOLL_CLOEXEC            = linux.EPOLL_CLOEXEC
	O_CLOEXEC                = linux.O_CLOEXEC
	O_NONBLOCK               = linux.O_NONBLOCK
	PROT_READ                = linux.PROT_READ
	PROT_WRITE               = linux.PROT_WRITE
	MAP_SHARED               = linux.MAP_SHARED
	PERF_TYPE_SOFTWARE       = linux.PERF_TYPE_SOFTWARE
	PERF_COUNT_SW_BPF_OUTPUT = linux.PERF_COUNT_SW_BPF_OUTPUT
	PerfBitWatermark         = linux.PerfBitWatermark
	PERF_SAMPLE_RAW          = linux.PERF_SAMPLE_RAW
	PERF_FLAG_FD_CLOEXEC     = linux.PERF_FLAG_FD_CLOEXEC
	RLIM_INFINITY            = linux.RLIM_INFINITY
	RLIMIT_MEMLOCK           = linux.RLIMIT_MEMLOCK
)

// Statfs_t is a wrapper
type Statfs_t = linux.Statfs_t

// Rlimit is a wrapper
type Rlimit = linux.Rlimit

// Setrlimit is a wrapper
func Setrlimit(resource int, rlim *Rlimit) (err error) {
	return linux.Setrlimit(resource, rlim)
}

// Syscall is a wrapper
func Syscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err syscall.Errno) {
	return linux.Syscall(trap, a1, a2, a3)
}

// FcntlInt is a wrapper
func FcntlInt(fd uintptr, cmd, arg int) (int, error) {
	return linux.FcntlInt(fd, cmd, arg)
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
