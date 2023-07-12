//go:build linux

package unix

import (
	"syscall"

	linux "golang.org/x/sys/unix"
)

const (
	ENOENT     = linux.ENOENT
	EEXIST     = linux.EEXIST
	EAGAIN     = linux.EAGAIN
	ENOSPC     = linux.ENOSPC
	EINVAL     = linux.EINVAL
	EPOLLIN    = linux.EPOLLIN
	EINTR      = linux.EINTR
	EPERM      = linux.EPERM
	ESRCH      = linux.ESRCH
	ENODEV     = linux.ENODEV
	EBADF      = linux.EBADF
	E2BIG      = linux.E2BIG
	EFAULT     = linux.EFAULT
	EACCES     = linux.EACCES
	EILSEQ     = linux.EILSEQ
	EOPNOTSUPP = linux.EOPNOTSUPP
)

const (
	BPF_F_NO_PREALLOC         = linux.BPF_F_NO_PREALLOC
	BPF_F_NUMA_NODE           = linux.BPF_F_NUMA_NODE
	BPF_F_RDONLY              = linux.BPF_F_RDONLY
	BPF_F_WRONLY              = linux.BPF_F_WRONLY
	BPF_F_RDONLY_PROG         = linux.BPF_F_RDONLY_PROG
	BPF_F_WRONLY_PROG         = linux.BPF_F_WRONLY_PROG
	BPF_F_SLEEPABLE           = linux.BPF_F_SLEEPABLE
	BPF_F_XDP_HAS_FRAGS       = linux.BPF_F_XDP_HAS_FRAGS
	BPF_F_MMAPABLE            = linux.BPF_F_MMAPABLE
	BPF_F_INNER_MAP           = linux.BPF_F_INNER_MAP
	BPF_F_KPROBE_MULTI_RETURN = linux.BPF_F_KPROBE_MULTI_RETURN
	BPF_OBJ_NAME_LEN          = linux.BPF_OBJ_NAME_LEN
	BPF_TAG_SIZE              = linux.BPF_TAG_SIZE
	BPF_RINGBUF_BUSY_BIT      = linux.BPF_RINGBUF_BUSY_BIT
	BPF_RINGBUF_DISCARD_BIT   = linux.BPF_RINGBUF_DISCARD_BIT
	BPF_RINGBUF_HDR_SZ        = linux.BPF_RINGBUF_HDR_SZ
	SYS_BPF                   = linux.SYS_BPF
	F_DUPFD_CLOEXEC           = linux.F_DUPFD_CLOEXEC
	EPOLL_CTL_ADD             = linux.EPOLL_CTL_ADD
	EPOLL_CLOEXEC             = linux.EPOLL_CLOEXEC
	O_CLOEXEC                 = linux.O_CLOEXEC
	O_NONBLOCK                = linux.O_NONBLOCK
	PROT_NONE                 = linux.PROT_NONE
	PROT_READ                 = linux.PROT_READ
	PROT_WRITE                = linux.PROT_WRITE
	MAP_ANON                  = linux.MAP_ANON
	MAP_SHARED                = linux.MAP_SHARED
	MAP_PRIVATE               = linux.MAP_PRIVATE
	PERF_ATTR_SIZE_VER1       = linux.PERF_ATTR_SIZE_VER1
	PERF_TYPE_SOFTWARE        = linux.PERF_TYPE_SOFTWARE
	PERF_TYPE_TRACEPOINT      = linux.PERF_TYPE_TRACEPOINT
	PERF_COUNT_SW_BPF_OUTPUT  = linux.PERF_COUNT_SW_BPF_OUTPUT
	PERF_EVENT_IOC_DISABLE    = linux.PERF_EVENT_IOC_DISABLE
	PERF_EVENT_IOC_ENABLE     = linux.PERF_EVENT_IOC_ENABLE
	PERF_EVENT_IOC_SET_BPF    = linux.PERF_EVENT_IOC_SET_BPF
	PerfBitWatermark          = linux.PerfBitWatermark
	PerfBitWriteBackward      = linux.PerfBitWriteBackward
	PERF_SAMPLE_RAW           = linux.PERF_SAMPLE_RAW
	PERF_FLAG_FD_CLOEXEC      = linux.PERF_FLAG_FD_CLOEXEC
	RLIM_INFINITY             = linux.RLIM_INFINITY
	RLIMIT_MEMLOCK            = linux.RLIMIT_MEMLOCK
	BPF_STATS_RUN_TIME        = linux.BPF_STATS_RUN_TIME
	PERF_RECORD_LOST          = linux.PERF_RECORD_LOST
	PERF_RECORD_SAMPLE        = linux.PERF_RECORD_SAMPLE
	AT_FDCWD                  = linux.AT_FDCWD
	RENAME_NOREPLACE          = linux.RENAME_NOREPLACE
	SO_ATTACH_BPF             = linux.SO_ATTACH_BPF
	SO_DETACH_BPF             = linux.SO_DETACH_BPF
	SOL_SOCKET                = linux.SOL_SOCKET
	SIGPROF                   = linux.SIGPROF
	SIG_BLOCK                 = linux.SIG_BLOCK
	SIG_UNBLOCK               = linux.SIG_UNBLOCK
	EM_NONE                   = linux.EM_NONE
	EM_BPF                    = linux.EM_BPF
	BPF_FS_MAGIC              = linux.BPF_FS_MAGIC
	TRACEFS_MAGIC             = linux.TRACEFS_MAGIC
	DEBUGFS_MAGIC             = linux.DEBUGFS_MAGIC
)

type Statfs_t = linux.Statfs_t
type Stat_t = linux.Stat_t
type Rlimit = linux.Rlimit
type Signal = linux.Signal
type Sigset_t = linux.Sigset_t
type PerfEventMmapPage = linux.PerfEventMmapPage
type EpollEvent = linux.EpollEvent
type PerfEventAttr = linux.PerfEventAttr
type Utsname = linux.Utsname

func Syscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err syscall.Errno) {
	return linux.Syscall(trap, a1, a2, a3)
}

func PthreadSigmask(how int, set, oldset *Sigset_t) error {
	return linux.PthreadSigmask(how, set, oldset)
}

func FcntlInt(fd uintptr, cmd, arg int) (int, error) {
	return linux.FcntlInt(fd, cmd, arg)
}

func IoctlSetInt(fd int, req uint, value int) error {
	return linux.IoctlSetInt(fd, req, value)
}

func Statfs(path string, buf *Statfs_t) (err error) {
	return linux.Statfs(path, buf)
}

func Close(fd int) (err error) {
	return linux.Close(fd)
}

func EpollWait(epfd int, events []EpollEvent, msec int) (n int, err error) {
	return linux.EpollWait(epfd, events, msec)
}

func EpollCtl(epfd int, op int, fd int, event *EpollEvent) (err error) {
	return linux.EpollCtl(epfd, op, fd, event)
}

func Eventfd(initval uint, flags int) (fd int, err error) {
	return linux.Eventfd(initval, flags)
}

func Write(fd int, p []byte) (n int, err error) {
	return linux.Write(fd, p)
}

func EpollCreate1(flag int) (fd int, err error) {
	return linux.EpollCreate1(flag)
}

func SetNonblock(fd int, nonblocking bool) (err error) {
	return linux.SetNonblock(fd, nonblocking)
}

func Mmap(fd int, offset int64, length int, prot int, flags int) (data []byte, err error) {
	return linux.Mmap(fd, offset, length, prot, flags)
}

func Munmap(b []byte) (err error) {
	return linux.Munmap(b)
}

func PerfEventOpen(attr *PerfEventAttr, pid int, cpu int, groupFd int, flags int) (fd int, err error) {
	return linux.PerfEventOpen(attr, pid, cpu, groupFd, flags)
}

func Uname(buf *Utsname) (err error) {
	return linux.Uname(buf)
}

func Getpid() int {
	return linux.Getpid()
}

func Gettid() int {
	return linux.Gettid()
}

func Tgkill(tgid int, tid int, sig syscall.Signal) (err error) {
	return linux.Tgkill(tgid, tid, sig)
}

func BytePtrFromString(s string) (*byte, error) {
	return linux.BytePtrFromString(s)
}

func ByteSliceToString(s []byte) string {
	return linux.ByteSliceToString(s)
}

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

func SetsockoptInt(fd, level, opt, value int) error {
	return linux.SetsockoptInt(fd, level, opt, value)
}
