package sys

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"

	"github.com/cilium/ebpf/internal/unix"
)

var ErrClosedFd = unix.EBADF

type FD struct {
	raw int
}

func newFD(value int) *FD {
	if onLeakFD != nil {
		// Attempt to store the caller's stack for the given fd value.
		// Panic if fds contains an existing stack for the fd.
		old, exist := fds.LoadOrStore(value, callersFrames())
		if exist {
			f := old.(*runtime.Frames)
			panic(fmt.Sprintf("found existing stack for fd %d:\n%s", value, FormatFrames(f)))
		}
	}

	fd := &FD{value}
	runtime.SetFinalizer(fd, (*FD).finalize)
	return fd
}

// finalize is set as the FD's runtime finalizer and
// sends a leak trace before calling FD.Close().
func (fd *FD) finalize() {
	if fd.raw < 0 {
		return
	}

	// Invoke the fd leak callback. Calls LoadAndDelete to guarantee the callback
	// is invoked at most once for one sys.FD allocation, runtime.Frames can only
	// be unwound once.
	f, ok := fds.LoadAndDelete(fd.Int())
	if ok && onLeakFD != nil {
		onLeakFD(f.(*runtime.Frames))
	}

	_ = fd.Close()
}

// NewFD wraps a raw fd with a finalizer.
//
// You must not use the raw fd after calling this function, since the underlying
// file descriptor number may change. This is because the BPF UAPI assumes that
// zero is not a valid fd value.
func NewFD(value int) (*FD, error) {
	if value < 0 {
		return nil, fmt.Errorf("invalid fd %d", value)
	}

	fd := newFD(value)
	if value != 0 {
		return fd, nil
	}

	dup, err := fd.Dup()
	_ = fd.Close()
	return dup, err
}

func (fd *FD) String() string {
	return strconv.FormatInt(int64(fd.raw), 10)
}

func (fd *FD) Int() int {
	return fd.raw
}

func (fd *FD) Uint() uint32 {
	if fd.raw < 0 || int64(fd.raw) > math.MaxUint32 {
		// Best effort: this is the number most likely to be an invalid file
		// descriptor. It is equal to -1 (on two's complement arches).
		return math.MaxUint32
	}
	return uint32(fd.raw)
}

func (fd *FD) Close() error {
	if fd.raw < 0 {
		return nil
	}

	return unix.Close(fd.disown())
}

func (fd *FD) disown() int {
	value := int(fd.raw)
	fds.Delete(int(value))
	fd.raw = -1

	runtime.SetFinalizer(fd, nil)
	return value
}

func (fd *FD) Dup() (*FD, error) {
	if fd.raw < 0 {
		return nil, ErrClosedFd
	}

	// Always require the fd to be larger than zero: the BPF API treats the value
	// as "no argument provided".
	dup, err := unix.FcntlInt(uintptr(fd.raw), unix.F_DUPFD_CLOEXEC, 1)
	if err != nil {
		return nil, fmt.Errorf("can't dup fd: %v", err)
	}

	return newFD(dup), nil
}

// File takes ownership of FD and turns it into an [*os.File].
//
// You must not use the FD after the call returns.
//
// Returns nil if the FD is not valid.
func (fd *FD) File(name string) *os.File {
	if fd.raw < 0 {
		return nil
	}

	return os.NewFile(uintptr(fd.disown()), name)
}
