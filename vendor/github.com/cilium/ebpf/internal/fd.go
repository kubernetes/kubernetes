package internal

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"strconv"

	"github.com/cilium/ebpf/internal/unix"
)

var ErrClosedFd = errors.New("use of closed file descriptor")

type FD struct {
	raw int64
}

func NewFD(value uint32) *FD {
	fd := &FD{int64(value)}
	runtime.SetFinalizer(fd, (*FD).Close)
	return fd
}

func (fd *FD) String() string {
	return strconv.FormatInt(fd.raw, 10)
}

func (fd *FD) Value() (uint32, error) {
	if fd.raw < 0 {
		return 0, ErrClosedFd
	}

	return uint32(fd.raw), nil
}

func (fd *FD) Close() error {
	if fd.raw < 0 {
		return nil
	}

	value := int(fd.raw)
	fd.raw = -1

	fd.Forget()
	return unix.Close(value)
}

func (fd *FD) Forget() {
	runtime.SetFinalizer(fd, nil)
}

func (fd *FD) Dup() (*FD, error) {
	if fd.raw < 0 {
		return nil, ErrClosedFd
	}

	dup, err := unix.FcntlInt(uintptr(fd.raw), unix.F_DUPFD_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("can't dup fd: %v", err)
	}

	return NewFD(uint32(dup)), nil
}

func (fd *FD) File(name string) *os.File {
	fd.Forget()
	return os.NewFile(uintptr(fd.raw), name)
}
