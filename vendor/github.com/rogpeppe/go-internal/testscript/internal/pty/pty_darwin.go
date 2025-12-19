package pty

import (
	"bytes"
	"os"
	"syscall"
	"unsafe"
)

func ptyName(f *os.File) (string, error) {
	// Parameter length is encoded in the low 13 bits of the top word.
	// See https://github.com/apple/darwin-xnu/blob/2ff845c2e0/bsd/sys/ioccom.h#L69-L77
	const IOCPARM_MASK = 0x1fff
	const TIOCPTYGNAME_PARM_LEN = (syscall.TIOCPTYGNAME >> 16) & IOCPARM_MASK
	out := make([]byte, TIOCPTYGNAME_PARM_LEN)

	err := ioctl(f, "TIOCPTYGNAME", syscall.TIOCPTYGNAME, uintptr(unsafe.Pointer(&out[0])))
	if err != nil {
		return "", err
	}

	i := bytes.IndexByte(out, 0x00)
	return string(out[:i]), nil
}

func ptyGrant(f *os.File) error {
	return ioctl(f, "TIOCPTYGRANT", syscall.TIOCPTYGRANT, 0)
}

func ptyUnlock(f *os.File) error {
	return ioctl(f, "TIOCPTYUNLK", syscall.TIOCPTYUNLK, 0)
}
