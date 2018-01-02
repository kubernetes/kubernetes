// +build !windows,!plan9,!solaris

package tsm1

import (
	"os"
	"syscall"
	"unsafe"
)

func mmap(f *os.File, offset int64, length int) ([]byte, error) {
	mmap, err := syscall.Mmap(int(f.Fd()), 0, length, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, err
	}

	return mmap, nil
}

func munmap(b []byte) (err error) {
	return syscall.Munmap(b)
}

// From: github.com/boltdb/bolt/bolt_unix.go
func madvise(b []byte, advice int) (err error) {
	_, _, e1 := syscall.Syscall(syscall.SYS_MADVISE, uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), uintptr(advice))
	if e1 != 0 {
		err = e1
	}
	return
}
