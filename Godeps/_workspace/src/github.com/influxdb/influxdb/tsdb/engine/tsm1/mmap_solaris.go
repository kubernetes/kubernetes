// +build solaris

package tsm1

import (
	"os"
	"syscall"

	"golang.org/x/sys/unix"
)

func mmap(f *os.File, offset int64, length int) ([]byte, error) {
	mmap, err := unix.Mmap(int(f.Fd()), 0, length, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, err
	}

	if err := unix.Madvise(mmap, syscall.MADV_RANDOM); err != nil {
		return nil, err
	}

	return mmap, nil
}

func munmap(b []byte) (err error) {
	return unix.Munmap(b)
}

// From: github.com/boltdb/bolt/bolt_unix.go
func madvise(b []byte, advice int) (err error) {
	return unix.Madvise(b, advice)
}
