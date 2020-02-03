// +build linux,!appengine

package msgp

import (
	"os"
	"syscall"
)

func adviseRead(mem []byte) {
	syscall.Madvise(mem, syscall.MADV_SEQUENTIAL|syscall.MADV_WILLNEED)
}

func adviseWrite(mem []byte) {
	syscall.Madvise(mem, syscall.MADV_SEQUENTIAL)
}

func fallocate(f *os.File, sz int64) error {
	err := syscall.Fallocate(int(f.Fd()), 0, 0, sz)
	if err == syscall.ENOTSUP {
		return f.Truncate(sz)
	}
	return err
}
