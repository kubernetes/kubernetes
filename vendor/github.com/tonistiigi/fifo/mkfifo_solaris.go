// +build solaris

package fifo

import (
	"golang.org/x/sys/unix"
)

func mkfifo(path string, mode uint32) (err error) {
	return unix.Mkfifo(path, mode)
}
