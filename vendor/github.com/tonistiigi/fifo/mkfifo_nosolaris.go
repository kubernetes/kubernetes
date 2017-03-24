// +build !solaris

package fifo

import "syscall"

func mkfifo(path string, mode uint32) (err error) {
	return syscall.Mkfifo(path, mode)
}
