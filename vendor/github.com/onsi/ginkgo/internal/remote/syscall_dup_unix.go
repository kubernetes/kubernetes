// +build !linux !arm64
// +build !windows

package remote

import "syscall"

func syscallDup(oldfd int, newfd int) (err error) {
	return syscall.Dup2(oldfd, newfd)
}