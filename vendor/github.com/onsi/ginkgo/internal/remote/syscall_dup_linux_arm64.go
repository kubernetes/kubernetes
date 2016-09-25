// +build linux,arm64

package remote

import "syscall"

// linux_arm64 doesn't have syscall.Dup2 which ginkgo uses, so
// use the nearly identical syscall.Dup3 instead
func syscallDup(oldfd int, newfd int) (err error) {
	return syscall.Dup3(oldfd, newfd, 0)
}
