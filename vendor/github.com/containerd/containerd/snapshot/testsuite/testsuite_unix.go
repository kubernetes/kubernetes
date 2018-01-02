// +build !windows

package testsuite

import "syscall"

func clearMask() func() {
	oldumask := syscall.Umask(0)
	return func() {
		syscall.Umask(oldumask)
	}
}
