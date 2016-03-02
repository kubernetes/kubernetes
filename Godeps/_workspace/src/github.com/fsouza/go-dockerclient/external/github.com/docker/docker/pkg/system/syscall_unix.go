// +build linux freebsd

package system

import "syscall"

// Unmount is a platform-specific helper function to call
// the unmount syscall.
func Unmount(dest string) error {
	return syscall.Unmount(dest, 0)
}
