// +build linux freebsd

package system

import "golang.org/x/sys/unix"

// Unmount is a platform-specific helper function to call
// the unmount syscall.
func Unmount(dest string) error {
	return unix.Unmount(dest, 0)
}

// CommandLineToArgv should not be used on Unix.
// It simply returns commandLine in the only element in the returned array.
func CommandLineToArgv(commandLine string) ([]string, error) {
	return []string{commandLine}, nil
}
