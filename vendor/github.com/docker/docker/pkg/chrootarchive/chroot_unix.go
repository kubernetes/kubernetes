// +build !windows,!linux

package chrootarchive

import "golang.org/x/sys/unix"

func chroot(path string) error {
	if err := unix.Chroot(path); err != nil {
		return err
	}
	return unix.Chdir("/")
}
