// +build linux

package mount

import (
	"fmt"
	"syscall"
)

func MsMoveRoot(rootfs string) error {
	if err := syscall.Mount(rootfs, "/", "", syscall.MS_MOVE, ""); err != nil {
		return fmt.Errorf("mount move %s into / %s", rootfs, err)
	}

	if err := syscall.Chroot("."); err != nil {
		return fmt.Errorf("chroot . %s", err)
	}

	return syscall.Chdir("/")
}
