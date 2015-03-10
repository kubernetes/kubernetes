// +build linux

package mount

import "syscall"

func RemountProc() error {
	if err := syscall.Unmount("/proc", syscall.MNT_DETACH); err != nil {
		return err
	}

	if err := syscall.Mount("proc", "/proc", "proc", uintptr(defaultMountFlags), ""); err != nil {
		return err
	}

	return nil
}

func RemountSys() error {
	if err := syscall.Unmount("/sys", syscall.MNT_DETACH); err != nil {
		if err != syscall.EINVAL {
			return err
		}
	} else {
		if err := syscall.Mount("sysfs", "/sys", "sysfs", uintptr(defaultMountFlags), ""); err != nil {
			return err
		}
	}

	return nil
}
