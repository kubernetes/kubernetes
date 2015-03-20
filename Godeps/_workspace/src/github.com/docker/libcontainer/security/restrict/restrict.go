// +build linux

package restrict

import (
	"fmt"
	"os"
	"syscall"
	"time"
)

const defaultMountFlags = syscall.MS_NOEXEC | syscall.MS_NOSUID | syscall.MS_NODEV

func mountReadonly(path string) error {
	for i := 0; i < 5; i++ {
		if err := syscall.Mount("", path, "", syscall.MS_REMOUNT|syscall.MS_RDONLY, ""); err != nil && !os.IsNotExist(err) {
			switch err {
			case syscall.EINVAL:
				// Probably not a mountpoint, use bind-mount
				if err := syscall.Mount(path, path, "", syscall.MS_BIND, ""); err != nil {
					return err
				}

				return syscall.Mount(path, path, "", syscall.MS_BIND|syscall.MS_REMOUNT|syscall.MS_RDONLY|syscall.MS_REC|defaultMountFlags, "")
			case syscall.EBUSY:
				time.Sleep(100 * time.Millisecond)
				continue
			default:
				return err
			}
		}

		return nil
	}

	return fmt.Errorf("unable to mount %s as readonly max retries reached", path)
}

// This has to be called while the container still has CAP_SYS_ADMIN (to be able to perform mounts).
// However, afterwards, CAP_SYS_ADMIN should be dropped (otherwise the user will be able to revert those changes).
func Restrict(mounts ...string) error {
	for _, dest := range mounts {
		if err := mountReadonly(dest); err != nil {
			return fmt.Errorf("unable to remount %s readonly: %s", dest, err)
		}
	}

	if err := syscall.Mount("/dev/null", "/proc/kcore", "", syscall.MS_BIND, ""); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("unable to bind-mount /dev/null over /proc/kcore: %s", err)
	}

	return nil
}
