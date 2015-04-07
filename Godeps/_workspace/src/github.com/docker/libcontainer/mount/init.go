// +build linux

package mount

import (
	"fmt"
	"os"
	"path/filepath"
	"syscall"

	"github.com/docker/libcontainer/label"
	"github.com/docker/libcontainer/mount/nodes"
)

// default mount point flags
const defaultMountFlags = syscall.MS_NOEXEC | syscall.MS_NOSUID | syscall.MS_NODEV

type mount struct {
	source string
	path   string
	device string
	flags  int
	data   string
}

// InitializeMountNamespace sets up the devices, mount points, and filesystems for use inside a
// new mount namespace.
func InitializeMountNamespace(rootfs, console string, sysReadonly bool, mountConfig *MountConfig) error {
	var (
		err  error
		flag = syscall.MS_PRIVATE
	)

	if mountConfig.NoPivotRoot {
		flag = syscall.MS_SLAVE
	}

	if err := syscall.Mount("", "/", "", uintptr(flag|syscall.MS_REC), ""); err != nil {
		return fmt.Errorf("mounting / with flags %X %s", (flag | syscall.MS_REC), err)
	}

	if err := syscall.Mount(rootfs, rootfs, "bind", syscall.MS_BIND|syscall.MS_REC, ""); err != nil {
		return fmt.Errorf("mouting %s as bind %s", rootfs, err)
	}

	if err := mountSystem(rootfs, sysReadonly, mountConfig); err != nil {
		return fmt.Errorf("mount system %s", err)
	}

	// apply any user specified mounts within the new mount namespace
	for _, m := range mountConfig.Mounts {
		if err := m.Mount(rootfs, mountConfig.MountLabel); err != nil {
			return err
		}
	}

	if err := nodes.CreateDeviceNodes(rootfs, mountConfig.DeviceNodes); err != nil {
		return fmt.Errorf("create device nodes %s", err)
	}

	if err := SetupPtmx(rootfs, console, mountConfig.MountLabel); err != nil {
		return err
	}

	// stdin, stdout and stderr could be pointing to /dev/null from parent namespace.
	// Re-open them inside this namespace.
	if err := reOpenDevNull(rootfs); err != nil {
		return fmt.Errorf("Failed to reopen /dev/null %s", err)
	}

	if err := setupDevSymlinks(rootfs); err != nil {
		return fmt.Errorf("dev symlinks %s", err)
	}

	if err := syscall.Chdir(rootfs); err != nil {
		return fmt.Errorf("chdir into %s %s", rootfs, err)
	}

	if mountConfig.NoPivotRoot {
		err = MsMoveRoot(rootfs)
	} else {
		err = PivotRoot(rootfs)
	}

	if err != nil {
		return err
	}

	if mountConfig.ReadonlyFs {
		if err := SetReadonly(); err != nil {
			return fmt.Errorf("set readonly %s", err)
		}
	}

	syscall.Umask(0022)

	return nil
}

// mountSystem sets up linux specific system mounts like mqueue, sys, proc, shm, and devpts
// inside the mount namespace
func mountSystem(rootfs string, sysReadonly bool, mountConfig *MountConfig) error {
	for _, m := range newSystemMounts(rootfs, mountConfig.MountLabel, sysReadonly) {
		if err := os.MkdirAll(m.path, 0755); err != nil && !os.IsExist(err) {
			return fmt.Errorf("mkdirall %s %s", m.path, err)
		}
		if err := syscall.Mount(m.source, m.path, m.device, uintptr(m.flags), m.data); err != nil {
			return fmt.Errorf("mounting %s into %s %s", m.source, m.path, err)
		}
	}
	return nil
}

func createIfNotExists(path string, isDir bool) error {
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			if isDir {
				if err := os.MkdirAll(path, 0755); err != nil {
					return err
				}
			} else {
				if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
					return err
				}
				f, err := os.OpenFile(path, os.O_CREATE, 0755)
				if err != nil {
					return err
				}
				f.Close()
			}
		}
	}
	return nil
}

func setupDevSymlinks(rootfs string) error {
	var links = [][2]string{
		{"/proc/self/fd", "/dev/fd"},
		{"/proc/self/fd/0", "/dev/stdin"},
		{"/proc/self/fd/1", "/dev/stdout"},
		{"/proc/self/fd/2", "/dev/stderr"},
	}

	// kcore support can be toggled with CONFIG_PROC_KCORE; only create a symlink
	// in /dev if it exists in /proc.
	if _, err := os.Stat("/proc/kcore"); err == nil {
		links = append(links, [2]string{"/proc/kcore", "/dev/kcore"})
	}

	for _, link := range links {
		var (
			src = link[0]
			dst = filepath.Join(rootfs, link[1])
		)

		if err := os.Symlink(src, dst); err != nil && !os.IsExist(err) {
			return fmt.Errorf("symlink %s %s %s", src, dst, err)
		}
	}

	return nil
}

// TODO: this is crappy right now and should be cleaned up with a better way of handling system and
// standard bind mounts allowing them to be more dynamic
func newSystemMounts(rootfs, mountLabel string, sysReadonly bool) []mount {
	systemMounts := []mount{
		{source: "proc", path: filepath.Join(rootfs, "proc"), device: "proc", flags: defaultMountFlags},
		{source: "tmpfs", path: filepath.Join(rootfs, "dev"), device: "tmpfs", flags: syscall.MS_NOSUID | syscall.MS_STRICTATIME, data: label.FormatMountLabel("mode=755", mountLabel)},
		{source: "shm", path: filepath.Join(rootfs, "dev", "shm"), device: "tmpfs", flags: defaultMountFlags, data: label.FormatMountLabel("mode=1777,size=65536k", mountLabel)},
		{source: "mqueue", path: filepath.Join(rootfs, "dev", "mqueue"), device: "mqueue", flags: defaultMountFlags},
		{source: "devpts", path: filepath.Join(rootfs, "dev", "pts"), device: "devpts", flags: syscall.MS_NOSUID | syscall.MS_NOEXEC, data: label.FormatMountLabel("newinstance,ptmxmode=0666,mode=620,gid=5", mountLabel)},
	}

	sysMountFlags := defaultMountFlags
	if sysReadonly {
		sysMountFlags |= syscall.MS_RDONLY
	}

	systemMounts = append(systemMounts, mount{source: "sysfs", path: filepath.Join(rootfs, "sys"), device: "sysfs", flags: sysMountFlags})

	return systemMounts
}

// Is stdin, stdout or stderr were to be pointing to '/dev/null',
// this method will make them point to '/dev/null' from within this namespace.
func reOpenDevNull(rootfs string) error {
	var stat, devNullStat syscall.Stat_t
	file, err := os.Open(filepath.Join(rootfs, "/dev/null"))
	if err != nil {
		return fmt.Errorf("Failed to open /dev/null - %s", err)
	}
	defer file.Close()
	if err = syscall.Fstat(int(file.Fd()), &devNullStat); err != nil {
		return fmt.Errorf("Failed to stat /dev/null - %s", err)
	}
	for fd := 0; fd < 3; fd++ {
		if err = syscall.Fstat(fd, &stat); err != nil {
			return fmt.Errorf("Failed to stat fd %d - %s", fd, err)
		}
		if stat.Rdev == devNullStat.Rdev {
			// Close and re-open the fd.
			if err = syscall.Dup2(int(file.Fd()), fd); err != nil {
				return fmt.Errorf("Failed to dup fd %d to fd %d - %s", file.Fd(), fd, err)
			}
		}
	}
	return nil
}
