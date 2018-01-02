package rootfs

import (
	"os"
	"path/filepath"
	"syscall"
)

const (
	defaultInitializer = "linux-init"
)

func init() {
	initializers[defaultInitializer] = initFS
}

func createDirectory(name string, uid, gid int) initializerFunc {
	return func(root string) error {
		dname := filepath.Join(root, name)
		st, err := os.Stat(dname)
		if err != nil && !os.IsNotExist(err) {
			return err
		} else if err == nil {
			if st.IsDir() {
				stat := st.Sys().(*syscall.Stat_t)
				if int(stat.Gid) == gid && int(stat.Uid) == uid {
					return nil
				}
			} else {
				if err := os.Remove(dname); err != nil {
					return err
				}
				if err := os.Mkdir(dname, 0755); err != nil {
					return err
				}
			}
		} else {
			if err := os.Mkdir(dname, 0755); err != nil {
				return err
			}
		}

		return os.Chown(dname, uid, gid)
	}
}

func touchFile(name string, uid, gid int) initializerFunc {
	return func(root string) error {
		fname := filepath.Join(root, name)

		st, err := os.Stat(fname)
		if err != nil && !os.IsNotExist(err) {
			return err
		} else if err == nil {
			stat := st.Sys().(*syscall.Stat_t)
			if int(stat.Gid) == gid && int(stat.Uid) == uid {
				return nil
			}
			return os.Chown(fname, uid, gid)
		}

		f, err := os.OpenFile(fname, os.O_CREATE, 0644)
		if err != nil {
			return err
		}
		defer f.Close()

		return f.Chown(uid, gid)
	}
}

func symlink(oldname, newname string) initializerFunc {
	return func(root string) error {
		linkName := filepath.Join(root, newname)
		if _, err := os.Stat(linkName); err != nil && !os.IsNotExist(err) {
			return err
		} else if err == nil {
			return nil
		}
		return os.Symlink(oldname, linkName)
	}
}

func initFS(root string) error {
	st, err := os.Stat(root)
	if err != nil {
		return err
	}
	stat := st.Sys().(*syscall.Stat_t)
	uid := int(stat.Uid)
	gid := int(stat.Gid)

	initFuncs := []initializerFunc{
		createDirectory("/dev", uid, gid),
		createDirectory("/dev/pts", uid, gid),
		createDirectory("/dev/shm", uid, gid),
		touchFile("/dev/console", uid, gid),
		createDirectory("/proc", uid, gid),
		createDirectory("/sys", uid, gid),
		createDirectory("/etc", uid, gid),
		touchFile("/etc/resolv.conf", uid, gid),
		touchFile("/etc/hosts", uid, gid),
		touchFile("/etc/hostname", uid, gid),
		symlink("/proc/mounts", "/etc/mtab"),
	}

	for _, fn := range initFuncs {
		if err := fn(root); err != nil {
			return err
		}
	}

	return nil
}
