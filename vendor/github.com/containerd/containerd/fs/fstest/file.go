package fstest

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"time"
)

// Applier applies single file changes
type Applier interface {
	Apply(root string) error
}

type applyFn func(root string) error

func (a applyFn) Apply(root string) error {
	return a(root)
}

// CreateFile returns a file applier which creates a file as the
// provided name with the given content and permission.
func CreateFile(name string, content []byte, perm os.FileMode) Applier {
	return applyFn(func(root string) error {
		fullPath := filepath.Join(root, name)
		if err := ioutil.WriteFile(fullPath, content, perm); err != nil {
			return err
		}
		return os.Chmod(fullPath, perm)
	})
}

// Remove returns a file applier which removes the provided file name
func Remove(name string) Applier {
	return applyFn(func(root string) error {
		return os.Remove(filepath.Join(root, name))
	})
}

// RemoveAll returns a file applier which removes the provided file name
// as in os.RemoveAll
func RemoveAll(name string) Applier {
	return applyFn(func(root string) error {
		return os.RemoveAll(filepath.Join(root, name))
	})
}

// CreateDir returns a file applier to create the directory with
// the provided name and permission
func CreateDir(name string, perm os.FileMode) Applier {
	return applyFn(func(root string) error {
		fullPath := filepath.Join(root, name)
		if err := os.MkdirAll(fullPath, perm); err != nil {
			return err
		}
		return os.Chmod(fullPath, perm)
	})
}

// Rename returns a file applier which renames a file
func Rename(old, new string) Applier {
	return applyFn(func(root string) error {
		return os.Rename(filepath.Join(root, old), filepath.Join(root, new))
	})
}

// Chown returns a file applier which changes the ownership of a file
func Chown(name string, uid, gid int) Applier {
	return applyFn(func(root string) error {
		return os.Chown(filepath.Join(root, name), uid, gid)
	})
}

// Chtime changes access and mod time of file
func Chtime(name string, t time.Time) Applier {
	return applyFn(func(root string) error {
		return os.Chtimes(filepath.Join(root, name), t, t)
	})
}

// Chmod returns a file applier which changes the file permission
func Chmod(name string, perm os.FileMode) Applier {
	return applyFn(func(root string) error {
		return os.Chmod(filepath.Join(root, name), perm)
	})
}

// Symlink returns a file applier which creates a symbolic link
func Symlink(oldname, newname string) Applier {
	return applyFn(func(root string) error {
		return os.Symlink(oldname, filepath.Join(root, newname))
	})
}

// Link returns a file applier which creates a hard link
func Link(oldname, newname string) Applier {
	return applyFn(func(root string) error {
		return os.Link(filepath.Join(root, oldname), filepath.Join(root, newname))
	})
}

// TODO: Make platform specific, windows applier is always no-op
//func Mknod(name string, mode int32, dev int) Applier {
//	return func(root string) error {
//		return return syscall.Mknod(path, mode, dev)
//	}
//}

// Apply returns a new applier from the given appliers
func Apply(appliers ...Applier) Applier {
	return applyFn(func(root string) error {
		for _, a := range appliers {
			if err := a.Apply(root); err != nil {
				return err
			}
		}
		return nil
	})
}
