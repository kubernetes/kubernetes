package procfs

import (
	"fmt"
	"os"
	"path"
)

// FS represents the pseudo-filesystem proc, which provides an interface to
// kernel data structures.
type FS string

// DefaultMountPoint is the common mount point of the proc filesystem.
const DefaultMountPoint = "/proc"

// NewFS returns a new FS mounted under the given mountPoint. It will error
// if the mount point can't be read.
func NewFS(mountPoint string) (FS, error) {
	info, err := os.Stat(mountPoint)
	if err != nil {
		return "", fmt.Errorf("could not read %s: %s", mountPoint, err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("mount point %s is not a directory", mountPoint)
	}

	return FS(mountPoint), nil
}

func (fs FS) stat(p string) (os.FileInfo, error) {
	return os.Stat(path.Join(string(fs), p))
}

func (fs FS) open(p string) (*os.File, error) {
	return os.Open(path.Join(string(fs), p))
}

func (fs FS) readlink(p string) (string, error) {
	return os.Readlink(path.Join(string(fs), p))
}
