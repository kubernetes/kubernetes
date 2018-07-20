package containerfs

import (
	"path/filepath"
	"runtime"

	"github.com/containerd/continuity/driver"
	"github.com/containerd/continuity/pathdriver"
	"github.com/docker/docker/pkg/symlink"
)

// ContainerFS is that represents a root file system
type ContainerFS interface {
	// Path returns the path to the root. Note that this may not exist
	// on the local system, so the continuity operations must be used
	Path() string

	// ResolveScopedPath evaluates the given path scoped to the root.
	// For example, if root=/a, and path=/b/c, then this function would return /a/b/c.
	// If rawPath is true, then the function will not preform any modifications
	// before path resolution. Otherwise, the function will clean the given path
	// by making it an absolute path.
	ResolveScopedPath(path string, rawPath bool) (string, error)

	Driver
}

// Driver combines both continuity's Driver and PathDriver interfaces with a Platform
// field to determine the OS.
type Driver interface {
	// OS returns the OS where the rootfs is located. Essentially,
	// runtime.GOOS for everything aside from LCOW, which is "linux"
	OS() string

	// Architecture returns the hardware architecture where the
	// container is located.
	Architecture() string

	// Driver & PathDriver provide methods to manipulate files & paths
	driver.Driver
	pathdriver.PathDriver
}

// NewLocalContainerFS is a helper function to implement daemon's Mount interface
// when the graphdriver mount point is a local path on the machine.
func NewLocalContainerFS(path string) ContainerFS {
	return &local{
		path:       path,
		Driver:     driver.LocalDriver,
		PathDriver: pathdriver.LocalPathDriver,
	}
}

// NewLocalDriver provides file and path drivers for a local file system. They are
// essentially a wrapper around the `os` and `filepath` functions.
func NewLocalDriver() Driver {
	return &local{
		Driver:     driver.LocalDriver,
		PathDriver: pathdriver.LocalPathDriver,
	}
}

type local struct {
	path string
	driver.Driver
	pathdriver.PathDriver
}

func (l *local) Path() string {
	return l.path
}

func (l *local) ResolveScopedPath(path string, rawPath bool) (string, error) {
	cleanedPath := path
	if !rawPath {
		cleanedPath = cleanScopedPath(path)
	}
	return symlink.FollowSymlinkInScope(filepath.Join(l.path, cleanedPath), l.path)
}

func (l *local) OS() string {
	return runtime.GOOS
}

func (l *local) Architecture() string {
	return runtime.GOARCH
}
