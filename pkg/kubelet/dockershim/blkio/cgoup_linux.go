// +build linux

package blkio

import (
	"path/filepath"
	"sync"

	"github.com/opencontainers/runc/libcontainer/configs"
)

var (
	// The absolute path to the root of the cgroup hierarchies.
	cgroupRootLock sync.Mutex
	cgroupRoot     string
	blkioSubsystem Subsystem
	// FindCgroupMountpointDir easy for test
	FindCgroupMountpointDir func() (string, error)
)

type Subsystem interface {
	// Name returns the name of the subsystem.
	Name() string
	// Set the cgroup represented by cgroup.
	Set(path string, cgroup *configs.Cgroup) error
}

func getBlkioCgroupPath(subsystemName, cgroupParent, containerID string) (path string, err error) {
	root, err := getCgroupRoot()
	if err != nil {
		return "", err
	}
	if len(cgroupParent) <= 0 {
		cgroupParent = DefaultCgroupParent
	}
	path = filepath.Join(root, subsystemName, cgroupParent, containerID)
	// path = filepath.Join(root, "blkio", cgroupParent, containerID)
	return path, nil
}

// Gets the cgroupRoot.
func getCgroupRoot() (string, error) {
	cgroupRootLock.Lock()
	defer cgroupRootLock.Unlock()

	if cgroupRoot != "" {
		return cgroupRoot, nil
	}

	root, err := FindCgroupMountpointDir()
	if err != nil {
		return "", err
	}

	if _, err := unixos.Stat(root); err != nil {
		return "", err
	}

	cgroupRoot = root
	return cgroupRoot, nil
}
