package fs

import (
	"errors"
	"os"
	"path/filepath"
	"sync"

	"golang.org/x/sys/unix"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/utils"
)

// The absolute path to the root of the cgroup hierarchies.
var (
	cgroupRootLock sync.Mutex
	cgroupRoot     string
)

const defaultCgroupRoot = "/sys/fs/cgroup"

func initPaths(cg *configs.Cgroup) (map[string]string, error) {
	root, err := rootPath()
	if err != nil {
		return nil, err
	}

	inner, err := innerPath(cg)
	if err != nil {
		return nil, err
	}

	paths := make(map[string]string)
	for _, sys := range subsystems {
		name := sys.Name()
		path, err := subsysPath(root, inner, name)
		if err != nil {
			// The non-presence of the devices subsystem
			// is considered fatal for security reasons.
			if cgroups.IsNotFound(err) && (cg.SkipDevices || name != "devices") {
				continue
			}

			return nil, err
		}
		paths[name] = path
	}

	return paths, nil
}

func tryDefaultCgroupRoot() string {
	var st, pst unix.Stat_t

	// (1) it should be a directory...
	err := unix.Lstat(defaultCgroupRoot, &st)
	if err != nil || st.Mode&unix.S_IFDIR == 0 {
		return ""
	}

	// (2) ... and a mount point ...
	err = unix.Lstat(filepath.Dir(defaultCgroupRoot), &pst)
	if err != nil {
		return ""
	}

	if st.Dev == pst.Dev {
		// parent dir has the same dev -- not a mount point
		return ""
	}

	// (3) ... of 'tmpfs' fs type.
	var fst unix.Statfs_t
	err = unix.Statfs(defaultCgroupRoot, &fst)
	if err != nil || fst.Type != unix.TMPFS_MAGIC {
		return ""
	}

	// (4) it should have at least 1 entry ...
	dir, err := os.Open(defaultCgroupRoot)
	if err != nil {
		return ""
	}
	defer dir.Close()
	names, err := dir.Readdirnames(1)
	if err != nil {
		return ""
	}
	if len(names) < 1 {
		return ""
	}
	// ... which is a cgroup mount point.
	err = unix.Statfs(filepath.Join(defaultCgroupRoot, names[0]), &fst)
	if err != nil || fst.Type != unix.CGROUP_SUPER_MAGIC {
		return ""
	}

	return defaultCgroupRoot
}

// rootPath finds and returns path to the root of the cgroup hierarchies.
func rootPath() (string, error) {
	cgroupRootLock.Lock()
	defer cgroupRootLock.Unlock()

	if cgroupRoot != "" {
		return cgroupRoot, nil
	}

	// fast path
	cgroupRoot = tryDefaultCgroupRoot()
	if cgroupRoot != "" {
		return cgroupRoot, nil
	}

	// slow path: parse mountinfo
	mi, err := cgroups.GetCgroupMounts(false)
	if err != nil {
		return "", err
	}
	if len(mi) < 1 {
		return "", errors.New("no cgroup mount found in mountinfo")
	}

	// Get the first cgroup mount (e.g. "/sys/fs/cgroup/memory"),
	// use its parent directory.
	root := filepath.Dir(mi[0].Mountpoint)

	if _, err := os.Stat(root); err != nil {
		return "", err
	}

	cgroupRoot = root
	return cgroupRoot, nil
}

func innerPath(c *configs.Cgroup) (string, error) {
	if (c.Name != "" || c.Parent != "") && c.Path != "" {
		return "", errors.New("cgroup: either Path or Name and Parent should be used")
	}

	// XXX: Do not remove CleanPath. Path safety is important! -- cyphar
	innerPath := utils.CleanPath(c.Path)
	if innerPath == "" {
		cgParent := utils.CleanPath(c.Parent)
		cgName := utils.CleanPath(c.Name)
		innerPath = filepath.Join(cgParent, cgName)
	}

	return innerPath, nil
}

func subsysPath(root, inner, subsystem string) (string, error) {
	// If the cgroup name/path is absolute do not look relative to the cgroup of the init process.
	if filepath.IsAbs(inner) {
		mnt, err := cgroups.FindCgroupMountpoint(root, subsystem)
		// If we didn't mount the subsystem, there is no point we make the path.
		if err != nil {
			return "", err
		}

		// Sometimes subsystems can be mounted together as 'cpu,cpuacct'.
		return filepath.Join(root, filepath.Base(mnt), inner), nil
	}

	// Use GetOwnCgroupPath for dind-like cases, when cgroupns is not
	// available. This is ugly.
	parentPath, err := cgroups.GetOwnCgroupPath(subsystem)
	if err != nil {
		return "", err
	}

	return filepath.Join(parentPath, inner), nil
}

func apply(path string, pid int) error {
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(path, 0o755); err != nil {
		return err
	}
	return cgroups.WriteCgroupProc(path, pid)
}
