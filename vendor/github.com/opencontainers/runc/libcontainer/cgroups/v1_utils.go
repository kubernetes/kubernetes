package cgroups

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"

	securejoin "github.com/cyphar/filepath-securejoin"
	"github.com/moby/sys/mountinfo"
	"golang.org/x/sys/unix"
)

// Code in this source file are specific to cgroup v1,
// and must not be used from any cgroup v2 code.

const (
	CgroupNamePrefix = "name="
	defaultPrefix    = "/sys/fs/cgroup"
)

var (
	errUnified     = errors.New("not implemented for cgroup v2 unified hierarchy")
	ErrV1NoUnified = errors.New("invalid configuration: cannot use unified on cgroup v1")

	readMountinfoOnce sync.Once
	readMountinfoErr  error
	cgroupMountinfo   []*mountinfo.Info
)

type NotFoundError struct {
	Subsystem string
}

func (e *NotFoundError) Error() string {
	return fmt.Sprintf("mountpoint for %s not found", e.Subsystem)
}

func NewNotFoundError(sub string) error {
	return &NotFoundError{
		Subsystem: sub,
	}
}

func IsNotFound(err error) bool {
	var nfErr *NotFoundError
	return errors.As(err, &nfErr)
}

func tryDefaultPath(cgroupPath, subsystem string) string {
	if !strings.HasPrefix(defaultPrefix, cgroupPath) {
		return ""
	}

	// remove possible prefix
	subsystem = strings.TrimPrefix(subsystem, CgroupNamePrefix)

	// Make sure we're still under defaultPrefix, and resolve
	// a possible symlink (like cpu -> cpu,cpuacct).
	path, err := securejoin.SecureJoin(defaultPrefix, subsystem)
	if err != nil {
		return ""
	}

	// (1) path should be a directory.
	st, err := os.Lstat(path)
	if err != nil || !st.IsDir() {
		return ""
	}

	// (2) path should be a mount point.
	pst, err := os.Lstat(filepath.Dir(path))
	if err != nil {
		return ""
	}

	if st.Sys().(*syscall.Stat_t).Dev == pst.Sys().(*syscall.Stat_t).Dev {
		// parent dir has the same dev -- path is not a mount point
		return ""
	}

	// (3) path should have 'cgroup' fs type.
	fst := unix.Statfs_t{}
	err = unix.Statfs(path, &fst)
	if err != nil || fst.Type != unix.CGROUP_SUPER_MAGIC {
		return ""
	}

	return path
}

// readCgroupMountinfo returns a list of cgroup v1 mounts (i.e. the ones
// with fstype of "cgroup") for the current running process.
//
// The results are cached (to avoid re-reading mountinfo which is relatively
// expensive), so it is assumed that cgroup mounts are not being changed.
func readCgroupMountinfo() ([]*mountinfo.Info, error) {
	readMountinfoOnce.Do(func() {
		// mountinfo.GetMounts uses /proc/thread-self, so we can use it without
		// issues.
		cgroupMountinfo, readMountinfoErr = mountinfo.GetMounts(
			mountinfo.FSTypeFilter("cgroup"),
		)
	})
	return cgroupMountinfo, readMountinfoErr
}

// https://www.kernel.org/doc/Documentation/cgroup-v1/cgroups.txt
func FindCgroupMountpoint(cgroupPath, subsystem string) (string, error) {
	if IsCgroup2UnifiedMode() {
		return "", errUnified
	}

	// If subsystem is empty, we look for the cgroupv2 hybrid path.
	if len(subsystem) == 0 {
		return hybridMountpoint, nil
	}

	// Avoid parsing mountinfo by trying the default path first, if possible.
	if path := tryDefaultPath(cgroupPath, subsystem); path != "" {
		return path, nil
	}

	mnt, _, err := FindCgroupMountpointAndRoot(cgroupPath, subsystem)
	return mnt, err
}

func FindCgroupMountpointAndRoot(cgroupPath, subsystem string) (string, string, error) {
	if IsCgroup2UnifiedMode() {
		return "", "", errUnified
	}

	mi, err := readCgroupMountinfo()
	if err != nil {
		return "", "", err
	}

	return findCgroupMountpointAndRootFromMI(mi, cgroupPath, subsystem)
}

func findCgroupMountpointAndRootFromMI(mounts []*mountinfo.Info, cgroupPath, subsystem string) (string, string, error) {
	for _, mi := range mounts {
		if strings.HasPrefix(mi.Mountpoint, cgroupPath) {
			for _, opt := range strings.Split(mi.VFSOptions, ",") {
				if opt == subsystem {
					return mi.Mountpoint, mi.Root, nil
				}
			}
		}
	}

	return "", "", NewNotFoundError(subsystem)
}

func (m Mount) GetOwnCgroup(cgroups map[string]string) (string, error) {
	if len(m.Subsystems) == 0 {
		return "", errors.New("no subsystem for mount")
	}

	return getControllerPath(m.Subsystems[0], cgroups)
}

func getCgroupMountsHelper(ss map[string]bool, mounts []*mountinfo.Info, all bool) ([]Mount, error) {
	res := make([]Mount, 0, len(ss))
	numFound := 0
	for _, mi := range mounts {
		m := Mount{
			Mountpoint: mi.Mountpoint,
			Root:       mi.Root,
		}
		for _, opt := range strings.Split(mi.VFSOptions, ",") {
			seen, known := ss[opt]
			if !known || (!all && seen) {
				continue
			}
			ss[opt] = true
			opt = strings.TrimPrefix(opt, CgroupNamePrefix)
			m.Subsystems = append(m.Subsystems, opt)
			numFound++
		}
		if len(m.Subsystems) > 0 || all {
			res = append(res, m)
		}
		if !all && numFound >= len(ss) {
			break
		}
	}
	return res, nil
}

func getCgroupMountsV1(all bool) ([]Mount, error) {
	mi, err := readCgroupMountinfo()
	if err != nil {
		return nil, err
	}

	// We don't need to use /proc/thread-self here because runc always runs
	// with every thread in the same cgroup. This lets us avoid having to do
	// runtime.LockOSThread.
	allSubsystems, err := ParseCgroupFile("/proc/self/cgroup")
	if err != nil {
		return nil, err
	}

	allMap := make(map[string]bool)
	for s := range allSubsystems {
		allMap[s] = false
	}

	return getCgroupMountsHelper(allMap, mi, all)
}

// GetOwnCgroup returns the relative path to the cgroup docker is running in.
func GetOwnCgroup(subsystem string) (string, error) {
	if IsCgroup2UnifiedMode() {
		return "", errUnified
	}

	// We don't need to use /proc/thread-self here because runc always runs
	// with every thread in the same cgroup. This lets us avoid having to do
	// runtime.LockOSThread.
	cgroups, err := ParseCgroupFile("/proc/self/cgroup")
	if err != nil {
		return "", err
	}

	return getControllerPath(subsystem, cgroups)
}

func GetOwnCgroupPath(subsystem string) (string, error) {
	cgroup, err := GetOwnCgroup(subsystem)
	if err != nil {
		return "", err
	}

	// If subsystem is empty, we look for the cgroupv2 hybrid path.
	if len(subsystem) == 0 {
		return hybridMountpoint, nil
	}

	return getCgroupPathHelper(subsystem, cgroup)
}

func getCgroupPathHelper(subsystem, cgroup string) (string, error) {
	mnt, root, err := FindCgroupMountpointAndRoot("", subsystem)
	if err != nil {
		return "", err
	}

	// This is needed for nested containers, because in /proc/self/cgroup we
	// see paths from host, which don't exist in container.
	relCgroup, err := filepath.Rel(root, cgroup)
	if err != nil {
		return "", err
	}

	return filepath.Join(mnt, relCgroup), nil
}

func getControllerPath(subsystem string, cgroups map[string]string) (string, error) {
	if IsCgroup2UnifiedMode() {
		return "", errUnified
	}

	if p, ok := cgroups[subsystem]; ok {
		return p, nil
	}

	if p, ok := cgroups[CgroupNamePrefix+subsystem]; ok {
		return p, nil
	}

	return "", NewNotFoundError(subsystem)
}
