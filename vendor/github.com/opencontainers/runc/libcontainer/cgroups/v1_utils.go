package cgroups

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	securejoin "github.com/cyphar/filepath-securejoin"
	"golang.org/x/sys/unix"
)

// Code in this source file are specific to cgroup v1,
// and must not be used from any cgroup v2 code.

const (
	CgroupNamePrefix = "name="
	defaultPrefix    = "/sys/fs/cgroup"
)

var (
	errUnified = errors.New("not implemented for cgroup v2 unified hierarchy")
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
	if err == nil {
		return false
	}
	_, ok := err.(*NotFoundError)
	return ok
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

// https://www.kernel.org/doc/Documentation/cgroup-v1/cgroups.txt
func FindCgroupMountpoint(cgroupPath, subsystem string) (string, error) {
	if IsCgroup2UnifiedMode() {
		return "", errUnified
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

	// Avoid parsing mountinfo by checking if subsystem is valid/available.
	if !isSubsystemAvailable(subsystem) {
		return "", "", NewNotFoundError(subsystem)
	}

	f, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		return "", "", err
	}
	defer f.Close()

	return findCgroupMountpointAndRootFromReader(f, cgroupPath, subsystem)
}

func findCgroupMountpointAndRootFromReader(reader io.Reader, cgroupPath, subsystem string) (string, string, error) {
	scanner := bufio.NewScanner(reader)
	for scanner.Scan() {
		txt := scanner.Text()
		fields := strings.Fields(txt)
		if len(fields) < 9 {
			continue
		}
		if strings.HasPrefix(fields[4], cgroupPath) {
			for _, opt := range strings.Split(fields[len(fields)-1], ",") {
				if opt == subsystem {
					return fields[4], fields[3], nil
				}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return "", "", err
	}

	return "", "", NewNotFoundError(subsystem)
}

func isSubsystemAvailable(subsystem string) bool {
	if IsCgroup2UnifiedMode() {
		panic("don't call isSubsystemAvailable from cgroupv2 code")
	}

	cgroups, err := ParseCgroupFile("/proc/self/cgroup")
	if err != nil {
		return false
	}
	_, avail := cgroups[subsystem]
	return avail
}

func (m Mount) GetOwnCgroup(cgroups map[string]string) (string, error) {
	if len(m.Subsystems) == 0 {
		return "", fmt.Errorf("no subsystem for mount")
	}

	return getControllerPath(m.Subsystems[0], cgroups)
}

func getCgroupMountsHelper(ss map[string]bool, mi io.Reader, all bool) ([]Mount, error) {
	res := make([]Mount, 0, len(ss))
	scanner := bufio.NewScanner(mi)
	numFound := 0
	for scanner.Scan() && numFound < len(ss) {
		txt := scanner.Text()
		sepIdx := strings.Index(txt, " - ")
		if sepIdx == -1 {
			return nil, fmt.Errorf("invalid mountinfo format")
		}
		if txt[sepIdx+3:sepIdx+10] == "cgroup2" || txt[sepIdx+3:sepIdx+9] != "cgroup" {
			continue
		}
		fields := strings.Split(txt, " ")
		m := Mount{
			Mountpoint: fields[4],
			Root:       fields[3],
		}
		for _, opt := range strings.Split(fields[len(fields)-1], ",") {
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
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return res, nil
}

func getCgroupMountsV1(all bool) ([]Mount, error) {
	f, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	allSubsystems, err := ParseCgroupFile("/proc/self/cgroup")
	if err != nil {
		return nil, err
	}

	allMap := make(map[string]bool)
	for s := range allSubsystems {
		allMap[s] = false
	}
	return getCgroupMountsHelper(allMap, f, all)
}

// GetOwnCgroup returns the relative path to the cgroup docker is running in.
func GetOwnCgroup(subsystem string) (string, error) {
	if IsCgroup2UnifiedMode() {
		return "", errUnified
	}
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

	return getCgroupPathHelper(subsystem, cgroup)
}

func GetInitCgroup(subsystem string) (string, error) {
	if IsCgroup2UnifiedMode() {
		return "", errUnified
	}
	cgroups, err := ParseCgroupFile("/proc/1/cgroup")
	if err != nil {
		return "", err
	}

	return getControllerPath(subsystem, cgroups)
}

func GetInitCgroupPath(subsystem string) (string, error) {
	cgroup, err := GetInitCgroup(subsystem)
	if err != nil {
		return "", err
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
