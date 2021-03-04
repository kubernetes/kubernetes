// +build linux

package cgroups

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	units "github.com/docker/go-units"
	"golang.org/x/sys/unix"
)

const (
	CgroupProcesses   = "cgroup.procs"
	unifiedMountpoint = "/sys/fs/cgroup"
)

var (
	isUnifiedOnce sync.Once
	isUnified     bool
)

// HugePageSizeUnitList is a list of the units used by the linux kernel when
// naming the HugePage control files.
// https://www.kernel.org/doc/Documentation/cgroup-v1/hugetlb.txt
// TODO Since the kernel only use KB, MB and GB; TB and PB should be removed,
// depends on https://github.com/docker/go-units/commit/a09cd47f892041a4fac473133d181f5aea6fa393
var HugePageSizeUnitList = []string{"B", "KB", "MB", "GB", "TB", "PB"}

// IsCgroup2UnifiedMode returns whether we are running in cgroup v2 unified mode.
func IsCgroup2UnifiedMode() bool {
	isUnifiedOnce.Do(func() {
		var st unix.Statfs_t
		if err := unix.Statfs(unifiedMountpoint, &st); err != nil {
			panic("cannot statfs cgroup root")
		}
		isUnified = st.Type == unix.CGROUP2_SUPER_MAGIC
	})
	return isUnified
}

type Mount struct {
	Mountpoint string
	Root       string
	Subsystems []string
}

// GetCgroupMounts returns the mounts for the cgroup subsystems.
// all indicates whether to return just the first instance or all the mounts.
// This function should not be used from cgroupv2 code, as in this case
// all the controllers are available under the constant unifiedMountpoint.
func GetCgroupMounts(all bool) ([]Mount, error) {
	if IsCgroup2UnifiedMode() {
		// TODO: remove cgroupv2 case once all external users are converted
		availableControllers, err := GetAllSubsystems()
		if err != nil {
			return nil, err
		}
		m := Mount{
			Mountpoint: unifiedMountpoint,
			Root:       unifiedMountpoint,
			Subsystems: availableControllers,
		}
		return []Mount{m}, nil
	}

	return getCgroupMountsV1(all)
}

// GetAllSubsystems returns all the cgroup subsystems supported by the kernel
func GetAllSubsystems() ([]string, error) {
	// /proc/cgroups is meaningless for v2
	// https://github.com/torvalds/linux/blob/v5.3/Documentation/admin-guide/cgroup-v2.rst#deprecated-v1-core-features
	if IsCgroup2UnifiedMode() {
		// "pseudo" controllers do not appear in /sys/fs/cgroup/cgroup.controllers.
		// - devices: implemented in kernel 4.15
		// - freezer: implemented in kernel 5.2
		// We assume these are always available, as it is hard to detect availability.
		pseudo := []string{"devices", "freezer"}
		data, err := ioutil.ReadFile("/sys/fs/cgroup/cgroup.controllers")
		if err != nil {
			return nil, err
		}
		subsystems := append(pseudo, strings.Fields(string(data))...)
		return subsystems, nil
	}
	f, err := os.Open("/proc/cgroups")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	subsystems := []string{}

	s := bufio.NewScanner(f)
	for s.Scan() {
		text := s.Text()
		if text[0] != '#' {
			parts := strings.Fields(text)
			if len(parts) >= 4 && parts[3] != "0" {
				subsystems = append(subsystems, parts[0])
			}
		}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	return subsystems, nil
}

func readProcsFile(file string) ([]int, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		s   = bufio.NewScanner(f)
		out = []int{}
	)

	for s.Scan() {
		if t := s.Text(); t != "" {
			pid, err := strconv.Atoi(t)
			if err != nil {
				return nil, err
			}
			out = append(out, pid)
		}
	}
	return out, s.Err()
}

// ParseCgroupFile parses the given cgroup file, typically /proc/self/cgroup
// or /proc/<pid>/cgroup, into a map of subsystems to cgroup paths, e.g.
//   "cpu": "/user.slice/user-1000.slice"
//   "pids": "/user.slice/user-1000.slice"
// etc.
//
// Note that for cgroup v2 unified hierarchy, there are no per-controller
// cgroup paths, so the resulting map will have a single element where the key
// is empty string ("") and the value is the cgroup path the <pid> is in.
func ParseCgroupFile(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return parseCgroupFromReader(f)
}

// helper function for ParseCgroupFile to make testing easier
func parseCgroupFromReader(r io.Reader) (map[string]string, error) {
	s := bufio.NewScanner(r)
	cgroups := make(map[string]string)

	for s.Scan() {
		text := s.Text()
		// from cgroups(7):
		// /proc/[pid]/cgroup
		// ...
		// For each cgroup hierarchy ... there is one entry
		// containing three colon-separated fields of the form:
		//     hierarchy-ID:subsystem-list:cgroup-path
		parts := strings.SplitN(text, ":", 3)
		if len(parts) < 3 {
			return nil, fmt.Errorf("invalid cgroup entry: must contain at least two colons: %v", text)
		}

		for _, subs := range strings.Split(parts[1], ",") {
			cgroups[subs] = parts[2]
		}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}

	return cgroups, nil
}

func PathExists(path string) bool {
	if _, err := os.Stat(path); err != nil {
		return false
	}
	return true
}

func EnterPid(cgroupPaths map[string]string, pid int) error {
	for _, path := range cgroupPaths {
		if PathExists(path) {
			if err := WriteCgroupProc(path, pid); err != nil {
				return err
			}
		}
	}
	return nil
}

// RemovePaths iterates over the provided paths removing them.
// We trying to remove all paths five times with increasing delay between tries.
// If after all there are not removed cgroups - appropriate error will be
// returned.
func RemovePaths(paths map[string]string) (err error) {
	delay := 10 * time.Millisecond
	for i := 0; i < 5; i++ {
		if i != 0 {
			time.Sleep(delay)
			delay *= 2
		}
		for s, p := range paths {
			os.RemoveAll(p)
			// TODO: here probably should be logging
			_, err := os.Stat(p)
			// We need this strange way of checking cgroups existence because
			// RemoveAll almost always returns error, even on already removed
			// cgroups
			if os.IsNotExist(err) {
				delete(paths, s)
			}
		}
		if len(paths) == 0 {
			return nil
		}
	}
	return fmt.Errorf("Failed to remove paths: %v", paths)
}

func GetHugePageSize() ([]string, error) {
	files, err := ioutil.ReadDir("/sys/kernel/mm/hugepages")
	if err != nil {
		return []string{}, err
	}
	var fileNames []string
	for _, st := range files {
		fileNames = append(fileNames, st.Name())
	}
	return getHugePageSizeFromFilenames(fileNames)
}

func getHugePageSizeFromFilenames(fileNames []string) ([]string, error) {
	var pageSizes []string
	for _, fileName := range fileNames {
		nameArray := strings.Split(fileName, "-")
		pageSize, err := units.RAMInBytes(nameArray[1])
		if err != nil {
			return []string{}, err
		}
		sizeString := units.CustomSize("%g%s", float64(pageSize), 1024.0, HugePageSizeUnitList)
		pageSizes = append(pageSizes, sizeString)
	}

	return pageSizes, nil
}

// GetPids returns all pids, that were added to cgroup at path.
func GetPids(dir string) ([]int, error) {
	return readProcsFile(filepath.Join(dir, CgroupProcesses))
}

// GetAllPids returns all pids, that were added to cgroup at path and to all its
// subcgroups.
func GetAllPids(path string) ([]int, error) {
	var pids []int
	// collect pids from all sub-cgroups
	err := filepath.Walk(path, func(p string, info os.FileInfo, iErr error) error {
		if iErr != nil {
			return iErr
		}
		if info.IsDir() || info.Name() != CgroupProcesses {
			return nil
		}
		cPids, err := readProcsFile(p)
		if err != nil {
			return err
		}
		pids = append(pids, cPids...)
		return nil
	})
	return pids, err
}

// WriteCgroupProc writes the specified pid into the cgroup's cgroup.procs file
func WriteCgroupProc(dir string, pid int) error {
	// Normally dir should not be empty, one case is that cgroup subsystem
	// is not mounted, we will get empty dir, and we want it fail here.
	if dir == "" {
		return fmt.Errorf("no such directory for %s", CgroupProcesses)
	}

	// Dont attach any pid to the cgroup if -1 is specified as a pid
	if pid == -1 {
		return nil
	}

	cgroupProcessesFile, err := os.OpenFile(filepath.Join(dir, CgroupProcesses), os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0700)
	if err != nil {
		return fmt.Errorf("failed to write %v to %v: %v", pid, CgroupProcesses, err)
	}
	defer cgroupProcessesFile.Close()

	for i := 0; i < 5; i++ {
		_, err = cgroupProcessesFile.WriteString(strconv.Itoa(pid))
		if err == nil {
			return nil
		}

		// EINVAL might mean that the task being added to cgroup.procs is in state
		// TASK_NEW. We should attempt to do so again.
		if errors.Is(err, unix.EINVAL) {
			time.Sleep(30 * time.Millisecond)
			continue
		}

		return fmt.Errorf("failed to write %v to %v: %v", pid, CgroupProcesses, err)
	}
	return err
}

// Since the OCI spec is designed for cgroup v1, in some cases
// there is need to convert from the cgroup v1 configuration to cgroup v2
// the formula for BlkIOWeight is y = (1 + (x - 10) * 9999 / 990)
// convert linearly from [10-1000] to [1-10000]
func ConvertBlkIOToCgroupV2Value(blkIoWeight uint16) uint64 {
	if blkIoWeight == 0 {
		return 0
	}
	return uint64(1 + (uint64(blkIoWeight)-10)*9999/990)
}

// Since the OCI spec is designed for cgroup v1, in some cases
// there is need to convert from the cgroup v1 configuration to cgroup v2
// the formula for cpuShares is y = (1 + ((x - 2) * 9999) / 262142)
// convert from [2-262144] to [1-10000]
// 262144 comes from Linux kernel definition "#define MAX_SHARES (1UL << 18)"
func ConvertCPUSharesToCgroupV2Value(cpuShares uint64) uint64 {
	if cpuShares == 0 {
		return 0
	}
	return (1 + ((cpuShares-2)*9999)/262142)
}

// ConvertMemorySwapToCgroupV2Value converts MemorySwap value from OCI spec
// for use by cgroup v2 drivers. A conversion is needed since Resources.MemorySwap
// is defined as memory+swap combined, while in cgroup v2 swap is a separate value.
func ConvertMemorySwapToCgroupV2Value(memorySwap, memory int64) (int64, error) {
	// for compatibility with cgroup1 controller, set swap to unlimited in
	// case the memory is set to unlimited, and swap is not explicitly set,
	// treating the request as "set both memory and swap to unlimited".
	if memory == -1 && memorySwap == 0 {
		return -1, nil
	}
	if memorySwap == -1 || memorySwap == 0 {
		// -1 is "max", 0 is "unset", so treat as is
		return memorySwap, nil
	}
	// sanity checks
	if memory == 0 || memory == -1 {
		return 0, errors.New("unable to set swap limit without memory limit")
	}
	if memory < 0 {
		return 0, fmt.Errorf("invalid memory value: %d", memory)
	}
	if memorySwap < memory {
		return 0, errors.New("memory+swap limit should be >= memory limit")
	}

	return memorySwap - memory, nil
}
