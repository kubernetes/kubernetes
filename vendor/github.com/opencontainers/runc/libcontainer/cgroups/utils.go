package cgroups

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/opencontainers/runc/libcontainer/userns"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

const (
	CgroupProcesses   = "cgroup.procs"
	unifiedMountpoint = "/sys/fs/cgroup"
	hybridMountpoint  = "/sys/fs/cgroup/unified"
)

var (
	isUnifiedOnce sync.Once
	isUnified     bool
	isHybridOnce  sync.Once
	isHybrid      bool
)

// IsCgroup2UnifiedMode returns whether we are running in cgroup v2 unified mode.
func IsCgroup2UnifiedMode() bool {
	isUnifiedOnce.Do(func() {
		var st unix.Statfs_t
		err := unix.Statfs(unifiedMountpoint, &st)
		if err != nil {
			if os.IsNotExist(err) && userns.RunningInUserNS() {
				// ignore the "not found" error if running in userns
				logrus.WithError(err).Debugf("%s missing, assuming cgroup v1", unifiedMountpoint)
				isUnified = false
				return
			}
			panic(fmt.Sprintf("cannot statfs cgroup root: %s", err))
		}
		isUnified = st.Type == unix.CGROUP2_SUPER_MAGIC
	})
	return isUnified
}

// IsCgroup2HybridMode returns whether we are running in cgroup v2 hybrid mode.
func IsCgroup2HybridMode() bool {
	isHybridOnce.Do(func() {
		var st unix.Statfs_t
		err := unix.Statfs(hybridMountpoint, &st)
		if err != nil {
			isHybrid = false
			if !os.IsNotExist(err) {
				// Report unexpected errors.
				logrus.WithError(err).Debugf("statfs(%q) failed", hybridMountpoint)
			}
			return
		}
		isHybrid = st.Type == unix.CGROUP2_SUPER_MAGIC
	})
	return isHybrid
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
		data, err := ReadFile("/sys/fs/cgroup", "cgroup.controllers")
		if err != nil {
			return nil, err
		}
		subsystems := append(pseudo, strings.Fields(data)...)
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

func readProcsFile(dir string) ([]int, error) {
	f, err := OpenFile(dir, CgroupProcesses, os.O_RDONLY)
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

func rmdir(path string) error {
	err := unix.Rmdir(path)
	if err == nil || err == unix.ENOENT { //nolint:errorlint // unix errors are bare
		return nil
	}
	return &os.PathError{Op: "rmdir", Path: path, Err: err}
}

// RemovePath aims to remove cgroup path. It does so recursively,
// by removing any subdirectories (sub-cgroups) first.
func RemovePath(path string) error {
	// try the fast path first
	if err := rmdir(path); err == nil {
		return nil
	}

	infos, err := os.ReadDir(path)
	if err != nil {
		if os.IsNotExist(err) {
			err = nil
		}
		return err
	}
	for _, info := range infos {
		if info.IsDir() {
			// We should remove subcgroups dir first
			if err = RemovePath(filepath.Join(path, info.Name())); err != nil {
				break
			}
		}
	}
	if err == nil {
		err = rmdir(path)
	}
	return err
}

// RemovePaths iterates over the provided paths removing them.
// We trying to remove all paths five times with increasing delay between tries.
// If after all there are not removed cgroups - appropriate error will be
// returned.
func RemovePaths(paths map[string]string) (err error) {
	const retries = 5
	delay := 10 * time.Millisecond
	for i := 0; i < retries; i++ {
		if i != 0 {
			time.Sleep(delay)
			delay *= 2
		}
		for s, p := range paths {
			if err := RemovePath(p); err != nil {
				// do not log intermediate iterations
				switch i {
				case 0:
					logrus.WithError(err).Warnf("Failed to remove cgroup (will retry)")
				case retries - 1:
					logrus.WithError(err).Error("Failed to remove cgroup")
				}
			}
			_, err := os.Stat(p)
			// We need this strange way of checking cgroups existence because
			// RemoveAll almost always returns error, even on already removed
			// cgroups
			if os.IsNotExist(err) {
				delete(paths, s)
			}
		}
		if len(paths) == 0 {
			//nolint:ineffassign,staticcheck // done to help garbage collecting: opencontainers/runc#2506
			paths = make(map[string]string)
			return nil
		}
	}
	return fmt.Errorf("Failed to remove paths: %v", paths)
}

var (
	hugePageSizes []string
	initHPSOnce   sync.Once
)

func HugePageSizes() []string {
	initHPSOnce.Do(func() {
		dir, err := os.OpenFile("/sys/kernel/mm/hugepages", unix.O_DIRECTORY|unix.O_RDONLY, 0)
		if err != nil {
			return
		}
		files, err := dir.Readdirnames(0)
		dir.Close()
		if err != nil {
			return
		}

		hugePageSizes, err = getHugePageSizeFromFilenames(files)
		if err != nil {
			logrus.Warn("HugePageSizes: ", err)
		}
	})

	return hugePageSizes
}

func getHugePageSizeFromFilenames(fileNames []string) ([]string, error) {
	pageSizes := make([]string, 0, len(fileNames))
	var warn error

	for _, file := range fileNames {
		// example: hugepages-1048576kB
		val := strings.TrimPrefix(file, "hugepages-")
		if len(val) == len(file) {
			// Unexpected file name: no prefix found, ignore it.
			continue
		}
		// The suffix is always "kB" (as of Linux 5.13). If we find
		// something else, produce an error but keep going.
		eLen := len(val) - 2
		val = strings.TrimSuffix(val, "kB")
		if len(val) != eLen {
			// Highly unlikely.
			if warn == nil {
				warn = errors.New(file + `: invalid suffix (expected "kB")`)
			}
			continue
		}
		size, err := strconv.Atoi(val)
		if err != nil {
			// Highly unlikely.
			if warn == nil {
				warn = fmt.Errorf("%s: %w", file, err)
			}
			continue
		}
		// Model after https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/mm/hugetlb_cgroup.c?id=eff48ddeab782e35e58ccc8853f7386bbae9dec4#n574
		// but in our case the size is in KB already.
		if size >= (1 << 20) {
			val = strconv.Itoa(size>>20) + "GB"
		} else if size >= (1 << 10) {
			val = strconv.Itoa(size>>10) + "MB"
		} else {
			val += "KB"
		}
		pageSizes = append(pageSizes, val)
	}

	return pageSizes, warn
}

// GetPids returns all pids, that were added to cgroup at path.
func GetPids(dir string) ([]int, error) {
	return readProcsFile(dir)
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

	file, err := OpenFile(dir, CgroupProcesses, os.O_WRONLY)
	if err != nil {
		return fmt.Errorf("failed to write %v: %w", pid, err)
	}
	defer file.Close()

	for i := 0; i < 5; i++ {
		_, err = file.WriteString(strconv.Itoa(pid))
		if err == nil {
			return nil
		}

		// EINVAL might mean that the task being added to cgroup.procs is in state
		// TASK_NEW. We should attempt to do so again.
		if errors.Is(err, unix.EINVAL) {
			time.Sleep(30 * time.Millisecond)
			continue
		}

		return fmt.Errorf("failed to write %v: %w", pid, err)
	}
	return err
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

// Since the OCI spec is designed for cgroup v1, in some cases
// there is need to convert from the cgroup v1 configuration to cgroup v2
// the formula for BlkIOWeight to IOWeight is y = (1 + (x - 10) * 9999 / 990)
// convert linearly from [10-1000] to [1-10000]
func ConvertBlkIOToIOWeightValue(blkIoWeight uint16) uint64 {
	if blkIoWeight == 0 {
		return 0
	}
	return 1 + (uint64(blkIoWeight)-10)*9999/990
}
