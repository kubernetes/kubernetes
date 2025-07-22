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

	"github.com/moby/sys/userns"
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
			level := logrus.WarnLevel
			if os.IsNotExist(err) && userns.RunningInUserNS() {
				// For rootless containers, sweep it under the rug.
				level = logrus.DebugLevel
			}
			logrus.StandardLogger().Logf(level,
				"statfs %s: %v; assuming cgroup v1", unifiedMountpoint, err)
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

func readProcsFile(dir string) (out []int, _ error) {
	file := CgroupProcesses
	retry := true

again:
	f, err := OpenFile(dir, file, os.O_RDONLY)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	s := bufio.NewScanner(f)
	for s.Scan() {
		if t := s.Text(); t != "" {
			pid, err := strconv.Atoi(t)
			if err != nil {
				return nil, err
			}
			out = append(out, pid)
		}
	}
	if errors.Is(s.Err(), unix.ENOTSUP) && retry {
		// For a threaded cgroup, read returns ENOTSUP, and we should
		// read from cgroup.threads instead.
		file = "cgroup.threads"
		retry = false
		goto again
	}
	return out, s.Err()
}

// ParseCgroupFile parses the given cgroup file, typically /proc/self/cgroup
// or /proc/<pid>/cgroup, into a map of subsystems to cgroup paths, e.g.
//
//	"cpu": "/user.slice/user-1000.slice"
//	"pids": "/user.slice/user-1000.slice"
//
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

// rmdir tries to remove a directory, optionally retrying on EBUSY.
func rmdir(path string, retry bool) error {
	delay := time.Millisecond
	tries := 10

again:
	err := unix.Rmdir(path)
	switch err { // nolint:errorlint // unix errors are bare
	case nil, unix.ENOENT:
		return nil
	case unix.EINTR:
		goto again
	case unix.EBUSY:
		if retry && tries > 0 {
			time.Sleep(delay)
			delay *= 2
			tries--
			goto again

		}
	}
	return &os.PathError{Op: "rmdir", Path: path, Err: err}
}

// RemovePath aims to remove cgroup path. It does so recursively,
// by removing any subdirectories (sub-cgroups) first.
func RemovePath(path string) error {
	// Try the fast path first; don't retry on EBUSY yet.
	if err := rmdir(path, false); err == nil {
		return nil
	}

	// There are many reasons why rmdir can fail, including:
	// 1. cgroup have existing sub-cgroups;
	// 2. cgroup (still) have some processes (that are about to vanish);
	// 3. lack of permission (one example is read-only /sys/fs/cgroup mount,
	//    in which case rmdir returns EROFS even for for a non-existent path,
	//    see issue 4518).
	//
	// Using os.ReadDir here kills two birds with one stone: check if
	// the directory exists (handling scenario 3 above), and use
	// directory contents to remove sub-cgroups (handling scenario 1).
	infos, err := os.ReadDir(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	// Let's remove sub-cgroups, if any.
	for _, info := range infos {
		if info.IsDir() {
			if err = RemovePath(filepath.Join(path, info.Name())); err != nil {
				return err
			}
		}
	}
	// Finally, try rmdir again, this time with retries on EBUSY,
	// which may help with scenario 2 above.
	return rmdir(path, true)
}

// RemovePaths iterates over the provided paths removing them.
func RemovePaths(paths map[string]string) (err error) {
	for s, p := range paths {
		if err := RemovePath(p); err == nil {
			delete(paths, s)
		}
	}
	if len(paths) == 0 {
		clear(paths)
		return nil
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
		val, ok := strings.CutPrefix(file, "hugepages-")
		if !ok {
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
// is defined as memory+swap combined, while in cgroup v2 swap is a separate value,
// so we need to subtract memory from it where it makes sense.
func ConvertMemorySwapToCgroupV2Value(memorySwap, memory int64) (int64, error) {
	switch {
	case memory == -1 && memorySwap == 0:
		// For compatibility with cgroup1 controller, set swap to unlimited in
		// case the memory is set to unlimited and the swap is not explicitly set,
		// treating the request as "set both memory and swap to unlimited".
		return -1, nil
	case memorySwap == -1, memorySwap == 0:
		// Treat -1 ("max") and 0 ("unset") swap as is.
		return memorySwap, nil
	case memory == -1:
		// Unlimited memory, so treat swap as is.
		return memorySwap, nil
	case memory == 0:
		// Unset or unknown memory, can't calculate swap.
		return 0, errors.New("unable to set swap limit without memory limit")
	case memory < 0:
		// Does not make sense to subtract a negative value.
		return 0, fmt.Errorf("invalid memory value: %d", memory)
	case memorySwap < memory:
		// Sanity check.
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
