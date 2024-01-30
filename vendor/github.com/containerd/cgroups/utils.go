/*
   Copyright The containerd Authors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

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
	"syscall"
	"time"

	units "github.com/docker/go-units"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"golang.org/x/sys/unix"
)

var (
	nsOnce    sync.Once
	inUserNS  bool
	checkMode sync.Once
	cgMode    CGMode
)

const unifiedMountpoint = "/sys/fs/cgroup"

// CGMode is the cgroups mode of the host system
type CGMode int

const (
	// Unavailable cgroup mountpoint
	Unavailable CGMode = iota
	// Legacy cgroups v1
	Legacy
	// Hybrid with cgroups v1 and v2 controllers mounted
	Hybrid
	// Unified with only cgroups v2 mounted
	Unified
)

// Mode returns the cgroups mode running on the host
func Mode() CGMode {
	checkMode.Do(func() {
		var st unix.Statfs_t
		if err := unix.Statfs(unifiedMountpoint, &st); err != nil {
			cgMode = Unavailable
			return
		}
		switch st.Type {
		case unix.CGROUP2_SUPER_MAGIC:
			cgMode = Unified
		default:
			cgMode = Legacy
			if err := unix.Statfs(filepath.Join(unifiedMountpoint, "unified"), &st); err != nil {
				return
			}
			if st.Type == unix.CGROUP2_SUPER_MAGIC {
				cgMode = Hybrid
			}
		}
	})
	return cgMode
}

// RunningInUserNS detects whether we are currently running in a user namespace.
// Copied from github.com/lxc/lxd/shared/util.go
func RunningInUserNS() bool {
	nsOnce.Do(func() {
		file, err := os.Open("/proc/self/uid_map")
		if err != nil {
			// This kernel-provided file only exists if user namespaces are supported
			return
		}
		defer file.Close()

		buf := bufio.NewReader(file)
		l, _, err := buf.ReadLine()
		if err != nil {
			return
		}

		line := string(l)
		var a, b, c int64
		fmt.Sscanf(line, "%d %d %d", &a, &b, &c)

		/*
		 * We assume we are in the initial user namespace if we have a full
		 * range - 4294967295 uids starting at uid 0.
		 */
		if a == 0 && b == 0 && c == 4294967295 {
			return
		}
		inUserNS = true
	})
	return inUserNS
}

// defaults returns all known groups
func defaults(root string) ([]Subsystem, error) {
	h, err := NewHugetlb(root)
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	s := []Subsystem{
		NewNamed(root, "systemd"),
		NewFreezer(root),
		NewPids(root),
		NewNetCls(root),
		NewNetPrio(root),
		NewPerfEvent(root),
		NewCpuset(root),
		NewCpu(root),
		NewCpuacct(root),
		NewMemory(root),
		NewBlkio(root),
		NewRdma(root),
	}
	// only add the devices cgroup if we are not in a user namespace
	// because modifications are not allowed
	if !RunningInUserNS() {
		s = append(s, NewDevices(root))
	}
	// add the hugetlb cgroup if error wasn't due to missing hugetlb
	// cgroup support on the host
	if err == nil {
		s = append(s, h)
	}
	return s, nil
}

// remove will remove a cgroup path handling EAGAIN and EBUSY errors and
// retrying the remove after a exp timeout
func remove(path string) error {
	delay := 10 * time.Millisecond
	for i := 0; i < 5; i++ {
		if i != 0 {
			time.Sleep(delay)
			delay *= 2
		}
		if err := os.RemoveAll(path); err == nil {
			return nil
		}
	}
	return fmt.Errorf("cgroups: unable to remove path %q", path)
}

// readPids will read all the pids of processes or tasks in a cgroup by the provided path
func readPids(path string, subsystem Name, pType procType) ([]Process, error) {
	f, err := os.Open(filepath.Join(path, pType))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var (
		out []Process
		s   = bufio.NewScanner(f)
	)
	for s.Scan() {
		if t := s.Text(); t != "" {
			pid, err := strconv.Atoi(t)
			if err != nil {
				return nil, err
			}
			out = append(out, Process{
				Pid:       pid,
				Subsystem: subsystem,
				Path:      path,
			})
		}
	}
	if err := s.Err(); err != nil {
		// failed to read all pids?
		return nil, err
	}
	return out, nil
}

func hugePageSizes() ([]string, error) {
	var (
		pageSizes []string
		sizeList  = []string{"B", "KB", "MB", "GB", "TB", "PB"}
	)
	files, err := os.ReadDir("/sys/kernel/mm/hugepages")
	if err != nil {
		return nil, err
	}
	for _, st := range files {
		nameArray := strings.Split(st.Name(), "-")
		pageSize, err := units.RAMInBytes(nameArray[1])
		if err != nil {
			return nil, err
		}
		pageSizes = append(pageSizes, units.CustomSize("%g%s", float64(pageSize), 1024.0, sizeList))
	}
	return pageSizes, nil
}

func readUint(path string) (uint64, error) {
	v, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	return parseUint(strings.TrimSpace(string(v)), 10, 64)
}

func parseUint(s string, base, bitSize int) (uint64, error) {
	v, err := strconv.ParseUint(s, base, bitSize)
	if err != nil {
		intValue, intErr := strconv.ParseInt(s, base, bitSize)
		// 1. Handle negative values greater than MinInt64 (and)
		// 2. Handle negative values lesser than MinInt64
		if intErr == nil && intValue < 0 {
			return 0, nil
		} else if intErr != nil &&
			intErr.(*strconv.NumError).Err == strconv.ErrRange &&
			intValue < 0 {
			return 0, nil
		}
		return 0, err
	}
	return v, nil
}

func parseKV(raw string) (string, uint64, error) {
	parts := strings.Fields(raw)
	switch len(parts) {
	case 2:
		v, err := parseUint(parts[1], 10, 64)
		if err != nil {
			return "", 0, err
		}
		return parts[0], v, nil
	default:
		return "", 0, ErrInvalidFormat
	}
}

// ParseCgroupFile parses the given cgroup file, typically /proc/self/cgroup
// or /proc/<pid>/cgroup, into a map of subsystems to cgroup paths, e.g.
//   "cpu": "/user.slice/user-1000.slice"
//   "pids": "/user.slice/user-1000.slice"
// etc.
//
// The resulting map does not have an element for cgroup v2 unified hierarchy.
// Use ParseCgroupFileUnified to get the unified path.
func ParseCgroupFile(path string) (map[string]string, error) {
	x, _, err := ParseCgroupFileUnified(path)
	return x, err
}

// ParseCgroupFileUnified returns legacy subsystem paths as the first value,
// and returns the unified path as the second value.
func ParseCgroupFileUnified(path string) (map[string]string, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()
	return parseCgroupFromReaderUnified(f)
}

func parseCgroupFromReaderUnified(r io.Reader) (map[string]string, string, error) {
	var (
		cgroups = make(map[string]string)
		unified = ""
		s       = bufio.NewScanner(r)
	)
	for s.Scan() {
		var (
			text  = s.Text()
			parts = strings.SplitN(text, ":", 3)
		)
		if len(parts) < 3 {
			return nil, unified, fmt.Errorf("invalid cgroup entry: %q", text)
		}
		for _, subs := range strings.Split(parts[1], ",") {
			if subs == "" {
				unified = parts[2]
			} else {
				cgroups[subs] = parts[2]
			}
		}
	}
	if err := s.Err(); err != nil {
		return nil, unified, err
	}
	return cgroups, unified, nil
}

func getCgroupDestination(subsystem string) (string, error) {
	f, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		return "", err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		fields := strings.Split(s.Text(), " ")
		if len(fields) < 10 {
			// broken mountinfo?
			continue
		}
		if fields[len(fields)-3] != "cgroup" {
			continue
		}
		for _, opt := range strings.Split(fields[len(fields)-1], ",") {
			if opt == subsystem {
				return fields[3], nil
			}
		}
	}
	if err := s.Err(); err != nil {
		return "", err
	}
	return "", ErrNoCgroupMountDestination
}

func pathers(subystems []Subsystem) []pather {
	var out []pather
	for _, s := range subystems {
		if p, ok := s.(pather); ok {
			out = append(out, p)
		}
	}
	return out
}

func initializeSubsystem(s Subsystem, path Path, resources *specs.LinuxResources) error {
	if c, ok := s.(creator); ok {
		p, err := path(s.Name())
		if err != nil {
			return err
		}
		if err := c.Create(p, resources); err != nil {
			return err
		}
	} else if c, ok := s.(pather); ok {
		p, err := path(s.Name())
		if err != nil {
			return err
		}
		// do the default create if the group does not have a custom one
		if err := os.MkdirAll(c.Path(p), defaultDirPerm); err != nil {
			return err
		}
	}
	return nil
}

func cleanPath(path string) string {
	if path == "" {
		return ""
	}
	path = filepath.Clean(path)
	if !filepath.IsAbs(path) {
		path, _ = filepath.Rel(string(os.PathSeparator), filepath.Clean(string(os.PathSeparator)+path))
	}
	return path
}

func retryingWriteFile(path string, data []byte, mode os.FileMode) error {
	// Retry writes on EINTR; see:
	//    https://github.com/golang/go/issues/38033
	for {
		err := os.WriteFile(path, data, mode)
		if err == nil {
			return nil
		} else if !errors.Is(err, syscall.EINTR) {
			return err
		}
	}
}
