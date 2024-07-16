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

package cgroup1

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/containerd/cgroups/v3"
	units "github.com/docker/go-units"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

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
	if !cgroups.RunningInUserNS() {
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
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	// We should only need 20 bytes for the max uint64, but for a nice power of 2
	// lets use 32.
	b := make([]byte, 32)
	n, err := f.Read(b)
	if err != nil {
		return 0, err
	}
	s := string(bytes.TrimSpace(b[:n]))
	if s == "max" {
		// Return 0 for the max value to maintain backward compatibility.
		return 0, nil
	}
	return parseUint(s, 10, 64)
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
//
//	"cpu": "/user.slice/user-1000.slice"
//	"pids": "/user.slice/user-1000.slice"
//
// etc.
//
// The resulting map does not have an element for cgroup v2 unified hierarchy.
// Use [cgroups.ParseCgroupFileUnified] to get the unified path.
func ParseCgroupFile(path string) (map[string]string, error) {
	x, _, err := ParseCgroupFileUnified(path)
	return x, err
}

// ParseCgroupFileUnified returns legacy subsystem paths as the first value,
// and returns the unified path as the second value.
//
// Deprecated: use [cgroups.ParseCgroupFileUnified] instead .
func ParseCgroupFileUnified(path string) (map[string]string, string, error) {
	return cgroups.ParseCgroupFileUnified(path)
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
