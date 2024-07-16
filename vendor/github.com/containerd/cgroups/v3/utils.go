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
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"

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

// ParseCgroupFileUnified returns legacy subsystem paths as the first value,
// and returns the unified path as the second value.
func ParseCgroupFileUnified(path string) (map[string]string, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()
	return ParseCgroupFromReaderUnified(f)
}

// ParseCgroupFromReaderUnified returns legacy subsystem paths as the first value,
// and returns the unified path as the second value.
func ParseCgroupFromReaderUnified(r io.Reader) (map[string]string, string, error) {
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
