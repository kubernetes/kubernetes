// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package mountinfo

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/hashicorp/errwrap"
)

// HasPrefix returns a FilterFunc which returns true if
// the mountpoint of a given mount has prefix p, else false.
func HasPrefix(p string) FilterFunc {
	return FilterFunc(func(m *Mount) bool {
		return strings.HasPrefix(m.MountPoint, p)
	})
}

// ParseMounts returns all mountpoints associated with a process mount namespace.
// The special value 0 as pid argument is used to specify the current process.
func ParseMounts(pid uint) (Mounts, error) {
	var procPath string
	if pid == 0 {
		procPath = "/proc/self/mountinfo"
	} else {
		procPath = fmt.Sprintf("/proc/%d/mountinfo", pid)
	}

	mi, err := os.Open(procPath)
	if err != nil {
		return nil, err
	}
	defer mi.Close()

	return parseMountinfo(mi)
}

// parseMountinfo parses mi (/proc/<pid>/mountinfo) and returns mounts information
// according to https://www.kernel.org/doc/Documentation/filesystems/proc.txt
func parseMountinfo(mi io.Reader) (Mounts, error) {
	var podMounts Mounts
	sc := bufio.NewScanner(mi)
	var (
		mountID    int
		parentID   int
		major      int
		minor      int
		root       string
		mountPoint string
		opt        map[string]struct{}
	)

	for sc.Scan() {
		line := sc.Text()
		columns := strings.Split(line, " ")
		if len(columns) < 7 {
			return nil, fmt.Errorf("Not enough fields from line %q: %+v", line, columns)
		}

		opt = map[string]struct{}{}
		for i, col := range columns {
			if col == "-" {
				// separator: a single hyphen "-" marks the end of "optional fields"
				break
			}
			var err error
			switch i {
			case 0:
				mountID, err = strconv.Atoi(col)
			case 1:
				parentID, err = strconv.Atoi(col)
			case 2:
				split := strings.Split(col, ":")
				if len(split) != 2 {
					err = fmt.Errorf("found unexpected key:value field with more than two colons: %s", col)
					break
				}
				major, err = strconv.Atoi(split[0])
				if err != nil {
					break
				}
				minor, err = strconv.Atoi(split[1])
				if err != nil {
					break
				}
			case 3:
				root = col
			case 4:
				mountPoint = col
			default:
				split := strings.Split(col, ":")
				switch len(split) {
				case 1:
					// we ignore modes like rw, relatime, etc.
				case 2:
					opt[split[0]] = struct{}{}
				default:
					err = fmt.Errorf("found unexpected key:value field with more than two colons: %s", col)
				}
			}
			if err != nil {
				return nil, errwrap.Wrap(fmt.Errorf("could not parse mountinfo line %q", line), err)
			}
		}

		mnt := &Mount{
			ID:         mountID,
			Parent:     parentID,
			Major:      major,
			Minor:      minor,
			Root:       root,
			MountPoint: mountPoint,
			Opts:       opt,
		}
		podMounts = append(podMounts, mnt)
	}
	if err := sc.Err(); err != nil {
		return nil, errwrap.Wrap(errors.New("problem parsing mountinfo"), err)
	}
	sort.Sort(podMounts)
	return podMounts, nil
}
