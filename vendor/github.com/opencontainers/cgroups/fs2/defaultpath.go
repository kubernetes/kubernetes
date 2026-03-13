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

package fs2

import (
	"bufio"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/opencontainers/cgroups"
	"github.com/opencontainers/cgroups/internal/path"
)

const UnifiedMountpoint = "/sys/fs/cgroup"

func defaultDirPath(c *cgroups.Cgroup) (string, error) {
	innerPath, err := path.Inner(c)
	if err != nil {
		return "", err
	}

	if filepath.IsAbs(innerPath) {
		return filepath.Join(UnifiedMountpoint, innerPath), nil
	}

	// we don't need to use /proc/thread-self here because runc always runs
	// with every thread in the same cgroup. This lets us avoid having to do
	// runtime.LockOSThread.
	ownCgroup, err := parseCgroupFile("/proc/self/cgroup")
	if err != nil {
		return "", err
	}
	// The current user scope most probably has tasks in it already,
	// making it impossible to enable controllers for its sub-cgroup.
	// A parent cgroup (with no tasks in it) is what we need.
	ownCgroup = filepath.Dir(ownCgroup)

	return filepath.Join(UnifiedMountpoint, ownCgroup, innerPath), nil
}

// parseCgroupFile parses /proc/PID/cgroup file and return string
func parseCgroupFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	return parseCgroupFromReader(f)
}

func parseCgroupFromReader(r io.Reader) (string, error) {
	s := bufio.NewScanner(r)
	for s.Scan() {
		// "0::/user.slice/user-1001.slice/session-1.scope"
		if path, ok := strings.CutPrefix(s.Text(), "0::"); ok {
			return path, nil
		}
	}
	if err := s.Err(); err != nil {
		return "", err
	}
	return "", errors.New("cgroup path not found")
}
