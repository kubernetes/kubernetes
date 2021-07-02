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
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/opencontainers/runc/libcontainer/configs"
	libcontainerUtils "github.com/opencontainers/runc/libcontainer/utils"
	"github.com/pkg/errors"
)

const UnifiedMountpoint = "/sys/fs/cgroup"

func defaultDirPath(c *configs.Cgroup) (string, error) {
	if (c.Name != "" || c.Parent != "") && c.Path != "" {
		return "", errors.Errorf("cgroup: either Path or Name and Parent should be used, got %+v", c)
	}
	if len(c.Paths) != 0 {
		// never set by specconv
		return "", errors.Errorf("cgroup: Paths is unsupported, use Path, got %+v", c)
	}

	// XXX: Do not remove this code. Path safety is important! -- cyphar
	cgPath := libcontainerUtils.CleanPath(c.Path)
	cgParent := libcontainerUtils.CleanPath(c.Parent)
	cgName := libcontainerUtils.CleanPath(c.Name)

	return _defaultDirPath(UnifiedMountpoint, cgPath, cgParent, cgName)
}

func _defaultDirPath(root, cgPath, cgParent, cgName string) (string, error) {
	if (cgName != "" || cgParent != "") && cgPath != "" {
		return "", errors.New("cgroup: either Path or Name and Parent should be used")
	}
	innerPath := cgPath
	if innerPath == "" {
		innerPath = filepath.Join(cgParent, cgName)
	}
	if filepath.IsAbs(innerPath) {
		return filepath.Join(root, innerPath), nil
	}

	ownCgroup, err := parseCgroupFile("/proc/self/cgroup")
	if err != nil {
		return "", err
	}
	// The current user scope most probably has tasks in it already,
	// making it impossible to enable controllers for its sub-cgroup.
	// A parent cgroup (with no tasks in it) is what we need.
	ownCgroup = filepath.Dir(ownCgroup)

	return filepath.Join(root, ownCgroup, innerPath), nil
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
		var (
			text  = s.Text()
			parts = strings.SplitN(text, ":", 3)
		)
		if len(parts) < 3 {
			return "", errors.Errorf("invalid cgroup entry: %q", text)
		}
		// text is like "0::/user.slice/user-1001.slice/session-1.scope"
		if parts[0] == "0" && parts[1] == "" {
			return parts[2], nil
		}
	}
	if err := s.Err(); err != nil {
		return "", err
	}
	return "", errors.New("cgroup path not found")
}
