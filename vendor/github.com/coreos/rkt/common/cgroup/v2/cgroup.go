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

//+build linux

package v2

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/hashicorp/errwrap"
)

// GetEnabledControllers returns a list of enabled cgroup controllers
func GetEnabledControllers() ([]string, error) {
	controllersFile, err := os.Open("/sys/fs/cgroup/cgroup.controllers")
	if err != nil {
		return nil, err
	}
	defer controllersFile.Close()

	sc := bufio.NewScanner(controllersFile)

	sc.Scan()
	if err := sc.Err(); err != nil {
		return nil, err
	}

	return strings.Split(sc.Text(), " "), nil
}

func parseProcCgroupInfo(procCgroupInfoPath string) (string, error) {
	cg, err := os.Open(procCgroupInfoPath)
	if err != nil {
		return "", errwrap.Wrap(errors.New("error opening /proc/self/cgroup"), err)
	}
	defer cg.Close()

	s := bufio.NewScanner(cg)
	s.Scan()
	parts := strings.SplitN(s.Text(), ":", 3)
	if len(parts) < 3 {
		return "", fmt.Errorf("error parsing /proc/self/cgroup")
	}

	return parts[2], nil
}

// GetOwnCgroupPath returns the cgroup path of this process
func GetOwnCgroupPath() (string, error) {
	return parseProcCgroupInfo("/proc/self/cgroup")
}

// GetCgroupPathByPid returns the cgroup path of the process
func GetCgroupPathByPid(pid int) (string, error) {
	return parseProcCgroupInfo(fmt.Sprintf("/proc/%d/cgroup", pid))
}
