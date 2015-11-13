/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package procfs

import (
	"fmt"
	"io/ioutil"
	"path"
	"strconv"
	"strings"
)

type ProcFS struct{}

func NewProcFS() ProcFSInterface {
	return &ProcFS{}
}

func containerNameFromProcCgroup(content string) (string, error) {
	lines := strings.Split(content, "\n")
	for _, line := range lines {
		entries := strings.SplitN(line, ":", 3)
		if len(entries) == 3 && entries[1] == "devices" {
			return strings.TrimSpace(entries[2]), nil
		}
	}
	return "", fmt.Errorf("could not find devices cgroup location")
}

// getFullContainerName gets the container name given the root process id of the container.
// Eg. If the devices cgroup for the container is stored in /sys/fs/cgroup/devices/docker/nginx,
// return docker/nginx. Assumes that the process is part of exactly one cgroup hierarchy.
func (pfs *ProcFS) GetFullContainerName(pid int) (string, error) {
	filePath := path.Join("/proc", strconv.Itoa(pid), "cgroup")
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		return "", err
	}
	return containerNameFromProcCgroup(string(content))
}
