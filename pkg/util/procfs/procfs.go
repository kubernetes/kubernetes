/*
Copyright 2015 The Kubernetes Authors.

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
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"unicode"

	"github.com/golang/glog"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
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
		if os.IsNotExist(err) {
			return "", os.ErrNotExist
		}
		return "", err
	}
	return containerNameFromProcCgroup(string(content))
}

// Find process(es) using a regular expression and send a specified
// signal to each process
func PKill(name string, sig syscall.Signal) error {
	if len(name) == 0 {
		return fmt.Errorf("name should not be empty")
	}
	re, err := regexp.Compile(name)
	if err != nil {
		return err
	}
	pids := getPids(re)
	if len(pids) == 0 {
		return fmt.Errorf("unable to fetch pids for process name : %q", name)
	}
	errList := []error{}
	for _, pid := range pids {
		if err = syscall.Kill(pid, sig); err != nil {
			errList = append(errList, err)
		}
	}
	return utilerrors.NewAggregate(errList)
}

// Find process(es) with a specified name (exact match)
// and return their pid(s)
func PidOf(name string) ([]int, error) {
	if len(name) == 0 {
		return []int{}, fmt.Errorf("name should not be empty")
	}
	re, err := regexp.Compile("(^|/)" + name + "$")
	if err != nil {
		return []int{}, err
	}
	return getPids(re), nil
}

func getPids(re *regexp.Regexp) []int {
	pids := []int{}
	filepath.Walk("/proc", func(path string, info os.FileInfo, err error) error {
		if err != nil {
			// We should continue processing other directories/files
			return nil
		}
		base := filepath.Base(path)
		// Traverse only the directories we are interested in
		if info.IsDir() && path != "/proc" {
			// If the directory is not a number (i.e. not a PID), skip it
			if _, err := strconv.Atoi(base); err != nil {
				return filepath.SkipDir
			}
		}
		if base != "cmdline" {
			return nil
		}
		cmdline, err := ioutil.ReadFile(path)
		if err != nil {
			glog.V(4).Infof("Error reading file %s: %+v", path, err)
			return nil
		}
		// The bytes we read have '\0' as a separator for the command line
		parts := bytes.SplitN(cmdline, []byte{0}, 2)
		if len(parts) == 0 {
			return nil
		}
		// Split the command line itself we are interested in just the first part
		exe := strings.FieldsFunc(string(parts[0]), func(c rune) bool {
			return unicode.IsSpace(c) || c == ':'
		})
		if len(exe) == 0 {
			return nil
		}
		// Check if the name of the executable is what we are looking for
		if re.MatchString(exe[0]) {
			dirname := filepath.Base(filepath.Dir(path))
			// Grab the PID from the directory path
			pid, _ := strconv.Atoi(dirname)
			pids = append(pids, pid)
		}
		return nil
	})
	return pids
}
