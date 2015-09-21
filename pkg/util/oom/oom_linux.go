// +build cgo,linux

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

package oom

import (
	"fmt"
	"io/ioutil"
	"path"
	"strconv"

	"github.com/docker/libcontainer/cgroups/fs"
	"github.com/docker/libcontainer/configs"
	"github.com/golang/glog"
)

func NewOomAdjuster() *OomAdjuster {
	oomAdjuster := &OomAdjuster{
		pidLister:        getPids,
		ApplyOomScoreAdj: applyOomScoreAdj,
	}
	oomAdjuster.ApplyOomScoreAdjContainer = oomAdjuster.applyOomScoreAdjContainer
	return oomAdjuster
}

func getPids(cgroupName string) ([]int, error) {
	fsManager := fs.Manager{
		Cgroups: &configs.Cgroup{
			Name: cgroupName,
		},
	}
	return fsManager.GetPids()
}

// Writes 'value' to /proc/<pid>/oom_score_adj. PID = 0 means self
func applyOomScoreAdj(pid int, oomScoreAdj int) error {
	if pid < 0 {
		return fmt.Errorf("invalid PID %d specified for oom_score_adj", pid)
	}

	var pidStr string
	if pid == 0 {
		pidStr = "self"
	} else {
		pidStr = strconv.Itoa(pid)
	}

	oomScoreAdjPath := path.Join("/proc", pidStr, "oom_score_adj")
	maxTries := 2
	var err error
	for i := 0; i < maxTries; i++ {
		_, readErr := ioutil.ReadFile(oomScoreAdjPath)
		if readErr != nil {
			err = fmt.Errorf("failed to read oom_score_adj: %v", readErr)
		} else if writeErr := ioutil.WriteFile(oomScoreAdjPath, []byte(strconv.Itoa(oomScoreAdj)), 0700); writeErr != nil {
			err = fmt.Errorf("failed to set oom_score_adj to %d: %v", oomScoreAdj, writeErr)
		} else {
			return nil
		}
	}

	return err
}

// Writes 'value' to /proc/<pid>/oom_score_adj for all processes in cgroup cgroupName.
// Keeps trying to write until the process list of the cgroup stabilizes, or until maxTries tries.
func (oomAdjuster *OomAdjuster) applyOomScoreAdjContainer(cgroupName string, oomScoreAdj, maxTries int) error {
	adjustedProcessSet := make(map[int]bool)
	for i := 0; i < maxTries; i++ {
		continueAdjusting := false
		pidList, err := oomAdjuster.pidLister(cgroupName)
		if err != nil {
			continueAdjusting = true
			glog.Errorf("Error getting process list for cgroup %s: %+v", cgroupName, err)
		} else if len(pidList) == 0 {
			continueAdjusting = true
		} else {
			for _, pid := range pidList {
				if !adjustedProcessSet[pid] {
					continueAdjusting = true
					if err = oomAdjuster.ApplyOomScoreAdj(pid, oomScoreAdj); err == nil {
						adjustedProcessSet[pid] = true
					}
				}
			}
		}
		if !continueAdjusting {
			return nil
		}
		// There's a slight race. A process might have forked just before we write its OOM score adjust.
		// The fork might copy the parent process's old OOM score, then this function might execute and
		// update the parent's OOM score, but the forked process id might not be reflected in cgroup.procs
		// for a short amount of time. So this function might return without changing the forked process's
		// OOM score. Very unlikely race, so ignoring this for now.
	}
	return fmt.Errorf("exceeded maxTries, some processes might not have desired OOM score")
}
