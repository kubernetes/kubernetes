// +build cgo,linux

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

package oom

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"time"

	cmutil "k8s.io/kubernetes/pkg/kubelet/cm/util"

	"github.com/golang/glog"
)

func NewOOMAdjuster() *OOMAdjuster {
	oomAdjuster := &OOMAdjuster{
		pidLister:        getPids,
		ApplyOOMScoreAdj: applyOOMScoreAdj,
	}
	oomAdjuster.ApplyOOMScoreAdjContainer = oomAdjuster.applyOOMScoreAdjContainer
	return oomAdjuster
}

func getPids(cgroupName string) ([]int, error) {
	return cmutil.GetPids(filepath.Join("/", cgroupName))
}

// Writes 'value' to /proc/<pid>/oom_score_adj. PID = 0 means self
// Returns os.ErrNotExist if the `pid` does not exist.
func applyOOMScoreAdj(pid int, oomScoreAdj int) error {
	if pid < 0 {
		return fmt.Errorf("invalid PID %d specified for oom_score_adj", pid)
	}

	var pidStr string
	if pid == 0 {
		pidStr = "self"
	} else {
		pidStr = strconv.Itoa(pid)
	}

	maxTries := 2
	oomScoreAdjPath := path.Join("/proc", pidStr, "oom_score_adj")
	value := strconv.Itoa(oomScoreAdj)
	glog.V(4).Infof("attempting to set %q to %q", oomScoreAdjPath, value)
	var err error
	for i := 0; i < maxTries; i++ {
		err = ioutil.WriteFile(oomScoreAdjPath, []byte(value), 0700)
		if err != nil {
			if os.IsNotExist(err) {
				glog.V(2).Infof("%q does not exist", oomScoreAdjPath)
				return os.ErrNotExist
			}

			glog.V(3).Info(err)
			time.Sleep(100 * time.Millisecond)
			continue
		}
		return nil
	}
	if err != nil {
		glog.V(2).Infof("failed to set %q to %q: %v", oomScoreAdjPath, value, err)
	}
	return err
}

// Writes 'value' to /proc/<pid>/oom_score_adj for all processes in cgroup cgroupName.
// Keeps trying to write until the process list of the cgroup stabilizes, or until maxTries tries.
func (oomAdjuster *OOMAdjuster) applyOOMScoreAdjContainer(cgroupName string, oomScoreAdj, maxTries int) error {
	adjustedProcessSet := make(map[int]bool)
	for i := 0; i < maxTries; i++ {
		continueAdjusting := false
		pidList, err := oomAdjuster.pidLister(cgroupName)
		if err != nil {
			if os.IsNotExist(err) {
				// Nothing to do since the container doesn't exist anymore.
				return os.ErrNotExist
			}
			continueAdjusting = true
			glog.V(10).Infof("Error getting process list for cgroup %s: %+v", cgroupName, err)
		} else if len(pidList) == 0 {
			glog.V(10).Infof("Pid list is empty")
			continueAdjusting = true
		} else {
			for _, pid := range pidList {
				if !adjustedProcessSet[pid] {
					glog.V(10).Infof("pid %d needs to be set", pid)
					if err = oomAdjuster.ApplyOOMScoreAdj(pid, oomScoreAdj); err == nil {
						adjustedProcessSet[pid] = true
					} else if err == os.ErrNotExist {
						continue
					} else {
						glog.V(10).Infof("cannot adjust oom score for pid %d - %v", pid, err)
						continueAdjusting = true
					}
					// Processes can come and go while we try to apply oom score adjust value. So ignore errors here.
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
