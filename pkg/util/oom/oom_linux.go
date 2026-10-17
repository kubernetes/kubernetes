//go:build linux

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
	"os"
	"path"
	"strconv"
	"time"

	"k8s.io/klog/v2"
)

func NewOOMAdjuster() *OOMAdjuster {
	return &OOMAdjuster{
		ApplyOOMScoreAdj: applyOOMScoreAdj,
	}
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
	klog.V(4).Infof("attempting to set %q to %q", oomScoreAdjPath, value)
	var err error
	for i := 0; i < maxTries; i++ {
		err = os.WriteFile(oomScoreAdjPath, []byte(value), 0700)
		if err != nil {
			if os.IsNotExist(err) {
				klog.V(2).Infof("%q does not exist", oomScoreAdjPath)
				return os.ErrNotExist
			}

			klog.V(3).Info(err)
			time.Sleep(100 * time.Millisecond)
			continue
		}
		return nil
	}
	if err != nil {
		klog.V(2).Infof("failed to set %q to %q: %v", oomScoreAdjPath, value, err)
	}
	return err
}
