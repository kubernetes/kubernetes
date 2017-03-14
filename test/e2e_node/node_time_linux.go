// +build cgo,linux

/*
Copyright 2016 The Kubernetes Authors.

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

package e2e_node

import (
	"time"
	"syscall"
)




// GetNodeTime gets node boot time and current time.
func GetNodeTime(initTime time.Time) (time.Time, time.Time, error) {
	// Get node current time.
	nodeTime := time.Now()

	// Get system uptime.
	var info syscall.Sysinfo_t
	if err := syscall.Sysinfo(&info); err != nil {
		return time.Time{}, time.Time{}, err
	}
	// Get node boot time. NOTE that because we get node current time before uptime, the boot time
	// calculated will be a little earlier than the real boot time. This won't affect the correctness
	// of the test result.
	bootTime := nodeTime.Add(-time.Duration(info.Uptime) * time.Second)

	return nodeTime, bootTime, nil
}


