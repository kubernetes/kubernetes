// +build cgo,darwin

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
)




// GetNodeTime gets node boot time and current time.
func GetNodeTime(initTime time.Time) (time.Time, time.Time, error) {
	// Get node current time.
	nodeTime := time.Now()

	// TODO (aescobar): init time is a poor approximation for uptime, but darwin has little choice.
	// Syscall is not supported on darwin so this change was necessary.
	// Get node boot time. NOTE that because we get node current time before uptime, the boot time
	// calculated will be a little earlier than the real boot time. This won't affect the correctness
	// of the test result.
	bootTime := nodeTime.Add(-time.Duration(initTime.Unix()) * time.Second)

	return nodeTime, bootTime, nil
}

