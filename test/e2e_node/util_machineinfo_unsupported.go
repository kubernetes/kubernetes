//go:build !linux

/*
Copyright 2017 The Kubernetes Authors.

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

package e2enode

import (
	"errors"

	"k8s.io/utils/cpuset"
)

// IsCgroup2UnifiedMode returns whether we are running in cgroup v2 unified mode.
func IsCgroup2UnifiedMode() bool {
	return false
}

func isHTEnabled() bool {
	return false
}

func isMultiNUMA() bool {
	return false
}

func getUncoreCPUGroupSize() int {
	return 1
}

func getCPUSiblingList(cpuRes int64) string {
	return ""
}

func getCoreSiblingList(cpuRes int64) string {
	return ""
}

// getNumaNodeCPUs retrieves CPUs for each NUMA node.
func getNumaNodeCPUs() (map[int]cpuset.CPUSet, error) {
	return nil, errors.New("not implemented")
}
