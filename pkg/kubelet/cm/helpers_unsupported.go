// +build !linux

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

package cm

import "k8s.io/kubernetes/pkg/api"

const (
	MinShares     = 0
	SharesPerCPU  = 0
	MilliCPUToCPU = 0

	QuotaPeriod    = 0
	MinQuotaPeriod = 0
)

// MilliCPUToQuota converts milliCPU to CFS quota and period values.
func MilliCPUToQuota(milliCPU int64) (int64, int64) {
	return 0, 0
}

// MilliCPUToShares converts the milliCPU to CFS shares.
func MilliCPUToShares(milliCPU int64) int64 {
	return 0
}

// ResourceConfigForPod takes the input pod and outputs the cgroup resource config.
func ResourceConfigForPod(pod *api.Pod) *ResourceConfig {
	return nil
}

// GetCgroupSubsystems returns information about the mounted cgroup subsystems
func GetCgroupSubsystems() (*CgroupSubsystems, error) {
	return nil, nil
}

func getCgroupProcs(dir string) ([]int, error) {
	return nil, nil
}
