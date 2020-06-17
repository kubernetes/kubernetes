// +build !linux

/*
Copyright 2018 The Kubernetes Authors.

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

package kuberuntime

// milliCPUToShares converts milliCPU to CPU shares
func milliCPUToShares(milliCPU int64) int64 {
	return 0
}

// sharesToMilliCPU converts CpuShares (cpu.shares) to milli-CPU value
func sharesToMilliCPU(shares int64) int64 {
	return 0
}

// quotaToMilliCPU converts cpu.cfs_quota_us and cpu.cfs_period_us to milli-CPU value
func quotaToMilliCPU(quota int64, period int64) int64 {
	return 0
}
