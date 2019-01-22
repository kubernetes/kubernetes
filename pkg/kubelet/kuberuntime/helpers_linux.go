// +build linux

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

import (
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	kubefeatures "k8s.io/kubernetes/pkg/features"
)

const (
	// Taken from lmctfy https://github.com/google/lmctfy/blob/master/lmctfy/controllers/cpu_controller.cc
	minShares     = 2
	sharesPerCPU  = 1024
	milliCPUToCPU = 1000

	// 100000 is equivalent to 100ms
	quotaPeriod    = 100000
	minQuotaPeriod = 1000
)

// milliCPUToShares converts milliCPU to CPU shares
func milliCPUToShares(milliCPU int64) int64 {
	if milliCPU == 0 {
		// Return 2 here to really match kernel default for zero milliCPU.
		return minShares
	}
	// Conceptually (milliCPU / milliCPUToCPU) * sharesPerCPU, but factored to improve rounding.
	shares := (milliCPU * sharesPerCPU) / milliCPUToCPU
	if shares < minShares {
		return minShares
	}
	return shares
}

// milliCPUToQuota converts milliCPU to CFS quota and period values
func milliCPUToQuota(milliCPU int64, period int64) (quota int64) {
	// CFS quota is measured in two values:
	//  - cfs_period_us=100ms (the amount of time to measure usage across)
	//  - cfs_quota=20ms (the amount of cpu time allowed to be used across a period)
	// so in the above example, you are limited to 20% of a single CPU
	// for multi-cpu environments, you just scale equivalent amounts
	// see https://www.kernel.org/doc/Documentation/scheduler/sched-bwc.txt for details
	if milliCPU == 0 {
		return
	}
	if !utilfeature.DefaultFeatureGate.Enabled(kubefeatures.CPUCFSQuotaPeriod) {
		period = quotaPeriod
	}

	// we then convert your milliCPU to a value normalized over a period
	quota = (milliCPU * period) / milliCPUToCPU

	// quota needs to be a minimum of 1ms.
	if quota < minQuotaPeriod {
		quota = minQuotaPeriod
	}

	return
}
