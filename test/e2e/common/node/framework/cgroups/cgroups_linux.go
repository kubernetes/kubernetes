/*
Copyright 2025 The Kubernetes Authors.

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

package cgroups

import (
	"strconv"

	libcontainercgroups "github.com/opencontainers/cgroups"
	"k8s.io/apimachinery/pkg/api/resource"
	kubecm "k8s.io/kubernetes/pkg/kubelet/cm"
)

func getExpectedCPUShares(cpuRequest, cpuLimit *resource.Quantity, podOnCgroupv2 bool) []string {
	// This function is moved out from cgroups.go because opencontainers/cgroups can only be compiled in linux platforms.
	var shares int64
	if cpuRequest.IsZero() && !cpuLimit.IsZero() {
		shares = int64(kubecm.MilliCPUToShares(cpuLimit.MilliValue()))
	} else {
		shares = int64(kubecm.MilliCPUToShares(cpuRequest.MilliValue()))
	}
	if podOnCgroupv2 {
		// Because of https://github.com/kubernetes/kubernetes/issues/131216, the way of conversion has been changed.
		// runc: https://github.com/opencontainers/runc/pull/4785
		// crun: https://github.com/containers/crun/issues/1721
		// This is dependent on the container runtime version. In order not to break the tests when we upgrade the
		// container runtimes, we check if either the old or the new conversion matches the actual value for now.
		// TODO: Remove the old conversion once container runtimes are updated.
		oldConverted := 1 + ((shares-2)*9999)/262142
		converted := libcontainercgroups.ConvertCPUSharesToCgroupV2Value(uint64(shares))
		return []string{strconv.FormatInt(oldConverted, 10), strconv.FormatInt(int64(converted), 10)}
	} else {
		return []string{strconv.FormatInt(shares, 10)}
	}
}
