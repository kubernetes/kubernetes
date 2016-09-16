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

package util

import (
	"k8s.io/kubernetes/pkg/api"
)

// For each of these resources, a pod that doesn't request the resource explicitly
// will be treated as having requested the amount indicated below, for the purpose
// of computing priority only. This ensures that when scheduling zero-request pods, such
// pods will not all be scheduled to the machine with the smallest in-use request,
// and that when scheduling regular pods, such pods will not see zero-request pods as
// consuming no resources whatsoever. We chose these values to be similar to the
// resources that we give to cluster addon pods (#10653). But they are pretty arbitrary.
// As described in #11713, we use request instead of limit to deal with resource requirements.
const DefaultMilliCpuRequest int64 = 100             // 0.1 core
const DefaultMemoryRequest int64 = 200 * 1024 * 1024 // 200 MB

// GetNonzeroRequests returns the default resource request if none is found or what is provided on the request
// TODO: Consider setting default as a fixed fraction of machine capacity (take "capacity api.ResourceList"
// as an additional argument here) rather than using constants
func GetNonzeroRequests(requests *api.ResourceList) (int64, int64) {
	var outMilliCPU, outMemory int64
	// Override if un-set, but not if explicitly set to zero
	if _, found := (*requests)[api.ResourceCPU]; !found {
		outMilliCPU = DefaultMilliCpuRequest
	} else {
		outMilliCPU = requests.Cpu().MilliValue()
	}
	// Override if un-set, but not if explicitly set to zero
	if _, found := (*requests)[api.ResourceMemory]; !found {
		outMemory = DefaultMemoryRequest
	} else {
		outMemory = requests.Memory().Value()
	}
	return outMilliCPU, outMemory
}
