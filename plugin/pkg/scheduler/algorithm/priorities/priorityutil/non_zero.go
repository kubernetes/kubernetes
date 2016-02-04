/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package priorityutil

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
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

// TODO: Consider setting default as a fixed fraction of machine capacity (take "capacity api.ResourceList"
// as an additional argument here) rather than using constants
func GetNonzeroRequests(requests *api.ResourceList) (int64, int64) {
	var out_millicpu, out_memory int64
	// Override if un-set, but not if explicitly set to zero
	if (*requests.Cpu() == resource.Quantity{}) {
		out_millicpu = DefaultMilliCpuRequest
	} else {
		out_millicpu = requests.Cpu().MilliValue()
	}
	// Override if un-set, but not if explicitly set to zero
	if (*requests.Memory() == resource.Quantity{}) {
		out_memory = DefaultMemoryRequest
	} else {
		out_memory = requests.Memory().Value()
	}
	return out_millicpu, out_memory
}
