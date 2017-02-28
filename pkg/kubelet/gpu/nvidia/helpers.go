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

package nvidia

import "k8s.io/apimachinery/pkg/util/sets"

// podGPUs represents a list of pod to GPU mappings.
type podGPUs struct {
	podGPUMapping map[string]sets.String
}

func newPodGPUs() *podGPUs {
	return &podGPUs{
		podGPUMapping: map[string]sets.String{},
	}
}
func (pgpu *podGPUs) pods() sets.String {
	ret := sets.NewString()
	for k := range pgpu.podGPUMapping {
		ret.Insert(k)
	}
	return ret
}

func (pgpu *podGPUs) insert(podUID string, device string) {
	if _, exists := pgpu.podGPUMapping[podUID]; !exists {
		pgpu.podGPUMapping[podUID] = sets.NewString(device)
	} else {
		pgpu.podGPUMapping[podUID].Insert(device)
	}
}

func (pgpu *podGPUs) delete(pods []string) {
	for _, uid := range pods {
		delete(pgpu.podGPUMapping, uid)
	}
}

func (pgpu *podGPUs) devices() sets.String {
	ret := sets.NewString()
	for _, devices := range pgpu.podGPUMapping {
		ret = ret.Union(devices)
	}
	return ret
}
