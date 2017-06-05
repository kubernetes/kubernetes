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

type containerToGPU map[string]sets.String

// podGPUs represents a list of pod to GPU mappings.
type podGPUs struct {
	podGPUMapping map[string]containerToGPU
}

func newPodGPUs() *podGPUs {
	return &podGPUs{
		podGPUMapping: make(map[string]containerToGPU),
	}
}
func (pgpu *podGPUs) pods() sets.String {
	ret := sets.NewString()
	for k := range pgpu.podGPUMapping {
		ret.Insert(k)
	}
	return ret
}

func (pgpu *podGPUs) insert(podUID, contName string, device string) {
	if _, exists := pgpu.podGPUMapping[podUID]; !exists {
		pgpu.podGPUMapping[podUID] = make(containerToGPU)
	}
	if _, exists := pgpu.podGPUMapping[podUID][contName]; !exists {
		pgpu.podGPUMapping[podUID][contName] = sets.NewString()
	}
	pgpu.podGPUMapping[podUID][contName].Insert(device)
}

func (pgpu *podGPUs) getGPUs(podUID, contName string) sets.String {
	containers, exists := pgpu.podGPUMapping[podUID]
	if !exists {
		return nil
	}
	devices, exists := containers[contName]
	if !exists {
		return nil
	}
	return devices
}

func (pgpu *podGPUs) delete(pods []string) {
	for _, uid := range pods {
		delete(pgpu.podGPUMapping, uid)
	}
}

func (pgpu *podGPUs) devices() sets.String {
	ret := sets.NewString()
	for _, containerToGPU := range pgpu.podGPUMapping {
		for _, deviceSet := range containerToGPU {
			ret = ret.Union(deviceSet)
		}
	}
	return ret
}
