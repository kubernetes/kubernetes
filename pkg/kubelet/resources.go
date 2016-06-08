/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package kubelet

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
)

func (kl *Kubelet) defaultPodLimitsForDownwardApi(pod *api.Pod) (*api.Pod, error) {
	capacity := make(api.ResourceList)
	lastUpdatedNodeObject := kl.lastUpdatedNodeObject.Load()
	if lastUpdatedNodeObject == nil {
		return nil, fmt.Errorf("Failed to find node object in cache. Expected a non-nil object in the cache.")
	} else {
		capacity = lastUpdatedNodeObject.(*api.Node).Status.Capacity
	}
	podCopy, err := api.Scheme.Copy(pod)
	if err != nil {
		return nil, fmt.Errorf("failed to perform a deep copy of pod object. Error: %v", err)
	}
	pod = podCopy.(*api.Pod)
	for idx, c := range pod.Spec.Containers {
		for _, resource := range []api.ResourceName{api.ResourceCPU, api.ResourceMemory} {
			if quantity, exists := c.Resources.Limits[resource]; !exists || quantity.IsZero() {
				if cap, exists := capacity[resource]; exists {
					if pod.Spec.Containers[idx].Resources.Limits == nil {
						pod.Spec.Containers[idx].Resources.Limits = make(api.ResourceList)
					}
					pod.Spec.Containers[idx].Resources.Limits[resource] = cap
				}

			}
		}
	}
	return pod, nil
}
