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

// defaultPodLimitsForDownwardApi copies the input pod, and optional container,
// and applies default resource limits. it returns a copy of the input pod,
// and a copy of the input container (if specified) with default limits
// applied. if a container has no limit specified, it will default the limit to
// the node capacity.
// TODO: if/when we have pod level resources, we need to update this function
// to use those limits instead of node capacity.
func (kl *Kubelet) defaultPodLimitsForDownwardApi(pod *api.Pod, container *api.Container) (*api.Pod, *api.Container, error) {
	if pod == nil {
		return nil, nil, fmt.Errorf("invalid input, pod cannot be nil")
	}

	node, err := kl.getNodeAnyWay()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to find node object, expected a node")
	}
	capacity := node.Status.Capacity

	podCopy, err := api.Scheme.Copy(pod)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to perform a deep copy of pod object: %v", err)
	}
	outputPod, ok := podCopy.(*api.Pod)
	if !ok {
		return nil, nil, fmt.Errorf("unexpected type returned from deep copy of pod object")
	}
	for idx := range outputPod.Spec.Containers {
		mergeContainerResourceLimitsWithCapacity(&outputPod.Spec.Containers[idx], capacity)
	}

	var outputContainer *api.Container
	if container != nil {
		containerCopy, err := api.Scheme.DeepCopy(container)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to perform a deep copy of container object: %v", err)
		}
		outputContainer, ok = containerCopy.(*api.Container)
		if !ok {
			return nil, nil, fmt.Errorf("unexpected type returned from deep copy of container object")
		}
		mergeContainerResourceLimitsWithCapacity(outputContainer, capacity)
	}
	return outputPod, outputContainer, nil
}

// mergeContainerResourceLimitsWithCapacity checks if a limit is applied for
// the container, and if not, it sets the limit based on the capacity.
func mergeContainerResourceLimitsWithCapacity(container *api.Container,
	capacity api.ResourceList) {
	if container.Resources.Limits == nil {
		container.Resources.Limits = make(api.ResourceList)
	}
	for _, resource := range []api.ResourceName{api.ResourceCPU, api.ResourceMemory} {
		if quantity, exists := container.Resources.Limits[resource]; !exists || quantity.IsZero() {
			if cap, exists := capacity[resource]; exists {
				container.Resources.Limits[resource] = *cap.Copy()
			}
		}
	}
}
