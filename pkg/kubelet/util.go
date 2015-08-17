/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/securitycontext"
	cadvisorApi "github.com/google/cadvisor/info/v1"
)

func CapacityFromMachineInfo(info *cadvisorApi.MachineInfo) api.ResourceList {
	c := api.ResourceList{
		api.ResourceCPU: *resource.NewMilliQuantity(
			int64(info.NumCores*1000),
			resource.DecimalSI),
		api.ResourceMemory: *resource.NewQuantity(
			info.MemoryCapacity,
			resource.BinarySI),
	}
	return c
}

// Check whether we have the capabilities to run the specified pod.
func canRunPod(pod *api.Pod) error {
	if pod.Spec.HostNetwork {
		allowed, err := allowHostNetwork(pod)
		if err != nil {
			return err
		}
		if !allowed {
			return fmt.Errorf("pod with UID %q specified host networking, but is disallowed", pod.UID)
		}
	}

	if !capabilities.Get().AllowPrivileged {
		for _, container := range pod.Spec.Containers {
			if securitycontext.HasPrivilegedRequest(&container) {
				return fmt.Errorf("pod with UID %q specified privileged container, but is disallowed", pod.UID)
			}
		}
	}
	return nil
}

// Determined whether the specified pod is allowed to use host networking
func allowHostNetwork(pod *api.Pod) (bool, error) {
	podSource, err := getPodSource(pod)
	if err != nil {
		return false, err
	}
	for _, source := range capabilities.Get().HostNetworkSources {
		if source == podSource {
			return true, nil
		}
	}
	return false, nil
}
