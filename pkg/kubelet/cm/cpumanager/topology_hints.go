/*
Copyright 2019 The Kubernetes Authors.

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

package cpumanager

import (
	"k8s.io/api/core/v1"
	"k8s.io/klog"

	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
)

func (m *manager) GetTopologyHints(pod v1.Pod, container v1.Container) []topologymanager.TopologyHint {
	var cpuHints []topologymanager.TopologyHint
	if requestedObj, ok := container.Resources.Requests[v1.ResourceCPU]; ok {
		// Get a count of how many CPUs have been requested
		requested := int(requestedObj.Value())
		klog.Infof("[cpumanager] Guaranteed CPUs detected: %v", requested)

		// Discover topology in order to establish the number
		// of available CPUs per socket.
		_, err := topology.Discover(m.machineInfo)
		if err != nil {
			klog.Infof("[cpu manager] error discovering topology")
			return nil
		}

		// TODO: Fill in cpuHints with proper TopologyHints
	}
	klog.Infof("[cpumanager] Topology Hints for pod: %v", cpuHints)
	return cpuHints
}
