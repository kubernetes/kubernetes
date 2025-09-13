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

package inplacepodresize

import (
	"context"

	v1 "k8s.io/api/core/v1"
	nodecapabilitieslib "k8s.io/component-helpers/nodecapabilities"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/features"
	nodecapabilitiesregistry "k8s.io/kubernetes/pkg/features/nodecapabilities"
)

// guaranteedCPUResizeInferrer is a struct that implements the PodUpdateInferrer interface.
type guaranteedCPUResizeInferrer struct{}

// Infer inspects a pod update and determines if it requires the GuaranteedQoSPodCPUResize capability.
func (i *guaranteedCPUResizeInferrer) Infer(ctx context.Context, oldPod, newPod *v1.Pod) *nodecapabilitieslib.CapabilityRequirement {
	// This check is only relevant for Guaranteed QoS pods.
	if v1qos.GetPodQOS(oldPod) != v1.PodQOSGuaranteed {
		return nil
	}

	cpuResizeRequirement := &nodecapabilitieslib.CapabilityRequirement{
		Key:   nodecapabilitiesregistry.GuaranteedQoSPodCPUResize,
		Value: "true",
	}

	// Check if CPU requests or limits have changed for any container.
	for i := range oldPod.Spec.Containers {
		if newPod.Spec.Containers[i].Resources.Requests.Cpu() != oldPod.Spec.Containers[i].Resources.Requests.Cpu() {
			return cpuResizeRequirement
		}
		if newPod.Spec.Containers[i].Resources.Limits.Cpu() != oldPod.Spec.Containers[i].Resources.Limits.Cpu() {
			return cpuResizeRequirement
		}
	}

	// Check for sidecar containers as well.
	for i, c := range oldPod.Spec.InitContainers {
		if c.RestartPolicy != nil && *c.RestartPolicy == v1.ContainerRestartPolicyAlways {
			if newPod.Spec.InitContainers[i].Resources.Limits.Cpu() != oldPod.Spec.InitContainers[i].Resources.Limits.Cpu() {
				return cpuResizeRequirement
			}
		}
	}

	return nil
}

func init() {
	fd := nodecapabilitiesregistry.BuildFeatureDependency(features.InPlacePodVerticalScalingExclusiveCPUs)
	nodecapabilitiesregistry.Register(nodecapabilitieslib.Capability{
		Name:              nodecapabilitiesregistry.GuaranteedQoSPodCPUResize,
		FeatureDependency: []nodecapabilitieslib.FeatureDependency{fd},
		UpdateInferrer:    &guaranteedCPUResizeInferrer{},
	})
}
