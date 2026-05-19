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
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures"
)

// Ensure the feature struct implements the unified Feature interface.
var _ nodedeclaredfeatures.Feature = &guaranteedQoSPodCPUResizeFeature{}

const (
	// IPPRExclusiveCPUsFeatureGate is the feature gate for IPPRExclusiveCPUsFeatureGate.
	IPPRExclusiveCPUsFeatureGate = "InPlacePodVerticalScalingExclusiveCPUs"
	// CPUManagerPolicyStatic is the value for the static CPUManagerPolicy.
	CPUManagerPolicyStatic = "static"
	// CPUManagerPolicyNone is the value for the none CPUManagerPolicy.
	CPUManagerPolicyNone = "none"
	// GuaranteedQoSPodCPUResize is a declared feature that indicates a node supports in-place pod resize for guaranteed QoS pods with exclusive CPUs.
	GuaranteedQoSPodCPUResize = "GuaranteedQoSPodCPUResize"
)

// GuaranteedQoSPodCPUResizeFeature is the implementation of the `GuaranteedQoSPodCPUResize` feature.
var GuaranteedQoSPodCPUResizeFeature = &guaranteedQoSPodCPUResizeFeature{}

type guaranteedQoSPodCPUResizeFeature struct{}

func (f *guaranteedQoSPodCPUResizeFeature) Name() string {
	return GuaranteedQoSPodCPUResize
}

func (f *guaranteedQoSPodCPUResizeFeature) Discover(cfg *nodedeclaredfeatures.NodeConfiguration) bool {
	featureGateEnabled := cfg.FeatureGates.Enabled(IPPRExclusiveCPUsFeatureGate)
	cpuManagerPolicy := cfg.StaticConfig.CPUManagerPolicy
	if (featureGateEnabled && cpuManagerPolicy == CPUManagerPolicyStatic) || (cpuManagerPolicy == CPUManagerPolicyNone) {
		return true
	}
	return false
}

func (f *guaranteedQoSPodCPUResizeFeature) InferForScheduling(podInfo *nodedeclaredfeatures.PodInfo) bool {
	// This feature is only relevant for pod updates (resizes).
	return false
}

func (f *guaranteedQoSPodCPUResizeFeature) InferForUpdate(oldPodInfo, newPodInfo *nodedeclaredfeatures.PodInfo) bool {
	// Since this is an update, the QOS class must already be se ans must be the same for old and new pod spec.
	if oldPodInfo.Status != nil && oldPodInfo.Status.QOSClass != v1.PodQOSGuaranteed {
		return false
	}

	oldPodSpec := oldPodInfo.Spec
	newPodSpec := newPodInfo.Spec

	// Check if CPU request is changing for any container.
	for i := range oldPodSpec.Containers {
		oldCPU := oldPodSpec.Containers[i].Resources.Requests.Cpu()
		newCPU := newPodSpec.Containers[i].Resources.Requests.Cpu()
		if oldCPU != nil && newCPU != nil && !oldCPU.Equal(*newCPU) {
			return true
		}
	}

	return false
}

func (f *guaranteedQoSPodCPUResizeFeature) MaxVersion() *version.Version {
	return nil
}
