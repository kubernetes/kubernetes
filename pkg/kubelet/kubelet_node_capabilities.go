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

package kubelet

import (
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	nodecapabilities "k8s.io/kubernetes/pkg/features/nodecapabilities"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
)

// gatherCapabilities gathers all potential node capabilities based on the Kubelet's configuration.
func (kl *Kubelet) gatherCapabilities() map[string]string {
	potentialCapabilities := make(map[string]string)
	// Gather and report capabilities based on the Kubelet's configuration.

	if kl.nodeCapabilitiesHelper.IsFeatureGateRelevant(features.InPlacePodVerticalScalingExclusiveCPUs, kl.kubeletVersion) {
		cpuManagerPolicy := kl.containerManager.GetNodeConfig().CPUManagerPolicy
		if cpuManagerPolicy == string(cpumanager.PolicyStatic) &&
			utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingExclusiveCPUs) {
			potentialCapabilities[nodecapabilities.GuaranteedQoSPodCPUResize] = "true"
		}
		// If the CPUManagerPolicy is None, we still report the GuaranteedQoSPodCPUResize capability even if InPlacePodVerticalScalingExclusiveCPUs is not enabled.
		if cpuManagerPolicy == string(cpumanager.PolicyNone) {
			potentialCapabilities[nodecapabilities.GuaranteedQoSPodCPUResize] = "true"
		}
	}
	return potentialCapabilities
}

// determineNodeCapabilities determines the final set of node capabilities to be reported.
// It validates the capabilities and filters them based on their lifecycle.
func (kl *Kubelet) determineNodeCapabilities() map[string]string {
	potentialCapabilities := kl.gatherCapabilities()
	finalCapabilities := make(map[string]string)

	if kl.kubeletVersion == nil {
		klog.ErrorS(nil, "Kubelet version is not set, cannot determine node capabilities correctly")
		return finalCapabilities
	}

	for k, v := range potentialCapabilities {
		if err := nodecapabilities.ValidateCapability(k, v); err != nil {
			klog.ErrorS(err, "Invalid capability")
			continue
		}
		finalCapabilities[k] = v
	}
	return finalCapabilities
}
