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
	"context"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	nodecapabilities "k8s.io/kubernetes/pkg/features/nodecapabilities"

	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
)

// determineNodeCapabilities determines the final set of node capabilities to be reported.
// It validates the capabilities and filters them based on their lifecycle.
func (kl *Kubelet) determineNodeCapabilities(ctx context.Context) map[string]string {
	potentialCapabilities := kl.gatherCapabilities(ctx)
	finalCapabilities := make(map[string]string)

	for k, v := range potentialCapabilities {
		if err := nodecapabilities.ValidateCapability(k, v); err != nil {
			klog.ErrorS(err, "Invalid capability")
			continue
		}
		finalCapabilities[k] = v
	}
	return finalCapabilities
}

// gatherCapabilities gathers all potential node capabilities based on the Kubelet's configuration.
func (kl *Kubelet) gatherCapabilities(ctx context.Context) map[string]string {
	potentialCapabilities := make(map[string]string)
	// Handle NodeCapability for in-place pod resize for guaranteed QoS pods.
	handleIPPRExlusiveCPUsCapability(ctx, kl.containerManager.GetNodeConfig().CPUManagerPolicy, potentialCapabilities)
	return potentialCapabilities
}

// handleIPPRExlusiveCPUsCapability handles the GuaranteedQoSPodCPUResize capability based on the Kubelet's configuration
func handleIPPRExlusiveCPUsCapability(ctx context.Context, cpuManagerPolicy string, capabilitiesMap map[string]string) {
	logger := klog.FromContext(ctx)
	featuregateEnabled := utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingExclusiveCPUs)

	if featuregateEnabled && cpuManagerPolicy == string(cpumanager.PolicyStatic) {
		logger.V(4).Info("Enabling GuaranteedQoSPodCPUResize capability", "cpuManagerPolicy", cpuManagerPolicy)
		capabilitiesMap[nodecapabilities.GuaranteedQoSPodCPUResize] = "true"
	}
	// If the CPUManagerPolicy is None, we still report the GuaranteedQoSPodCPUResize capability even if InPlacePodVerticalScalingExclusiveCPUs is not enabled.
	if cpuManagerPolicy == string(cpumanager.PolicyNone) {
		logger.V(4).Info("Enabling GuaranteedQoSPodCPUResize capability", "cpuManagerPolicy", cpuManagerPolicy)
		capabilitiesMap[nodecapabilities.GuaranteedQoSPodCPUResize] = "true"
	}
}
