// +build windows

/*
Copyright 2018 The Kubernetes Authors.

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

package apis

import (
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
)

const (
	// HypervIsolationAnnotationKey is used to run windows containers with hyperv isolation.
	// Refer https://aka.ms/hyperv-container.
	HypervIsolationAnnotationKey = "experimental.windows.kubernetes.io/isolation-type"
	// HypervIsolationValue is used to run windows containers with hyperv isolation.
	// Refer https://aka.ms/hyperv-container.
	HypervIsolationValue = "hyperv"
)

// ShouldIsolatedByHyperV returns true if a windows container should be run with hyperv isolation.
func ShouldIsolatedByHyperV(annotations map[string]string) bool {
	klog.Warningf("The hyper-v FeatureGate is deprecated in 1.20 and will be removed in 1.21")

	if !utilfeature.DefaultFeatureGate.Enabled(features.HyperVContainer) {
		return false
	}

	v, ok := annotations[HypervIsolationAnnotationKey]
	return ok && v == HypervIsolationValue
}
