/*
Copyright 2016 The Kubernetes Authors.

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

package dns

import (
	"k8s.io/kubernetes/pkg/util/version"
)

const (
	kubeDNSv180AndAboveVersion = "1.14.5"
	kubeDNSv190AndAboveVersion = "1.14.7"

	kubeDNSProbeSRV = "SRV"
	kubeDNSProbeA   = "A"
)

// GetKubeDNSVersion returns the right kube-dns version for a specific k8s version
func GetKubeDNSVersion(kubeVersion *version.Version) string {
	// v1.8.0+ uses 1.14.5
	// v1.9.0+ uses 1.14.7
	// In the future when the kube-dns version is bumped at HEAD; add conditional logic to return the right versions
	// Also, the version might be bumped for different k8s releases on the same branch
	if kubeVersion.Major() == 1 && kubeVersion.Minor() >= 9 {
		return kubeDNSv190AndAboveVersion
	}
	return kubeDNSv180AndAboveVersion
}

// GetKubeDNSProbeType returns the right kube-dns probe for a specific k8s version
func GetKubeDNSProbeType(kubeVersion *version.Version) string {
	// v1.8.0+ uses type A, just return that here
	// In the future when the kube-dns version is bumped at HEAD; add conditional logic to return the right versions
	// Also, the version might be bumped for different k8s releases on the same branch
	if kubeVersion.Major() == 1 && kubeVersion.Minor() >= 9 {
		return kubeDNSProbeSRV
	}
	return kubeDNSProbeA
}

// GetKubeDNSManifest returns the right kube-dns YAML manifest for a specific k8s version
func GetKubeDNSManifest(kubeVersion *version.Version) string {
	// v1.8.0+ has only one known YAML manifest spec, just return that here
	// In the future when the kube-dns version is bumped at HEAD; add conditional logic to return the right manifest
	return v180AndAboveKubeDNSDeployment
}
