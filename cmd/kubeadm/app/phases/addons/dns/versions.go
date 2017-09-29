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
	kubeDNSv170AndAboveVersion = "1.14.5"
)

// GetKubeDNSVersion returns the right kube-dns version for a specific k8s version
func GetKubeDNSVersion(kubeVersion *version.Version) string {
	// v1.7.0+ uses 1.14.5, just return that here
	// In the future when the kube-dns version is bumped at HEAD; add conditional logic to return the right versions
	// Also, the version might be bumped for different k8s releases on the same branch
	return kubeDNSv170AndAboveVersion
}

// GetKubeDNSManifest returns the right kube-dns YAML manifest for a specific k8s version
func GetKubeDNSManifest(kubeVersion *version.Version) string {
	// v1.7.0+ has only one known YAML manifest spec, just return that here
	// In the future when the kube-dns version is bumped at HEAD; add conditional logic to return the right manifest
	return v170AndAboveKubeDNSDeployment
}
