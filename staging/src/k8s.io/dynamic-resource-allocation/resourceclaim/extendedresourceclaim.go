/*
Copyright The Kubernetes Authors.

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

package resourceclaim

import "strings"

// maxGeneratedNameBaseLength mirrors MaxGeneratedNameLength in
// k8s.io/apiserver/pkg/storage/names, which this module cannot import.
const maxGeneratedNameBaseLength = 58

// ExtendedResourceClaimNameBase returns the GenerateName base for the
// ResourceClaim that the scheduler creates for the extended resources of a pod.
// It is truncated the same way the name generator truncates it, so that every
// generated name starts with the returned base.
func ExtendedResourceClaimNameBase(podName string) string {
	base := podName + "-extended-resources-"
	if len(base) > maxGeneratedNameBaseLength {
		base = base[:maxGeneratedNameBaseLength]
	}
	return base
}

// IsExtendedResourceClaimNameForPod returns true if claimName could have been
// generated from ExtendedResourceClaimNameBase for the pod.
func IsExtendedResourceClaimNameForPod(podName, claimName string) bool {
	return strings.HasPrefix(claimName, ExtendedResourceClaimNameBase(podName))
}
