/*
Copyright 2020 The Kubernetes Authors.

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

package corev1

import (
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation"
)

// IsPrefixedNativeResourceName is true if the resource name is explicitly in the kubernetes.io domain.
// I.e. "kubernetes.io/" is part of the resource name.
func IsPrefixedNativeResourceName(name v1.ResourceName) bool {
	return strings.Contains(string(name), v1.ResourceDefaultNamespacePrefix)
}

// IsNativeResourceName is true if the resource name is in kubernetes.io domain.
// Partially-qualified (unprefixed) names are implicitly in the kubernetes.io domain.
func IsNativeResourceName(name v1.ResourceName) bool {
	return !strings.Contains(string(name), "/") ||
		IsPrefixedNativeResourceName(name)
}

// IsExtendedResourceName is true if the resource name is fully-qualified resource name
// outside the kubernetes.io domain. Given extended resources can have a quota,
// the resource name can not be prefixed with "requests." quota prefix.
// Yet, when prefixed with the quota prefix the name has to be a valid qualified name.
func IsExtendedResourceName(name v1.ResourceName) bool {
	if IsNativeResourceName(name) || strings.HasPrefix(string(name), v1.DefaultResourceRequestsPrefix) {
		return false
	}
	// Ensure it satisfies the rules in IsQualifiedName() after converted into quota resource name
	if errs := validation.IsQualifiedName(v1.DefaultResourceRequestsPrefix + string(name)); len(errs) != 0 {
		return false
	}
	return true
}

// IsHugePageResourceName is true if the resource name has the huge page resource prefix.
func IsHugePageResourceName(name v1.ResourceName) bool {
	return strings.HasPrefix(string(name), v1.ResourceHugePagesPrefix)
}

// HugePageResourceName returns a ResourceName with the canonical hugepage
// prefix prepended for the specified page size.  The page size is converted
// to its canonical representation.
func HugePageResourceName(pageSize resource.Quantity) v1.ResourceName {
	return v1.ResourceName(v1.ResourceHugePagesPrefix + pageSize.String())
}

// IsAttachableVolumeResourceName is true when the resource name has the attachable volume prefix
func IsAttachableVolumeResourceName(name v1.ResourceName) bool {
	return strings.HasPrefix(string(name), v1.ResourceAttachableVolumesPrefix)
}
