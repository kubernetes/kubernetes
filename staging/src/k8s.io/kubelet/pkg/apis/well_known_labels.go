/*
Copyright 2015 The Kubernetes Authors.

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
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

const (
	// LabelOS is a label to indicate the operating system of the node.
	// The OS labels are promoted to GA in 1.14. kubelet applies GA labels and stop applying the beta OS labels in Kubernetes 1.19.
	LabelOS = "beta.kubernetes.io/os"
	// LabelArch is a label to indicate the architecture of the node.
	// The Arch labels are promoted to GA in 1.14. kubelet applies GA labels and stop applying the beta Arch labels in Kubernetes 1.19.
	LabelArch = "beta.kubernetes.io/arch"
)

var kubeletLabels = sets.NewString(
	v1.LabelHostname,
	v1.LabelTopologyZone,
	v1.LabelTopologyRegion,
	v1.LabelFailureDomainBetaZone,
	v1.LabelFailureDomainBetaRegion,
	v1.LabelInstanceType,
	v1.LabelInstanceTypeStable,
	v1.LabelOSStable,
	v1.LabelArchStable,

	LabelOS,
	LabelArch,
)

var kubeletLabelNamespaces = sets.NewString(
	v1.LabelNamespaceSuffixKubelet,
	v1.LabelNamespaceSuffixNode,
)

// KubeletLabels returns the list of label keys kubelets are allowed to set on their own Node objects
func KubeletLabels() []string {
	return kubeletLabels.List()
}

// KubeletLabelNamespaces returns the list of label key namespaces kubelets are allowed to set on their own Node objects
func KubeletLabelNamespaces() []string {
	return kubeletLabelNamespaces.List()
}

// IsKubeletLabel returns true if the label key is one that kubelets are allowed to set on their own Node object.
// This checks if the key is in the KubeletLabels() list, or has a namespace in the KubeletLabelNamespaces() list.
func IsKubeletLabel(key string) bool {
	if kubeletLabels.Has(key) {
		return true
	}

	namespace := getLabelNamespace(key)
	for allowedNamespace := range kubeletLabelNamespaces {
		if namespace == allowedNamespace || strings.HasSuffix(namespace, "."+allowedNamespace) {
			return true
		}
	}

	return false
}

func getLabelNamespace(key string) string {
	if parts := strings.SplitN(key, "/", 2); len(parts) == 2 {
		return parts[0]
	}
	return ""
}
