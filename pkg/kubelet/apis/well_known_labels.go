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

	"k8s.io/api/core/v1"
)

// KubeletLabels returns the list of label keys kubelets are allowed to set on their own Node objects
func KubeletLabels() []string {
	return v1.KubeletLabels.List()
}

// KubeletLabelNamespaces returns the list of label key namespaces kubelets are allowed to set on their own Node objects
func KubeletLabelNamespaces() []string {
	return v1.KubeletLabelNamespaces.List()
}

// IsKubeletLabel returns true if the label key is one that kubelets are allowed to set on their own Node object.
// This checks if the key is in the KubeletLabels() list, or has a namespace in the KubeletLabelNamespaces() list.
func IsKubeletLabel(key string) bool {
	if v1.KubeletLabels.Has(key) {
		return true
	}

	namespace := getLabelNamespace(key)
	for allowedNamespace := range v1.KubeletLabelNamespaces {
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
