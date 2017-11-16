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

// +k8s:openapi-gen=true
package v1beta1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Policy contains a single ABAC policy rule
type Policy struct {
	metav1.TypeMeta `json:",inline"`

	// Spec describes the policy rule
	Spec PolicySpec `json:"spec"`
}

// PolicySpec contains the attributes for a policy rule
type PolicySpec struct {
	// User is the username this rule applies to.
	// Either user or group is required to match the request.
	// "*" matches all users.
	// +optional
	User string `json:"user,omitempty"`

	// Group is the group this rule applies to.
	// Either user or group is required to match the request.
	// "*" matches all groups.
	// +optional
	Group string `json:"group,omitempty"`

	// Readonly matches readonly requests when true, and all requests when false
	// +optional
	Readonly bool `json:"readonly,omitempty"`

	// APIGroup is the name of an API group. APIGroup, Resource, and Namespace are required to match resource requests.
	// "*" matches all API groups
	// +optional
	APIGroup string `json:"apiGroup,omitempty"`

	// Resource is the name of a resource. APIGroup, Resource, and Namespace are required to match resource requests.
	// "*" matches all resources
	// +optional
	Resource string `json:"resource,omitempty"`

	// Namespace is the name of a namespace. APIGroup, Resource, and Namespace are required to match resource requests.
	// "*" matches all namespaces (including unnamespaced requests)
	// +optional
	Namespace string `json:"namespace,omitempty"`

	// NonResourcePath matches non-resource request paths.
	// "*" matches all paths
	// "/foo/*" matches all subpaths of foo
	// +optional
	NonResourcePath string `json:"nonResourcePath,omitempty"`
}
