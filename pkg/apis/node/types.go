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

package node

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// RuntimeClass defines a class of runtime supported in the cluster.
type RuntimeClass struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Specification of the RuntimeClass
	Spec RuntimeClassSpec
}

// RuntimeClassSpec is a specification of a RuntimeClass.
type RuntimeClassSpec struct {
	// RuntimeHandler specifies the underlying runtime the CRI calls to handle pod and/or container
	// creation. The possible values are specific to a given configuration & CRI implementation.
	// The empty string is equivalent to the default behavior.
	// The name must conform to the DNS subdomain spec (RFC 1123).
	// +optional
	RuntimeHandler string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// RuntimeClassList is a list of RuntimeClass objects.
type RuntimeClassList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	// Items is a list of schema objects.
	Items []RuntimeClass
}
