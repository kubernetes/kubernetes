/*
Copyright 2017 The Kubernetes Authors.

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

package v1alpha1

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type NodeConfigSourcePool struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec defines the spec for the NodeConfigSourcePool
	// +optional
	Spec NodeConfigSourcePoolSpec `json:"spec, omitempty" protobuf:"bytes,2,name=spec"`
}

type NodeConfigSourcePoolSpec struct {
	// LabelSelector selects this "pool" of commonly-configured Nodes
	// +optional
	LabelSelector *metav1.LabelSelector `json:"labelSelector,omitempty" protobuf:"bytes,1,opt,name=labelSelector"`

	// The percentage of nodes in the selector that should use the new config
	// Assumed to be 0 if unset
	// TODO(mtaufen): Add validation that this is 0-100
	// +optional
	PercentNew int `json:"percentNew,omitempty" protobuf:"bytes,2,opt,name=percentNew"`

	// History is the history of config pushes, from first to last
	// slice of pointers, since you need to be able to reset to nil (default on-disk config)
	// +optional
	History []*v1.NodeConfigSource `json:"history,omitempty" protobuf:"bytes,3,rep,name=history"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type NodeConfigSourcePoolList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#types-kinds
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// List of NodeConfigSourcePools
	Items []NodeConfigSourcePool `json:"items" protobuf:"bytes,2,rep,name=items"`
}
