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

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.0
// +k8s:prerelease-lifecycle-gen:removed=1.10

// FlunderList is a list of Flunder objects.
type FlunderList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	Items []Flunder `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// ReferenceType
//
// +enum
type ReferenceType string

const (
	FlunderReferenceType = ReferenceType("Flunder")
	FischerReferenceType = ReferenceType("Fischer")
)

type FlunderSpec struct {
	// A name of another flunder or fischer, depending on the reference type.
	//
	// +k8s:validation:format="ip"
	// +k8s:validation:maxLength=128
	// +k8s:validation:cel[0]:rule="oldSelf == self"
	// +k8s:validation:cel[0]:message="immutable field"
	Reference string `json:"reference,omitempty" protobuf:"bytes,1,opt,name=reference"`
	// The reference type, defaults to "Flunder" if reference is set.
	ReferenceType *ReferenceType `json:"referenceType,omitempty" protobuf:"bytes,2,opt,name=referenceType"`

	Primary Widget `json:"primary,omitempty" protobuf:"bytes,3,opt,name=primary"`

	Extras []Widget `json:"extras,omitempty" protobuf:"bytes,4,rep,name=extras"`

	More map[string]Widget `json:"more,omitempty" protobuf:"bytes,5,rep,name=more"`

	Layer *Layer `json:"layer,omitempty" protobuf:"bytes,6,opt,name=layer"`
}

type Layer struct {
	Extras []Widget `json:"extras" protobuf:"bytes,1,rep,name=extras"`
}

// Widget
type Widget struct {
	// +k8s:validation:format="ip"
	Name string `json:"name,omitempty" protobuf:"bytes,1,opt,name=name"`

	Something []Something `json:"something" protobuf:"bytes,2,rep,name=something"`
}

type Something struct {
	// +k8s:validation:format="ip"
	Name string `json:"name,omitempty" protobuf:"bytes,1,opt,name=name"`
}

type FlunderStatus struct {
	// +k8s:validation:format="ip"
	Name string `json:"name,omitempty" protobuf:"bytes,1,opt,name=name"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.0
// +k8s:prerelease-lifecycle-gen:removed=1.10

type Flunder struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Spec              FlunderSpec   `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
	Status            FlunderStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.0
// +k8s:prerelease-lifecycle-gen:removed=1.10

type Fischer struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// DisallowedFlunders holds a list of Flunder.Names that are disallowed.
	// +listType=atomic
	DisallowedFlunders []string `json:"disallowedFlunders,omitempty" protobuf:"bytes,2,rep,name=disallowedFlunders"`

	// +k8s:validation:format="ip"
	Reference string `json:"reference,omitempty" protobuf:"bytes,4,opt,name=reference"`

	Primary Widget `json:"primary,omitempty" protobuf:"bytes,3,opt,name=primary"`
}

// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.0
// +k8s:prerelease-lifecycle-gen:removed=1.10

// FischerList is a list of Fischer objects.
type FischerList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	Items []Fischer `json:"items" protobuf:"bytes,2,rep,name=items"`
}
