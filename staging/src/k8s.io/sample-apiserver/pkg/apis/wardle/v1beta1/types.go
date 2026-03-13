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

package v1beta1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// FlunderList is a list of Flunder objects.
type FlunderList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	Items []Flunder `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// ReferenceType defines the type of an object reference.
type ReferenceType string

const (
	// FlunderReferenceType is used for Flunder references.
	FlunderReferenceType = ReferenceType("Flunder")
	// FischerReferenceType is used for Fischer references.
	FischerReferenceType = ReferenceType("Fischer")
)

// FlunderSpec is the specification of a Flunder.
type FlunderSpec struct {
	// A name of another flunder, mutually exclusive to the FischerReference.
	FlunderReference string `json:"flunderReference,omitempty" protobuf:"bytes,1,opt,name=flunderReference"`
	// A name of a fischer, mutually exclusive to the FlunderReference.
	FischerReference string `json:"fischerReference,omitempty" protobuf:"bytes,2,opt,name=fischerReference"`
	// The reference type.
	ReferenceType ReferenceType `json:"referenceType,omitempty" protobuf:"bytes,3,opt,name=referenceType"`
}

// FlunderStatus is the status of a Flunder.
type FlunderStatus struct {
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Flunder is an example type with a spec and a status.
type Flunder struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	Spec   FlunderSpec   `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
	Status FlunderStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}
