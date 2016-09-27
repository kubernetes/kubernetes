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

package v1alpha1

import (
	"k8s.io/client-go/1.5/pkg/api/unversioned"
	"k8s.io/client-go/1.5/pkg/api/v1"
)

// +genclient=true

// PetSet represents a set of pods with consistent identities.
// Identities are defined as:
//  - Network: A single stable DNS and hostname.
//  - Storage: As many VolumeClaims as requested.
// The PetSet guarantees that a given network identity will always
// map to the same storage identity. PetSet is currently in alpha
// and subject to change without notice.
type PetSet struct {
	unversioned.TypeMeta `json:",inline"`
	v1.ObjectMeta        `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Spec defines the desired identities of pets in this set.
	Spec PetSetSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// Status is the current status of Pets in this PetSet. This data
	// may be out of date by some window of time.
	Status PetSetStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// A PetSetSpec is the specification of a PetSet.
type PetSetSpec struct {
	// Replicas is the desired number of replicas of the given Template.
	// These are replicas in the sense that they are instantiations of the
	// same Template, but individual replicas also have a consistent identity.
	// If unspecified, defaults to 1.
	// TODO: Consider a rename of this field.
	Replicas *int32 `json:"replicas,omitempty" protobuf:"varint,1,opt,name=replicas"`

	// Selector is a label query over pods that should match the replica count.
	// If empty, defaulted to labels on the pod template.
	// More info: http://releases.k8s.io/HEAD/docs/user-guide/labels.md#label-selectors
	Selector *unversioned.LabelSelector `json:"selector,omitempty" protobuf:"bytes,2,opt,name=selector"`

	// Template is the object that describes the pod that will be created if
	// insufficient replicas are detected. Each pod stamped out by the PetSet
	// will fulfill this Template, but have a unique identity from the rest
	// of the PetSet.
	Template v1.PodTemplateSpec `json:"template" protobuf:"bytes,3,opt,name=template"`

	// VolumeClaimTemplates is a list of claims that pets are allowed to reference.
	// The PetSet controller is responsible for mapping network identities to
	// claims in a way that maintains the identity of a pet. Every claim in
	// this list must have at least one matching (by name) volumeMount in one
	// container in the template. A claim in this list takes precedence over
	// any volumes in the template, with the same name.
	// TODO: Define the behavior if a claim already exists with the same name.
	VolumeClaimTemplates []v1.PersistentVolumeClaim `json:"volumeClaimTemplates,omitempty" protobuf:"bytes,4,rep,name=volumeClaimTemplates"`

	// ServiceName is the name of the service that governs this PetSet.
	// This service must exist before the PetSet, and is responsible for
	// the network identity of the set. Pets get DNS/hostnames that follow the
	// pattern: pet-specific-string.serviceName.default.svc.cluster.local
	// where "pet-specific-string" is managed by the PetSet controller.
	ServiceName string `json:"serviceName" protobuf:"bytes,5,opt,name=serviceName"`
}

// PetSetStatus represents the current state of a PetSet.
type PetSetStatus struct {
	// most recent generation observed by this autoscaler.
	ObservedGeneration *int64 `json:"observedGeneration,omitempty" protobuf:"varint,1,opt,name=observedGeneration"`

	// Replicas is the number of actual replicas.
	Replicas int32 `json:"replicas" protobuf:"varint,2,opt,name=replicas"`
}

// PetSetList is a collection of PetSets.
type PetSetList struct {
	unversioned.TypeMeta `json:",inline"`
	unversioned.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items                []PetSet `json:"items" protobuf:"bytes,2,rep,name=items"`
}
