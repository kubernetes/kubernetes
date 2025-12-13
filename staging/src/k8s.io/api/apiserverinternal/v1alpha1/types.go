/*
Copyright 2019 The Kubernetes Authors.

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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Storage version of a specific resource.
type StorageVersion struct {
	metav1.TypeMeta `json:",inline"`
	// metadata is the standard object metadata.
	// The name is <group>.<resource>.
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec is an empty spec. It is here to comply with Kubernetes API style.
	Spec StorageVersionSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// status hold the StorageVersionStatus, reporting the version an API server
	// instance can decode and the version it encodes objects to when persisting
	// objects in the backend.
	Status StorageVersionStatus `json:"status" protobuf:"bytes,3,opt,name=status"`
}

// StorageVersionSpec is an empty spec.
type StorageVersionSpec struct{}

// API server instances report the versions they can decode and the version they
// encode objects to when persisting objects in the backend.
type StorageVersionStatus struct {
	// storageVersions lists the reported versions per API server instance.
	// +optional
	// +listType=map
	// +listMapKey=apiServerID
	StorageVersions []ServerStorageVersion `json:"storageVersions,omitempty" protobuf:"bytes,1,opt,name=storageVersions"`
	// commonEncodingVersion is set to an encoding storage version if all API server
	// instances share that same version. If they don't share one storage version, this
	// field is left empty.
	// API servers should finish updating its storageVersionStatus entry before
	// serving write operations, so that this field will be in sync with the reality.
	// +optional
	CommonEncodingVersion *string `json:"commonEncodingVersion,omitempty" protobuf:"bytes,2,opt,name=commonEncodingVersion"`

	// conditions lists the latest available observations of the storageVersion's state.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []StorageVersionCondition `json:"conditions,omitempty" protobuf:"bytes,3,opt,name=conditions"`
}

// An API server instance reports the version it can decode and the version it
// encodes objects to when persisting objects in the backend.
type ServerStorageVersion struct {
	// apiServerID is the ID of the reporting API server.
	APIServerID string `json:"apiServerID,omitempty" protobuf:"bytes,1,opt,name=apiServerID"`

	// encodingVersion is the version the API server encodes the object to when
	// persisting it in the backend (e.g., etcd).
	EncodingVersion string `json:"encodingVersion,omitempty" protobuf:"bytes,2,opt,name=encodingVersion"`

	// decodableVersions are the encoding versions the API server can handle to decode.
	// The API server can decode objects encoded in these versions.
	// The encodingVersion must be included in the decodableVersions.
	// +listType=set
	DecodableVersions []string `json:"decodableVersions,omitempty" protobuf:"bytes,3,opt,name=decodableVersions"`

	// servedVersions lists all versions the API server can serve.
	// DecodableVersions must include all ServedVersions.
	// +listType=set
	ServedVersions []string `json:"servedVersions,omitempty" protobuf:"bytes,4,opt,name=servedVersions"`
}

type StorageVersionConditionType string

const (
	// Indicates that encoding storage versions reported by all servers are equal.
	AllEncodingVersionsEqual StorageVersionConditionType = "AllEncodingVersionsEqual"
)

type ConditionStatus string

const (
	ConditionTrue    ConditionStatus = "True"
	ConditionFalse   ConditionStatus = "False"
	ConditionUnknown ConditionStatus = "Unknown"
)

// Describes the state of the storageVersion at a certain point.
type StorageVersionCondition struct {
	// type of the condition.
	// +required
	Type StorageVersionConditionType `json:"type" protobuf:"bytes,1,opt,name=type"`
	// status of the condition, one of True, False, Unknown.
	// +required
	Status ConditionStatus `json:"status" protobuf:"bytes,2,opt,name=status"`
	// observedGeneration represents the .metadata.generation that the condition was set based upon, if field is set.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty" protobuf:"varint,3,opt,name=observedGeneration"`
	// lastTransitionTime is the last time the condition transitioned from one status to another.
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty" protobuf:"bytes,4,opt,name=lastTransitionTime"`
	// reason for the condition's last transition.
	// +required
	Reason string `json:"reason" protobuf:"bytes,5,opt,name=reason"`
	// message is a human readable string indicating details about the transition.
	// +required
	Message string `json:"message,omitempty" protobuf:"bytes,6,opt,name=message"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// A list of StorageVersions.
type StorageVersionList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// Items holds a list of StorageVersion
	Items []StorageVersion `json:"items" protobuf:"bytes,2,rep,name=items"`
}
