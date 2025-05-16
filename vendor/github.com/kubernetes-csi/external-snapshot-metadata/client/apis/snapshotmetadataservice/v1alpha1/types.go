/*
Copyright 2024 The Kubernetes Authors.

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

// +kubebuilder:object:generate=true
package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SnapshotMetadataServiceSpec defines the desired state of SnapshotMetadataService
// This contains data needed to connect to a Kubernetes SnapshotMetadata gRPC service.
type SnapshotMetadataServiceSpec struct {
	// The audience string value expected in a client's authentication token passed
	// in the "security_token" field of each gRPC call.
	// Required.
	Audience string `json:"audience"`
	// The TCP endpoint address of the gRPC service.
	// Required.
	Address string `json:"address"`
	// Certificate authority bundle needed by the client to validate the service.
	// Required.
	CACert []byte `json:"caCert"`
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:object:root=true
// +kubebuilder:resource:scope=Cluster,shortName=sms
// SnapshotMetadataService is the Schema for the snapshotmetadataservices API
// The presence of a SnapshotMetadataService CR advertises the existence of a CSI
// driver's Kubernetes SnapshotMetadata gRPC service.
// An audience scoped Kubernetes authentication bearer token must be passed in the
// "security_token" field of each gRPC call made by a Kubernetes backup client.
type SnapshotMetadataService struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`
	// Required.
	Spec SnapshotMetadataServiceSpec `json:"spec"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// SnapshotMetadataServiceList contains a list of SnapshotMetadataService
type SnapshotMetadataServiceList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []SnapshotMetadataService `json:"items"`
}
