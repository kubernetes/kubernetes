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

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.30

// StorageVersionMigration represents a migration of stored data to the latest
// storage version.
type StorageVersionMigration struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// Specification of the migration.
	// +optional
	Spec StorageVersionMigrationSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`
	// Status of the migration.
	// +optional
	Status StorageVersionMigrationStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// Spec of the storage version migration.
type StorageVersionMigrationSpec struct {
	// The resource that is being migrated. The migrator sends requests to
	// the endpoint serving the resource.
	// Immutable.
	Resource GroupVersionResource `json:"resource" protobuf:"bytes,1,opt,name=resource"`
	// The token used in the list options to get the next chunk of objects
	// to migrate. When the .status.conditions indicates the migration is
	// "Running", users can use this token to check the progress of the
	// migration.
	// +optional
	ContinueToken string `json:"continueToken,omitempty" protobuf:"bytes,2,opt,name=continueToken"`
	// TODO: consider recording the storage version hash when the migration
	// is created. It can avoid races.
}

// The names of the group, the version, and the resource.
type GroupVersionResource struct {
	// The name of the group.
	Group string `json:"group,omitempty" protobuf:"bytes,1,opt,name=group"`
	// The name of the version.
	Version string `json:"version,omitempty" protobuf:"bytes,2,opt,name=version"`
	// The name of the resource.
	Resource string `json:"resource,omitempty" protobuf:"bytes,3,opt,name=resource"`
}

type MigrationConditionType string

const (
	// Indicates that the migration is running.
	MigrationRunning MigrationConditionType = "Running"
	// Indicates that the migration has completed successfully.
	MigrationSucceeded MigrationConditionType = "Succeeded"
	// Indicates that the migration has failed.
	MigrationFailed MigrationConditionType = "Failed"
)

// Describes the state of a migration at a certain point.
type MigrationCondition struct {
	// Type of the condition.
	Type MigrationConditionType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=MigrationConditionType"`
	// Status of the condition, one of True, False, Unknown.
	Status corev1.ConditionStatus `json:"status" protobuf:"bytes,2,opt,name=status,casttype=k8s.io/api/core/v1.ConditionStatus"`
	// The last time this condition was updated.
	// +optional
	LastUpdateTime metav1.Time `json:"lastUpdateTime,omitempty" protobuf:"bytes,3,opt,name=lastUpdateTime"`
	// The reason for the condition's last transition.
	// +optional
	Reason string `json:"reason,omitempty" protobuf:"bytes,4,opt,name=reason"`
	// A human readable message indicating details about the transition.
	// +optional
	Message string `json:"message,omitempty" protobuf:"bytes,5,opt,name=message"`
}

// Status of the storage version migration.
type StorageVersionMigrationStatus struct {
	// The latest available observations of the migration's current state.
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []MigrationCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,1,rep,name=conditions"`
	// ResourceVersion to compare with the GC cache for performing the migration.
	// This is the current resource version of given group, version and resource when
	// kube-controller-manager first observes this StorageVersionMigration resource.
	ResourceVersion string `json:"resourceVersion,omitempty" protobuf:"bytes,2,opt,name=resourceVersion"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.30

// StorageVersionMigrationList is a collection of storage version migrations.
type StorageVersionMigrationList struct {
	metav1.TypeMeta `json:",inline"`

	// Standard list metadata
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// Items is the list of StorageVersionMigration
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	Items []StorageVersionMigration `json:"items" listType:"map" listMapKey:"type" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,2,rep,name=items"`
}
