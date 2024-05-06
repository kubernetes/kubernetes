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

package storagemigration

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
	metav1.TypeMeta
	// Standard object metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta
	// Specification of the migration.
	// +optional
	Spec StorageVersionMigrationSpec
	// Status of the migration.
	// +optional
	Status StorageVersionMigrationStatus
}

// Spec of the storage version migration.
type StorageVersionMigrationSpec struct {
	// The resource that is being migrated. The migrator sends requests to
	// the endpoint serving the resource.
	// Immutable.
	Resource GroupVersionResource
	// The token used in the list options to get the next chunk of objects
	// to migrate. When the .status.conditions indicates the migration is
	// "Running", users can use this token to check the progress of the
	// migration.
	// +optional
	ContinueToken string
	// TODO: consider recording the storage version hash when the migration
	// is created. It can avoid races.
}

// The names of the group, the version, and the resource.
type GroupVersionResource struct {
	// The name of the group.
	Group string
	// The name of the version.
	Version string
	// The name of the resource.
	Resource string
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
	Type MigrationConditionType
	// Status of the condition, one of True, False, Unknown.
	Status corev1.ConditionStatus
	// The last time this condition was updated.
	// +optional
	LastUpdateTime metav1.Time
	// The reason for the condition's last transition.
	// +optional
	Reason string
	// A human readable message indicating details about the transition.
	// +optional
	Message string
}

// Status of the storage version migration.
type StorageVersionMigrationStatus struct {
	// The latest available observations of the migration's current state.
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []MigrationCondition
	// ResourceVersion to compare with the GC cache for performing the migration.
	// This is the current resource version of given group, version and resource when
	// kube-controller-manager first observes this StorageVersionMigration resource.
	ResourceVersion string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.30

// StorageVersionMigrationList is a collection of storage version migrations.
type StorageVersionMigrationList struct {
	metav1.TypeMeta

	// Standard list metadata
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta
	// Items is the list of StorageVersionMigration
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	Items []StorageVersionMigration
}
