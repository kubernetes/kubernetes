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
	Resource metav1.GroupResource
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

// Status of the storage version migration.
type StorageVersionMigrationStatus struct {
	// The latest available observations of the migration's current state.
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition
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
	Items []StorageVersionMigration
}
