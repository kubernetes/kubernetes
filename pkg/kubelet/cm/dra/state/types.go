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

package state

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
)

const (
	Version = "v1"
)

type ClaimInfoStateList []ClaimInfoState

type CheckpointData struct {
	metav1.TypeMeta
	Entries ClaimInfoStateList
}

// +k8s:deepcopy-gen=true
type ClaimInfoState struct {
	// ClaimUID is the UID of a resource claim
	ClaimUID types.UID

	// ClaimName is the name of a resource claim
	ClaimName string

	// Namespace is a claim namespace
	Namespace string

	// PodUIDs is a set of pod UIDs that reference a resource
	PodUIDs sets.Set[string]

	// DriverState contains information about all drivers which have allocation
	// results in the claim, even if they don't provide devices for their results.
	DriverState map[string]DriverState
}

// DriverState is used to store per-device claim info state in a checkpoint
// +k8s:deepcopy-gen=true
type DriverState struct {
	Devices []Device
}

// Device is how a DRA driver described an allocated device in a claim
// to kubelet. RequestName and CDI device IDs are optional.
// +k8s:deepcopy-gen=true
type Device struct {
	PoolName     string
	DeviceName   string
	RequestNames []string
	CDIDeviceIDs []string
}
