/*
Copyright 2023 The Kubernetes Authors.

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
	"fmt"
	"sync"

	resourcev1alpha2 "k8s.io/api/resource/v1alpha2"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/errors"
)

var _ CheckpointState = &stateCheckpoint{}

// CheckpointState interface provides to get and store state
type CheckpointState interface {
	GetOrCreate() (ClaimInfoStateList, error)
	Store(ClaimInfoStateList) error
}

// ClaimInfoState is used to store claim info state in a checkpoint
// +k8s:deepcopy-gen=true
type ClaimInfoState struct {
	// Name of the DRA driver
	DriverName string

	// ClassName is a resource class of the claim
	ClassName string

	// ClaimUID is an UID of the resource claim
	ClaimUID types.UID

	// ClaimName is a name of the resource claim
	ClaimName string

	// Namespace is a claim namespace
	Namespace string

	// PodUIDs is a set of pod UIDs that reference a resource
	PodUIDs sets.Set[string]

	// ResourceHandles is a list of opaque resource data for processing by a specific kubelet plugin
	ResourceHandles []resourcev1alpha2.ResourceHandle

	// CDIDevices is a map of DriverName --> CDI devices returned by the
	// GRPC API call NodePrepareResource
	CDIDevices map[string][]string
}

// ClaimInfoStateWithoutResourceHandles is an old implementation of the ClaimInfoState
// TODO: remove in Beta
type ClaimInfoStateWithoutResourceHandles struct {
	// Name of the DRA driver
	DriverName string

	// ClassName is a resource class of the claim
	ClassName string

	// ClaimUID is an UID of the resource claim
	ClaimUID types.UID

	// ClaimName is a name of the resource claim
	ClaimName string

	// Namespace is a claim namespace
	Namespace string

	// PodUIDs is a set of pod UIDs that reference a resource
	PodUIDs sets.Set[string]

	// CDIDevices is a map of DriverName --> CDI devices returned by the
	// GRPC API call NodePrepareResource
	CDIDevices map[string][]string
}

type stateCheckpoint struct {
	sync.RWMutex
	checkpointManager checkpointmanager.CheckpointManager
	checkpointName    string
}

// NewCheckpointState creates new State for keeping track of claim info  with checkpoint backend
func NewCheckpointState(stateDir, checkpointName string) (*stateCheckpoint, error) {
	if len(checkpointName) == 0 {
		return nil, fmt.Errorf("received empty string instead of checkpointName")
	}

	checkpointManager, err := checkpointmanager.NewCheckpointManager(stateDir)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize checkpoint manager: %v", err)
	}
	stateCheckpoint := &stateCheckpoint{
		checkpointManager: checkpointManager,
		checkpointName:    checkpointName,
	}

	return stateCheckpoint, nil
}

// get state from a checkpoint and creates it if it doesn't exist
func (sc *stateCheckpoint) GetOrCreate() (ClaimInfoStateList, error) {
	sc.Lock()
	defer sc.Unlock()

	checkpoint := NewDRAManagerCheckpoint()
	err := sc.checkpointManager.GetCheckpoint(sc.checkpointName, checkpoint)
	if err == errors.ErrCheckpointNotFound {
		sc.store(ClaimInfoStateList{})
		return ClaimInfoStateList{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get checkpoint %v: %v", sc.checkpointName, err)
	}

	return checkpoint.Entries, nil
}

// saves state to a checkpoint
func (sc *stateCheckpoint) Store(claimInfoStateList ClaimInfoStateList) error {
	sc.Lock()
	defer sc.Unlock()

	return sc.store(claimInfoStateList)
}

// saves state to a checkpoint, caller is responsible for locking
func (sc *stateCheckpoint) store(claimInfoStateList ClaimInfoStateList) error {
	checkpoint := NewDRAManagerCheckpoint()
	checkpoint.Entries = claimInfoStateList

	err := sc.checkpointManager.CreateCheckpoint(sc.checkpointName, checkpoint)
	if err != nil {
		return fmt.Errorf("could not save checkpoint %s: %v", sc.checkpointName, err)
	}
	return nil
}
