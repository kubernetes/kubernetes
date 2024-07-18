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

package checkpoint

import (
	"errors"
	"fmt"
	"sync"

	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	checkpointerrors "k8s.io/kubernetes/pkg/kubelet/checkpointmanager/errors"
	state "k8s.io/kubernetes/pkg/kubelet/cm/dra/state"
)

type Checkpointer interface {
	GetOrCreate() (*Checkpoint, error)
	Store(*Checkpoint) error
}

type checkpointer struct {
	sync.RWMutex
	checkpointManager checkpointmanager.CheckpointManager
	checkpointName    string
}

// NewCheckpointer creates new checkpointer for keeping track of claim info  with checkpoint backend
func NewCheckpointer(stateDir, checkpointName string) (Checkpointer, error) {
	if len(checkpointName) == 0 {
		return nil, fmt.Errorf("received empty string instead of checkpointName")
	}

	checkpointManager, err := checkpointmanager.NewCheckpointManager(stateDir)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize checkpoint manager: %w", err)
	}
	checkpointer := &checkpointer{
		checkpointManager: checkpointManager,
		checkpointName:    checkpointName,
	}

	return checkpointer, nil
}

// GetOrCreate gets list of claim info states from a checkpoint
// or creates empty list it checkpoint doesn't exist yet
func (sc *checkpointer) GetOrCreate() (*Checkpoint, error) {
	sc.Lock()
	defer sc.Unlock()

	checkpoint, err := NewCheckpoint(nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create new checkpoint: %w", err)
	}

	err = sc.checkpointManager.GetCheckpoint(sc.checkpointName, checkpoint)
	if errors.Is(err, checkpointerrors.ErrCheckpointNotFound) {
		err = sc.store(checkpoint)
		if err != nil {
			return nil, fmt.Errorf("failed to store checkpoint %v: %w", sc.checkpointName, err)
		}
		return checkpoint, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get checkpoint %v: %w", sc.checkpointName, err)
	}

	return checkpoint, nil
}

// Store stores checkpoint to the file
func (sc *checkpointer) Store(checkpoint *Checkpoint) error {
	sc.Lock()
	defer sc.Unlock()

	return sc.store(checkpoint)
}

// store saves state to a checkpoint, caller is responsible for locking
func (sc *checkpointer) store(checkpoint *Checkpoint) error {
	if err := sc.checkpointManager.CreateCheckpoint(sc.checkpointName, checkpoint); err != nil {
		return fmt.Errorf("could not save checkpoint %s: %w", sc.checkpointName, err)
	}
	return nil
}
