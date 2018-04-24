/*
Copyright 2017 The Kubernetes Authors.

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

package store

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint"
)

// Store saves checkpoints and information about which is the current and last-known-good checkpoint to a storage layer
type Store interface {
	// Initialize sets up the storage layer
	Initialize() error

	// Exists returns true if the object referenced by `source` has been checkpointed.
	Exists(source checkpoint.RemoteConfigSource) (bool, error)
	// Save Kubelet config payloads to the storage layer. It must be possible to unmarshal the payload to a KubeletConfiguration.
	// The following payload types are supported:
	// - k8s.io/api/core/v1.ConfigMap
	Save(c checkpoint.Payload) error
	// Load loads the KubeletConfiguration from the checkpoint referenced by `source`.
	Load(source checkpoint.RemoteConfigSource) (*kubeletconfig.KubeletConfiguration, error)

	// CurrentModified returns the last time that the current UID was set
	CurrentModified() (time.Time, error)
	// Current returns the source that points to the current checkpoint, or nil if no current checkpoint is set
	Current() (checkpoint.RemoteConfigSource, error)
	// LastKnownGood returns the source that points to the last-known-good checkpoint, or nil if no last-known-good checkpoint is set
	LastKnownGood() (checkpoint.RemoteConfigSource, error)

	// SetCurrent saves the source that points to the current checkpoint, set to nil to unset
	SetCurrent(source checkpoint.RemoteConfigSource) error
	// SetLastKnownGood saves the source that points to the last-known-good checkpoint, set to nil to unset
	SetLastKnownGood(source checkpoint.RemoteConfigSource) error
	// Reset unsets the current and last-known-good UIDs and returns whether the current UID was unset as a result of the reset
	Reset() (bool, error)
}

// reset is a helper for implementing Reset, which can be implemented in terms of Store methods
func reset(s Store) (bool, error) {
	current, err := s.Current()
	if err != nil {
		return false, err
	}
	if err := s.SetLastKnownGood(nil); err != nil {
		return false, fmt.Errorf("failed to reset last-known-good UID in checkpoint store, error: %v", err)
	}
	if err := s.SetCurrent(nil); err != nil {
		return false, fmt.Errorf("failed to reset current UID in checkpoint store, error: %v", err)
	}
	return current != nil, nil
}
