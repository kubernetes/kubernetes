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

	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint"
)

// Store saves checkpoints and information about which is the assigned and last-known-good checkpoint to a storage layer
type Store interface {
	// Initialize sets up the storage layer
	Initialize() error

	// Exists returns true if the object referenced by `source` has been checkpointed.
	// The source must be unambiguous - e.g. if referencing an API object it must specify both uid and resourceVersion.
	Exists(source checkpoint.RemoteConfigSource) (bool, error)
	// Save Kubelet config payloads to the storage layer. It must be possible to unmarshal the payload to a KubeletConfiguration.
	// The following payload types are supported:
	// - k8s.io/api/core/v1.ConfigMap
	Save(c checkpoint.Payload) error
	// Load loads the KubeletConfiguration from the checkpoint referenced by `source`.
	Load(source checkpoint.RemoteConfigSource) (*kubeletconfig.KubeletConfiguration, error)

	// AssignedModified returns the last time that the assigned checkpoint was set
	AssignedModified() (time.Time, error)
	// Assigned returns the source that points to the checkpoint currently assigned to the Kubelet, or nil if no assigned checkpoint is set
	Assigned() (checkpoint.RemoteConfigSource, error)
	// LastKnownGood returns the source that points to the last-known-good checkpoint, or nil if no last-known-good checkpoint is set
	LastKnownGood() (checkpoint.RemoteConfigSource, error)

	// SetAssigned saves the source that points to the assigned checkpoint, set to nil to unset
	SetAssigned(source checkpoint.RemoteConfigSource) error
	// SetLastKnownGood saves the source that points to the last-known-good checkpoint, set to nil to unset
	SetLastKnownGood(source checkpoint.RemoteConfigSource) error
	// Reset unsets the assigned and last-known-good checkpoints and returns whether the assigned checkpoint was unset as a result of the reset
	Reset() (bool, error)
}

// reset is a helper for implementing Reset, which can be implemented in terms of Store methods
func reset(s Store) (bool, error) {
	assigned, err := s.Assigned()
	if err != nil {
		return false, err
	}
	if err := s.SetLastKnownGood(nil); err != nil {
		return false, fmt.Errorf("failed to reset last-known-good UID in checkpoint store, error: %v", err)
	}
	if err := s.SetAssigned(nil); err != nil {
		return false, fmt.Errorf("failed to reset assigned UID in checkpoint store, error: %v", err)
	}
	return assigned != nil, nil
}
