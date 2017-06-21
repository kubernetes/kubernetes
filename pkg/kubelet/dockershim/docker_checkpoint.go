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

package dockershim

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"path/filepath"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/errors"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
)

const (
	// default directory to store pod sandbox checkpoint files
	sandboxCheckpointDir = "sandbox"
	protocolTCP          = Protocol("tcp")
	protocolUDP          = Protocol("udp")
	schemaVersion        = "v1"
)

type Protocol string

// PortMapping is the port mapping configurations of a sandbox.
type PortMapping struct {
	// Protocol of the port mapping.
	Protocol *Protocol `json:"protocol,omitempty"`
	// Port number within the container.
	ContainerPort *int32 `json:"container_port,omitempty"`
	// Port number on the host.
	HostPort *int32 `json:"host_port,omitempty"`
}

// CheckpointData contains all types of data that can be stored in the checkpoint.
type CheckpointData struct {
	PortMappings []*PortMapping `json:"port_mappings,omitempty"`
	HostNetwork  bool           `json:"host_network,omitempty"`
}

// PodSandboxCheckpoint is the checkpoint structure for a sandbox
type PodSandboxCheckpoint struct {
	// Version of the pod sandbox checkpoint schema.
	Version string `json:"version"`
	// Pod name of the sandbox. Same as the pod name in the PodSpec.
	Name string `json:"name"`
	// Pod namespace of the sandbox. Same as the pod namespace in the PodSpec.
	Namespace string `json:"namespace"`
	// Data to checkpoint for pod sandbox.
	Data *CheckpointData `json:"data,omitempty"`
	// Checksum is calculated with fnv hash of the checkpoint object with checksum field set to be zero
	CheckSum uint64 `json:"checksum"`
}

// CheckpointHandler provides the interface to manage PodSandbox checkpoint
type CheckpointHandler interface {
	// CreateCheckpoint persists sandbox checkpoint in CheckpointStore.
	CreateCheckpoint(podSandboxID string, checkpoint *PodSandboxCheckpoint) error
	// GetCheckpoint retrieves sandbox checkpoint from CheckpointStore.
	GetCheckpoint(podSandboxID string) (*PodSandboxCheckpoint, error)
	// RemoveCheckpoint removes sandbox checkpoint form CheckpointStore.
	// WARNING: RemoveCheckpoint will not return error if checkpoint does not exist.
	RemoveCheckpoint(podSandboxID string) error
	// ListCheckpoint returns the list of existing checkpoints.
	ListCheckpoints() ([]string, error)
}

// PersistentCheckpointHandler is an implementation of CheckpointHandler. It persists checkpoint in CheckpointStore
type PersistentCheckpointHandler struct {
	store CheckpointStore
}

func NewPersistentCheckpointHandler(dockershimRootDir string) (CheckpointHandler, error) {
	fstore, err := NewFileStore(filepath.Join(dockershimRootDir, sandboxCheckpointDir))
	if err != nil {
		return nil, err
	}
	return &PersistentCheckpointHandler{store: fstore}, nil
}

func (handler *PersistentCheckpointHandler) CreateCheckpoint(podSandboxID string, checkpoint *PodSandboxCheckpoint) error {
	checkpoint.CheckSum = calculateChecksum(*checkpoint)
	blob, err := json.Marshal(checkpoint)
	if err != nil {
		return err
	}
	return handler.store.Write(podSandboxID, blob)
}

func (handler *PersistentCheckpointHandler) GetCheckpoint(podSandboxID string) (*PodSandboxCheckpoint, error) {
	blob, err := handler.store.Read(podSandboxID)
	if err != nil {
		return nil, err
	}
	var checkpoint PodSandboxCheckpoint
	//TODO: unmarhsal into a struct with just Version, check version, unmarshal into versioned type.
	err = json.Unmarshal(blob, &checkpoint)
	if err != nil {
		glog.Errorf("Failed to unmarshal checkpoint %q. Checkpoint content: %q. ErrMsg: %v", podSandboxID, string(blob), err)
		return &checkpoint, errors.CorruptCheckpointError
	}
	if checkpoint.CheckSum != calculateChecksum(checkpoint) {
		glog.Errorf("Checksum of checkpoint %q is not valid", podSandboxID)
		return &checkpoint, errors.CorruptCheckpointError
	}
	return &checkpoint, nil
}

func (handler *PersistentCheckpointHandler) RemoveCheckpoint(podSandboxID string) error {
	return handler.store.Delete(podSandboxID)
}

func (handler *PersistentCheckpointHandler) ListCheckpoints() ([]string, error) {
	keys, err := handler.store.List()
	if err != nil {
		return []string{}, fmt.Errorf("failed to list checkpoint store: %v", err)
	}
	return keys, nil
}

func NewPodSandboxCheckpoint(namespace, name string) *PodSandboxCheckpoint {
	return &PodSandboxCheckpoint{
		Version:   schemaVersion,
		Namespace: namespace,
		Name:      name,
		Data:      &CheckpointData{},
	}
}

func calculateChecksum(checkpoint PodSandboxCheckpoint) uint64 {
	checkpoint.CheckSum = 0
	hash := fnv.New32a()
	hashutil.DeepHashObject(hash, checkpoint)
	return uint64(hash.Sum32())
}
