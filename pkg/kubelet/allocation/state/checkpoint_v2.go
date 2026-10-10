/*
Copyright The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"sort"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
)

const (
	// checkpointVersionV2 is the version of checkpoints whose Data holds a protobuf-encoded v1.PodList.
	checkpointVersionV2 = "v2"
)

// CheckpointData is the V2 payload, stored as JSON in Checkpoint.Data.
type CheckpointData struct {
	// PodListProto is the protobuf encoding of the v1.PodList holding the state's pods. It is what
	// this kubelet reads back. encoding/json stores []byte as base64.
	PodListProto []byte `json:"podList,omitempty"`

	// Entries is the legacy V1 representation of the pods in PodListProto. It is written only so that
	// a kubelet that predates V2 can still read this checkpoint after a rollback; this kubelet never
	// reads it.
	//
	// Deprecated: Use PodListProto.
	Entries PodResourceInfoMap `json:"entries,omitempty"`
}

// NewCheckpointV2 creates a V2 checkpoint holding the given pods, along with their legacy V1
// representation for rollbacks.
func NewCheckpointV2(podList *v1.PodList) (*Checkpoint, error) {
	if podList == nil {
		podList = &v1.PodList{}
	}

	// Sort a copy so that the caller's list keeps its order.
	sorted := &v1.PodList{
		TypeMeta: podList.TypeMeta,
		ListMeta: podList.ListMeta,
		Items:    make([]v1.Pod, len(podList.Items)),
	}
	copy(sorted.Items, podList.Items)

	// Pods are sorted by UID so that identical state always serializes to
	// identical bytes, which lets callers skip rewriting an unchanged checkpoint by comparing checksums.
	sort.Slice(sorted.Items, func(i, j int) bool { return sorted.Items[i].UID < sorted.Items[j].UID })

	protoBytes, err := sorted.Marshal()
	if err != nil {
		return nil, fmt.Errorf("failed to marshal PodList to protobuf for checkpointing: %w", err)
	}

	data, err := json.Marshal(CheckpointData{
		PodListProto: protoBytes,
		Entries:      podListToV1Entries(sorted),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to serialize checkpoint data: %w", err)
	}

	cp := &Checkpoint{
		Version: checkpointVersionV2,
		Data:    string(data),
	}
	cp.Checksum = checksum.New(cp.Data)
	return cp, nil
}

// GetPodList returns the pods stored in the checkpoint. migrated is true when the checkpoint predates
// V2 and the list was rebuilt from its legacy V1 entries, so the caller should persist it as V2.
func (cp *Checkpoint) GetPodList() (*v1.PodList, bool, error) {
	switch cp.Version {
	case checkpointVersionV2:
		var data CheckpointData
		if err := json.Unmarshal([]byte(cp.Data), &data); err != nil {
			return nil, false, fmt.Errorf("failed to unmarshal V2 checkpoint data: %w", err)
		}
		podList := &v1.PodList{}
		if err := podList.Unmarshal(data.PodListProto); err != nil {
			return nil, false, fmt.Errorf("failed to unmarshal protobuf PodList: %w", err)
		}
		return podList, false, nil

	case "":
		// V1 checkpoints carry no version.
		podList, err := migrateV1ToV2(cp.Data)
		if err != nil {
			return nil, false, fmt.Errorf("failed to migrate legacy V1 checkpoint: %w", err)
		}
		return podList, true, nil

	default:
		return nil, false, fmt.Errorf("unsupported checkpoint version %q: this kubelet reads legacy (unversioned) and %q checkpoints", cp.Version, checkpointVersionV2)
	}
}
