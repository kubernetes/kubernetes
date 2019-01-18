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

package checkpoint

import (
	"encoding/json"

	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
)

type DeviceManagerCheckpoint interface {
	checkpointmanager.Checkpoint
	GetData() ([]PodDevicesEntry, map[string][]string)
}

type PodDevicesEntry struct {
	PodUID        string
	ContainerName string
	ResourceName  string
	DeviceIDs     []string
	AllocResp     []byte
}

// checkpointData struct is used to store pod to device allocation information
// in a checkpoint file.
// TODO: add version control when we need to change checkpoint format.
type checkpointData struct {
	PodDeviceEntries  []PodDevicesEntry
	RegisteredDevices map[string][]string
}

type Data struct {
	Data     checkpointData
	Checksum checksum.Checksum
}

// NewDeviceManagerCheckpoint returns an instance of Checkpoint
func New(devEntries []PodDevicesEntry,
	devices map[string][]string) DeviceManagerCheckpoint {
	return &Data{
		Data: checkpointData{
			PodDeviceEntries:  devEntries,
			RegisteredDevices: devices,
		},
	}
}

// MarshalCheckpoint returns marshalled data
func (cp *Data) MarshalCheckpoint() ([]byte, error) {
	cp.Checksum = checksum.New(cp.Data)
	return json.Marshal(*cp)
}

// UnmarshalCheckpoint returns unmarshalled data
func (cp *Data) UnmarshalCheckpoint(blob []byte) error {
	return json.Unmarshal(blob, cp)
}

// VerifyChecksum verifies that passed checksum is same as calculated checksum
func (cp *Data) VerifyChecksum() error {
	return cp.Checksum.Verify(cp.Data)
}

func (cp *Data) GetData() ([]PodDevicesEntry, map[string][]string) {
	return cp.Data.PodDeviceEntries, cp.Data.RegisteredDevices
}
