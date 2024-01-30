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

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
)

// DeviceManagerCheckpoint defines the operations to retrieve pod devices
type DeviceManagerCheckpoint interface {
	checkpointmanager.Checkpoint
	GetDataInLatestFormat() ([]PodDevicesEntry, map[string][]string)
}

// DevicesPerNUMA represents device ids obtained from device plugin per NUMA node id
type DevicesPerNUMA map[int64][]string

// PodDevicesEntry connects pod information to devices
type PodDevicesEntry struct {
	PodUID        string
	ContainerName string
	ResourceName  string
	DeviceIDs     DevicesPerNUMA
	AllocResp     []byte
}

// checkpointData struct is used to store pod to device allocation information
// in a checkpoint file.
// TODO: add version control when we need to change checkpoint format.
type checkpointData struct {
	PodDeviceEntries  []PodDevicesEntry
	RegisteredDevices map[string][]string
}

// Data holds checkpoint data and its checksum
type Data struct {
	Data     checkpointData
	Checksum checksum.Checksum
}

// NewDevicesPerNUMA is a function that creates DevicesPerNUMA map
func NewDevicesPerNUMA() DevicesPerNUMA {
	return make(DevicesPerNUMA)
}

// Devices is a function that returns all device ids for all NUMA nodes
// and represent it as sets.Set[string]
func (dev DevicesPerNUMA) Devices() sets.Set[string] {
	result := sets.New[string]()

	for _, devs := range dev {
		result.Insert(devs...)
	}
	return result
}

// New returns an instance of Checkpoint - must be an alias for the most recent version
func New(devEntries []PodDevicesEntry, devices map[string][]string) DeviceManagerCheckpoint {
	return newV2(devEntries, devices)
}

func newV2(devEntries []PodDevicesEntry, devices map[string][]string) DeviceManagerCheckpoint {
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

// GetDataInLatestFormat returns device entries and registered devices in the *most recent*
// checkpoint format, *not* in the original format stored on disk.
func (cp *Data) GetDataInLatestFormat() ([]PodDevicesEntry, map[string][]string) {
	return cp.Data.PodDeviceEntries, cp.Data.RegisteredDevices
}
