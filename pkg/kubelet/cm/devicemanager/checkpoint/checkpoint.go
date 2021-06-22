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
	"hash/fnv"
	"strings"

	"github.com/davecgh/go-spew/spew"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
	"k8s.io/kubernetes/pkg/kubelet/cm/devicemanager/device"
)

// DeviceManagerCheckpoint defines the operations to retrieve pod devices
type DeviceManagerCheckpoint interface {
	checkpointmanager.Checkpoint
	GetData() ([]PodDevicesEntry, map[string][]string, device.ResourceDeviceInstances)
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
type checkpointDataV1 struct {
	PodDeviceEntries  []PodDevicesEntry
	RegisteredDevices map[string][]string
}

// checkpointData struct is used to store pod to device allocation information
// in a checkpoint file.
// version 2: add AllDevices field
type checkpointDataV2 struct {
	PodDeviceEntries  []PodDevicesEntry
	RegisteredDevices map[string][]string
	AllDevices        device.ResourceDeviceInstances `json:"AllDevices,omitempty"`
}

// Data holds checkpoint data and its checksum
type Data struct {
	Data     checkpointDataV2
	Checksum checksum.Checksum
}

// NewDevicesPerNUMA is a function that creates DevicesPerNUMA map
func NewDevicesPerNUMA() DevicesPerNUMA {
	return make(DevicesPerNUMA)
}

// Devices is a function that returns all device ids for all NUMA nodes
// and represent it as sets.String
func (dev DevicesPerNUMA) Devices() sets.String {
	result := sets.NewString()

	for _, devs := range dev {
		result.Insert(devs...)
	}
	return result
}

// New returns an instance of Checkpoint
func New(devEntries []PodDevicesEntry,
	devices map[string][]string, allDevices device.ResourceDeviceInstances) DeviceManagerCheckpoint {
	return &Data{
		Data: checkpointDataV2{
			PodDeviceEntries:  devEntries,
			RegisteredDevices: devices,
			AllDevices:        allDevices,
		},
	}
}

// checksum Compatible with the old version
// Notice: muse be `checkpointDataV1` not `*checkpointDataV1`
func (v1 checkpointDataV1) checksum() checksum.Checksum {
	printer := spew.ConfigState{
		Indent:         " ",
		SortKeys:       true,
		DisableMethods: true,
		SpewKeys:       true,
	}

	object := printer.Sprintf("%#v", v1)
	object = strings.Replace(object, "checkpointDataV1", "checkpointData", 1)
	hash := fnv.New32a()
	printer.Fprintf(hash, "%v", object)
	return checksum.Checksum(hash.Sum32())
}

// MarshalCheckpoint returns marshalled data
func (cp *Data) MarshalCheckpoint() ([]byte, error) {
	if len(cp.Data.AllDevices) == 0 {
		v1 := checkpointDataV1{
			PodDeviceEntries:  cp.Data.PodDeviceEntries,
			RegisteredDevices: cp.Data.RegisteredDevices,
		}
		cp.Checksum = v1.checksum()
	} else {
		cp.Checksum = checksum.New(cp.Data)
	}
	return json.Marshal(*cp)
}

// UnmarshalCheckpoint returns unmarshalled data
func (cp *Data) UnmarshalCheckpoint(blob []byte) error {
	return json.Unmarshal(blob, cp)
}

// VerifyChecksum verifies that passed checksum is same as calculated checksum
func (cp *Data) VerifyChecksum() error {
	if len(cp.Data.AllDevices) == 0 {
		v1 := checkpointDataV1{
			PodDeviceEntries:  cp.Data.PodDeviceEntries,
			RegisteredDevices: cp.Data.RegisteredDevices,
		}
		if cp.Checksum == v1.checksum() {
			return nil
		}
	}
	return cp.Checksum.Verify(cp.Data)
}

// GetData returns device entries and registered device
func (cp *Data) GetData() ([]PodDevicesEntry, map[string][]string, device.ResourceDeviceInstances) {
	return cp.Data.PodDeviceEntries, cp.Data.RegisteredDevices, cp.Data.AllDevices
}
