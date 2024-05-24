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
	"fmt"
	"hash/fnv"
	"strings"

	"k8s.io/apimachinery/pkg/util/dump"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/errors"
)

// PodDevicesEntryV1 connects pod information to devices, without topology information (k8s <= 1.19)
type PodDevicesEntryV1 struct {
	PodUID        string
	ContainerName string
	ResourceName  string
	DeviceIDs     []string
	AllocResp     []byte
}

// checkpointDataV1 struct is used to store pod to device allocation information
// in a checkpoint file, without topology information (k8s <= 1.19)
type checkpointDataV1 struct {
	PodDeviceEntries  []PodDevicesEntryV1
	RegisteredDevices map[string][]string
}

// checksum compute the checksum using the same algorithms (and data type names) k8s 1.19 used.
// We need this special code path to be able to correctly validate the checksum k8s 1.19 wrote.
// credits to https://github.com/kubernetes/kubernetes/pull/102717/commits/353f93895118d2ffa2d59a29a1fbc225160ea1d6
func (cp checkpointDataV1) checksum() checksum.Checksum {
	object := dump.ForHash(cp)
	object = strings.Replace(object, "checkpointDataV1", "checkpointData", 1)
	object = strings.Replace(object, "PodDevicesEntryV1", "PodDevicesEntry", -1)
	hash := fnv.New32a()
	fmt.Fprintf(hash, "%v", object)
	return checksum.Checksum(hash.Sum32())
}

// DataV1 holds checkpoint data and its checksum, in V1 (k8s <= 1.19) format
type DataV1 struct {
	Data     checkpointDataV1
	Checksum checksum.Checksum
}

// NewV1 returns an instance of Checkpoint, in V1 (k8s <= 1.19) format.
// Users should avoid creating checkpoints in formats different from the most recent one,
// use the old formats only to validate existing checkpoint and convert them to most recent
// format. The only exception should be test code.
func NewV1(devEntries []PodDevicesEntryV1,
	devices map[string][]string) DeviceManagerCheckpoint {
	return &DataV1{
		Data: checkpointDataV1{
			PodDeviceEntries:  devEntries,
			RegisteredDevices: devices,
		},
	}
}

// MarshalCheckpoint is needed to implement the Checkpoint interface, but should not be called anymore
func (cp *DataV1) MarshalCheckpoint() ([]byte, error) {
	klog.InfoS("Marshalling a device manager V1 checkpoint")
	cp.Checksum = cp.Data.checksum()
	return json.Marshal(*cp)
}

// UnmarshalCheckpoint returns unmarshalled data
func (cp *DataV1) UnmarshalCheckpoint(blob []byte) error {
	return json.Unmarshal(blob, cp)
}

// VerifyChecksum verifies that passed checksum is same as calculated checksum
func (cp *DataV1) VerifyChecksum() error {
	if cp.Checksum != cp.Data.checksum() {
		return errors.ErrCorruptCheckpoint
	}
	return nil
}

// GetDataInLatestFormat returns device entries and registered devices in the *most recent*
// checkpoint format, *not* in the original format stored on disk.
func (cp *DataV1) GetDataInLatestFormat() ([]PodDevicesEntry, map[string][]string) {
	var podDevs []PodDevicesEntry
	for _, entryV1 := range cp.Data.PodDeviceEntries {
		devsPerNuma := NewDevicesPerNUMA()
		// no NUMA cell affinity was recorded. The only possible choice
		// is to set all the devices affine to node 0.
		devsPerNuma[0] = entryV1.DeviceIDs
		podDevs = append(podDevs, PodDevicesEntry{
			PodUID:        entryV1.PodUID,
			ContainerName: entryV1.ContainerName,
			ResourceName:  entryV1.ResourceName,
			DeviceIDs:     devsPerNuma,
			AllocResp:     entryV1.AllocResp,
		})
	}
	return podDevs, cp.Data.RegisteredDevices
}
