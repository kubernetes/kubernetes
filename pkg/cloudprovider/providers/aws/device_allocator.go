/*
Copyright 2016 The Kubernetes Authors.

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

package aws

import "fmt"

// ExistingDevices is a map of assigned devices. Presence of a key with a device
// name in the map means that the device is allocated. Value is irrelevant and
// can be used for anything that DeviceAllocator user wants.
// Only the relevant part of device name should be in the map, e.g. "ba" for
// "/dev/xvdba".
type ExistingDevices map[mountDevice]awsVolumeID

// On AWS, we should assign new (not yet used) device names to attached volumes.
// If we reuse a previously used name, we may get the volume "attaching" forever,
// see https://aws.amazon.com/premiumsupport/knowledge-center/ebs-stuck-attaching/.
// DeviceAllocator finds available device name, taking into account already
// assigned device names from ExistingDevices map. It tries to find the next
// device name to the previously assigned one (from previous DeviceAllocator
// call), so all available device names are used eventually and it minimizes
// device name reuse.
// All these allocations are in-memory, nothing is written to / read from
// /dev directory.
type DeviceAllocator interface {
	// GetNext returns a free device name or error when there is no free device
	// name. Only the device suffix is returned, e.g. "ba" for "/dev/xvdba".
	// It's up to the called to add appropriate "/dev/sd" or "/dev/xvd" prefix.
	GetNext(existingDevices ExistingDevices) (mountDevice, error)
}

type deviceAllocator struct {
	possibleDevices []mountDevice
	lastIndex       int
}

// Allocates device names according to scheme ba..bz, ca..cz
// it moves along the ring and always picks next device until
// device list is exhausted.
func NewDeviceAllocator(lastIndex int) DeviceAllocator {
	possibleDevices := []mountDevice{}
	for _, firstChar := range []rune{'b', 'c'} {
		for i := 'a'; i <= 'z'; i++ {
			dev := mountDevice([]rune{firstChar, i})
			possibleDevices = append(possibleDevices, dev)
		}
	}
	return &deviceAllocator{
		possibleDevices: possibleDevices,
		lastIndex:       lastIndex,
	}
}

func (d *deviceAllocator) GetNext(existingDevices ExistingDevices) (mountDevice, error) {
	var candidate mountDevice
	foundIndex := d.lastIndex
	for {
		candidate, foundIndex = d.nextDevice(foundIndex + 1)
		if _, found := existingDevices[candidate]; !found {
			d.lastIndex = foundIndex
			return candidate, nil
		}
		if foundIndex == d.lastIndex {
			return "", fmt.Errorf("no devices are available")
		}
	}
}

func (d *deviceAllocator) nextDevice(nextIndex int) (mountDevice, int) {
	if nextIndex < len(d.possibleDevices) {
		return d.possibleDevices[nextIndex], nextIndex
	}
	return d.possibleDevices[0], 0
}
