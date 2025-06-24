/*
Copyright 2024 The Kubernetes Authors.

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

package structured

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sync"

	draapi "k8s.io/dynamic-resource-allocation/api"
)

type SharedDeviceIDList map[SharedDeviceID]struct{}

type SharedDeviceID struct {
	Driver, Pool, Device, ShareID draapi.UniqueString
}

func (s SharedDeviceIDList) Clone() SharedDeviceIDList {
	cloneList := make(SharedDeviceIDList, len(s))
	for k, v := range s {
		cloneList[k] = v
	}
	return cloneList
}

func MakeSharedDeviceID(deviceID DeviceID, ShareID string) SharedDeviceID {
	return SharedDeviceID{
		Driver:  deviceID.Driver,
		Pool:    deviceID.Pool,
		Device:  deviceID.Device,
		ShareID: draapi.MakeUniqueString(ShareID),
	}
}

func (d SharedDeviceID) String() string {
	return d.Driver.String() + "/" + d.Pool.String() + "/" + GetSharedDeviceName(d.Device.String(), d.ShareID.String())
}

func GetSharedDeviceName(device, ShareID string) string {
	return device + "/" + ShareID
}

type UniqueHexStringFactory struct {
	mu      sync.Mutex
	usedIDs SharedDeviceIDList
	nBytes  int
}

func NewUniqueHexStringFactory(nBytes int) *UniqueHexStringFactory {
	return &UniqueHexStringFactory{
		usedIDs: make(SharedDeviceIDList, 0),
		nBytes:  nBytes,
	}
}

func (f *UniqueHexStringFactory) SetUsedShareIDs(usedIDs SharedDeviceIDList) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if usedIDs != nil {
		f.usedIDs = usedIDs
	}
}

// GenerateNewShareID generates a new random hexadecimal string of length nBytes*2.
// It combines the generated string with the given driver, pool, and device identifiers
// to form a composite key, ensuring uniqueness within the factory's usedIDs map.
//
// The function attempts up to maxTry times to generate a unique ID. If a unique ID
// is found, it is added to the usedIDs map and returned. If all attempts fail,
// an error is returned.
func (f *UniqueHexStringFactory) GenerateNewShareID(deviceID DeviceID, maxTry int) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	count := 0
	for {
		b := make([]byte, f.nBytes)
		_, err := rand.Read(b)
		if err != nil {
			return "", fmt.Errorf("failed to generate random bytes: %w", err)
		}
		ShareID := hex.EncodeToString(b)
		sharedDeviceID := MakeSharedDeviceID(deviceID, ShareID)
		if _, exists := f.usedIDs[sharedDeviceID]; !exists {
			f.usedIDs[sharedDeviceID] = struct{}{} // Mark UID as used
			return ShareID, nil
		}
		count += 1
		if count > maxTry {
			return "", fmt.Errorf("failed to find unique hex string within %d try", maxTry)
		}
	}
}

func (f *UniqueHexStringFactory) DeleteShareID(deviceID DeviceID, ShareID string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	sharedDeviceID := MakeSharedDeviceID(deviceID, ShareID)
	if _, exists := f.usedIDs[sharedDeviceID]; !exists {
		delete(f.usedIDs, sharedDeviceID)
	}
}
