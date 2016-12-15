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

import "testing"

func TestDeviceAllocator(t *testing.T) {
	tests := []struct {
		name            string
		existingDevices ExistingDevices
		length          int
		firstDevice     mountDevice
		lastAllocated   mountDevice
		expectedOutput  mountDevice
	}{
		{
			"empty device list",
			ExistingDevices{},
			2,
			"aa",
			"aa",
			"ab",
		},
		{
			"empty device list with wrap",
			ExistingDevices{},
			2,
			"ba",
			"zz",
			"ba", // next to 'zz' is the first one, 'ba'
		},
		{
			"device list",
			ExistingDevices{"aa": "used", "ab": "used", "ac": "used"},
			2,
			"aa",
			"aa",
			"ad", // all up to "ac" are used
		},
		{
			"device list with wrap",
			ExistingDevices{"zy": "used", "zz": "used", "ba": "used"},
			2,
			"ba",
			"zx",
			"bb", // "zy", "zz" and "ba" are used
		},
		{
			"three characters with wrap",
			ExistingDevices{"zzy": "used", "zzz": "used", "baa": "used"},
			3,
			"baa",
			"zzx",
			"bab",
		},
	}

	for _, test := range tests {
		allocator := NewDeviceAllocator(test.length, test.firstDevice).(*deviceAllocator)
		allocator.lastAssignedDevice = test.lastAllocated

		got, err := allocator.GetNext(test.existingDevices)
		if err != nil {
			t.Errorf("text %q: unexpected error: %v", test.name, err)
		}
		if got != test.expectedOutput {
			t.Errorf("text %q: expected %q, got %q", test.name, test.expectedOutput, got)
		}
	}
}

func TestDeviceAllocatorError(t *testing.T) {
	allocator := NewDeviceAllocator(2, "ba").(*deviceAllocator)
	existingDevices := ExistingDevices{}

	// make all devices used
	var first, second byte
	for first = 'b'; first <= 'z'; first++ {
		for second = 'a'; second <= 'z'; second++ {
			device := [2]byte{first, second}
			existingDevices[mountDevice(device[:])] = "used"
		}
	}

	device, err := allocator.GetNext(existingDevices)
	if err == nil {
		t.Errorf("expected error, got device  %q", device)
	}
}
