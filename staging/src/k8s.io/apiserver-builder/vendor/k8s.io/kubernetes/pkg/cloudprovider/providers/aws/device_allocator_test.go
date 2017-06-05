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
		lastIndex       int
		expectedOutput  mountDevice
	}{
		{
			"empty device list",
			ExistingDevices{},
			0,
			"bb",
		},
		{
			"empty device list with wrap",
			ExistingDevices{},
			51,
			"ba", // next to 'zz' is the first one, 'ba'
		},
		{
			"device list",
			ExistingDevices{"ba": "used", "bb": "used", "bc": "used"},
			0,
			"bd",
		},
		{
			"device list with wrap",
			ExistingDevices{"cy": "used", "cz": "used", "ba": "used"},
			49,
			"bb", // "cy", "cz" and "ba" are used
		},
	}

	for _, test := range tests {
		allocator := NewDeviceAllocator(test.lastIndex).(*deviceAllocator)

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
	allocator := NewDeviceAllocator(0).(*deviceAllocator)
	existingDevices := ExistingDevices{}

	// make all devices used
	var first, second byte
	for first = 'b'; first <= 'c'; first++ {
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
