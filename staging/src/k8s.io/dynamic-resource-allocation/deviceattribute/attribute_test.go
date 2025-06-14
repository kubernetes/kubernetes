/*
Copyright 2025 The Kubernetes Authors.

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

package deviceattribute

import (
	"os"
	"reflect"
	"testing"

	resourceapi "k8s.io/dynamic-resource-allocation/api"
	dratesting "k8s.io/dynamic-resource-allocation/testing"
	"k8s.io/dynamic-resource-allocation/utils"
	"k8s.io/utils/ptr"
)

func TestAddStandardDeviceAttributes(t *testing.T) {
	// Create mock sysfs for testing
	testSysfsDir, err := os.MkdirTemp("", "test-sysfs")
	if err != nil {
		t.Fatalf("failed to create temp sysfs directory: %v", err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(testSysfsDir); err != nil {
			t.Fatalf("failed to clean up temp sysfs directory: %v", err)
		}
	})

	testSysFs := utils.NewSysfsWithRoot(testSysfsDir)
	// "0123:45:1e.7"
	testPCIAddress := utils.MustNewPCIAddress(0x0123, 0x45, 0x1e, 0x7)
	// "1234:56"
	testPCIRoot := utils.MustNewPCIRoot(0x1234, 0x56)

	// /sys/bus/pci/devices/0123:45:1e.7 --> /sys/devices/pci1234:56/0123:45:1e.7
	testPCIDevicePath := testSysFs.Devices("pci" + testPCIRoot.String() + "/" + testPCIAddress.String())
	dratesting.TouchFile(t, testPCIDevicePath)
	testPCIBusPath := testSysFs.Bus("pci/devices/" + testPCIAddress.String())
	dratesting.CreateSymlink(t, testPCIDevicePath, testPCIBusPath)

	tests := map[string]struct {
		opts          []StandardDeviceAttributesOption
		expectedAttrs map[resourceapi.QualifiedName]resourceapi.DeviceAttribute
	}{
		"pci attrs": {
			opts: []StandardDeviceAttributesOption{
				WithStandardPCIDeviceAttributesOpts(
					WithPCIDeviceAddress(testPCIAddress),
					withSysfs(testSysFs), // Use the mock sysfs
				),
			},
			expectedAttrs: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				qualifiedNamePCIeRoot: {
					StringValue: ptr.To("pci" + testPCIRoot.String()),
				},
			},
		},
		"nil attrs": {
			opts:          nil,
			expectedAttrs: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{},
		},
		"empty attrs": {
			opts:          []StandardDeviceAttributesOption{},
			expectedAttrs: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{},
		},
		"empty pci attrs": {
			opts:          []StandardDeviceAttributesOption{WithStandardPCIDeviceAttributesOpts()},
			expectedAttrs: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			attrs, err := StandardDeviceAttributes(test.opts...)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(attrs, test.expectedAttrs) {
				t.Errorf("expected %v, got %v", test.expectedAttrs, attrs)
			}
		})
	}
}
