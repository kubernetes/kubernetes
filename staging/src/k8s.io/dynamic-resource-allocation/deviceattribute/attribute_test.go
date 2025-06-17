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
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"k8s.io/utils/ptr"

	resourceapi "k8s.io/dynamic-resource-allocation/api"
)

func TestAddStandardDeviceAttributes(t *testing.T) {
	testPCIBusID := "0123:45:1e.7"
	testACIIntermediateBusID := "0123:46:7.1"
	testPCIRoot := "pci0123:56"

	tests := map[string]struct {
		mockSysfsSetup func(t *testing.T, mockSysfs sysfs)
		opts           []StandardDeviceAttributesOption
		expectedAttrs  map[resourceapi.QualifiedName]resourceapi.DeviceAttribute
		expectsError   bool
		expectedErrMsg string
	}{
		"valid nil attrs": {
			opts:          nil,
			expectedAttrs: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{},
		},
		"valid empty attrs": {
			opts:          []StandardDeviceAttributesOption{},
			expectedAttrs: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{},
		},
		"valid empty pci attrs": {
			opts:          []StandardDeviceAttributesOption{WithStandardPCIDeviceAttributesOpts()},
			expectedAttrs: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{},
		},
		"valid pci attrs": {
			mockSysfsSetup: func(t *testing.T, mockSysfs sysfs) {
				// /sys/bus/pci/devices/0123:45:1e.7 --> /sys/devices/pci0123:56/0123:46:7.1/10123:45:1e.7
				testPCIDevicePath := mockSysfs.Devices(filepath.Join(testPCIRoot, testACIIntermediateBusID, testPCIBusID))
				TouchFile(t, testPCIDevicePath)
				testPCIBusPath := mockSysfs.Bus(filepath.Join("pci", "devices", testPCIBusID))
				CreateSymlink(t, testPCIDevicePath, testPCIBusPath)
			},
			opts: []StandardDeviceAttributesOption{
				WithStandardPCIDeviceAttributesOpts(
					WithPCIDeviceAddress(testPCIBusID),
				),
			},
			expectedAttrs: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				qualifiedNamePCIeRoot: {
					StringValue: ptr.To(testPCIRoot),
				},
			},
		},
		"invalid empty pciBusID": {
			opts: []StandardDeviceAttributesOption{
				WithStandardPCIDeviceAttributesOpts(
					WithPCIDeviceAddress(""),
				),
			},
			expectsError:   true,
			expectedErrMsg: "PCI Bus ID cannot be empty",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			mockSysfsPath := t.TempDir()

			mockSysfs := sysfs(mockSysfsPath)
			if test.mockSysfsSetup != nil {
				test.mockSysfsSetup(t, mockSysfs)
				test.opts = append(test.opts, WithStandardPCIDeviceAttributesOpts(withSysfs(mockSysfs)))
			}

			attrs, err := StandardDeviceAttributes(test.opts...)

			if test.expectsError {
				if err == nil {
					t.Errorf("expected error but got nil")
					return
				}
				if !strings.Contains(err.Error(), test.expectedErrMsg) {
					t.Errorf("expected error message to contain %q, got %q", test.expectedErrMsg, err.Error())
					return
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(attrs, test.expectedAttrs) {
				t.Errorf("expected attributes %v, got %v", test.expectedAttrs, attrs)
				return
			}
		})
	}
}
