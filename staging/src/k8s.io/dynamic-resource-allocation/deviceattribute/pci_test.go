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
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"k8s.io/utils/ptr"

	resourceapi "k8s.io/dynamic-resource-allocation/api"
)

func TestStandardPCIDeviceAttributes(t *testing.T) {
	pciBusID := "0000:01:02.3"
	pciIntermediateBusID := "0000:00:13.1"
	pcieRoot := "pci0000:01"
	tests := map[string]struct {
		smockSysfsSetup func(t *testing.T, mockSysfs sysfs)
		opts            []StandardPCIDeviceAttributesOption
		expected        map[resourceapi.QualifiedName]resourceapi.DeviceAttribute
		expectsError    bool
		expectedErrMsg  string
	}{
		"valid": {
			smockSysfsSetup: func(t *testing.T, mockSysfs sysfs) {
				devicePath := mockSysfs.Devices(filepath.Join(pcieRoot, pciIntermediateBusID, pciBusID))
				TouchFile(t, devicePath)
				busPath := mockSysfs.Bus(filepath.Join("pci", "devices", pciBusID))
				CreateSymlink(t, devicePath, busPath)
			},
			opts: []StandardPCIDeviceAttributesOption{
				WithPCIDeviceAddress(pciBusID),
			},
			expected: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				qualifiedNamePCIeRoot: {
					StringValue: ptr.To(pcieRoot),
				},
			},
		},
		"invalid empty PCI Bus ID": {
			opts: []StandardPCIDeviceAttributesOption{
				WithPCIDeviceAddress(""),
			},
			expected:       nil,
			expectsError:   true,
			expectedErrMsg: "PCI Bus ID cannot be empty",
		},
		"invalid PCI Bus ID format": {
			opts: []StandardPCIDeviceAttributesOption{
				WithPCIDeviceAddress("invalid-pci-id"),
			},
			expected:       nil,
			expectsError:   true,
			expectedErrMsg: "invalid PCI Bus ID format: invalid-pci-id",
		},
		"invalid no exist PCI Bus ID": {
			opts: []StandardPCIDeviceAttributesOption{
				WithPCIDeviceAddress(pciBusID),
			},
			expected:       nil,
			expectsError:   true,
			expectedErrMsg: "no such file or directory",
		},
		"invalid symlink": {
			smockSysfsSetup: func(t *testing.T, mockSysfs sysfs) {
				devicePath := mockSysfs.Devices(filepath.Join("invalid-pci-root", "0000:00:13.1", pciBusID))
				TouchFile(t, devicePath)
				busPath := mockSysfs.Bus(filepath.Join("pci", "devices", pciBusID))
				CreateSymlink(t, devicePath, busPath)
			},
			opts: []StandardPCIDeviceAttributesOption{
				WithPCIDeviceAddress(pciBusID),
			},
			expected:       nil,
			expectsError:   true,
			expectedErrMsg: "invalid symlink target for PCI Bus ID",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			mockSysfsPath := t.TempDir()

			mockSysfs := sysfs(mockSysfsPath)
			if test.smockSysfsSetup != nil {
				test.smockSysfsSetup(t, mockSysfs)
			}

			got, err := StandardPCIDeviceAttributes(append(test.opts, withSysfs(mockSysfs))...)
			if test.expectsError {
				if err == nil {
					t.Errorf("Expected error but got none")
					return
				}
				if !strings.Contains(err.Error(), test.expectedErrMsg) {
					t.Errorf("Expected error message to contain %q, got %q", test.expectedErrMsg, err.Error())
					return
				}
				return
			}
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			if !reflect.DeepEqual(got, test.expected) {
				t.Errorf("Expected attributes %v, got %v", test.expected, got)
			}
		})
	}
}

func TouchFile(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatalf("Failed to create directory %s: %v", filepath.Dir(path), err)
	}
	if _, err := os.Create(path); err != nil {
		t.Fatalf("Failed to create file %s: %v", path, err)
	}
}

func CreateSymlink(t *testing.T, target, link string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(link), 0755); err != nil {
		t.Fatalf("Failed to create directory for symlink %s: %v", filepath.Dir(link), err)
	}
	if err := os.Symlink(target, link); err != nil {
		t.Fatalf("Failed to create symlink from %s to %s: %v", target, link, err)
	}
}
