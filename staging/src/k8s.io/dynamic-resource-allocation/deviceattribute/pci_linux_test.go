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

func TestGetPCIeRootBAttributeyPCIBusID(t *testing.T) {
	pciBusID := "0000:01:02.3"
	pcieRoot := "pci0000:01"
	expectedAttribute := DeviceAttribute{
		Name:  StandardDeviceAttributePCIeRoot,
		Value: resourceapi.DeviceAttribute{StringValue: ptr.To(pcieRoot)},
	}

	tests := map[string]struct {
		mockSysfsSetup    func(t *testing.T, mockSysfs sysfsPath)
		address           string
		expectedAttribute *DeviceAttribute
		expectsError      bool
		expectedErrMsg    string
	}{
		"valid": {
			mockSysfsSetup: func(t *testing.T, mockSysfs sysfsPath) {
				devicePath := mockSysfs.Devices(filepath.Join(pcieRoot, "0000:00:13.1", pciBusID))
				TouchFile(t, devicePath)
				busPath := mockSysfs.Bus(filepath.Join("pci", "devices", pciBusID))
				CreateSymlink(t, devicePath, busPath)
			},
			address:           pciBusID,
			expectedAttribute: &expectedAttribute,
			expectsError:      false,
		},
		"invalid empty PCI Bus ID": {
			address:           "",
			expectedAttribute: nil,
			expectsError:      true,
			expectedErrMsg:    "PCI Bus ID cannot be empty",
		},
		"invalid PCI Bus ID format": {
			address:           "invalid-pci-id",
			expectedAttribute: nil,
			expectsError:      true,
			expectedErrMsg:    "invalid PCI Bus ID format: invalid-pci-id",
		},
		"invalid no exist PCI Bus ID": {
			address:           pciBusID,
			expectedAttribute: nil,
			expectsError:      true,
			expectedErrMsg:    "no such file or directory",
		},
		"invalid symlink": {
			mockSysfsSetup: func(t *testing.T, mockSysfs sysfsPath) {
				devicePath := mockSysfs.Devices(filepath.Join("invalid-pci-root", "0000:00:13.1", pciBusID))
				TouchFile(t, devicePath)
				busPath := mockSysfs.Bus(filepath.Join("pci", "devices", pciBusID))
				CreateSymlink(t, devicePath, busPath)
			},
			address:           pciBusID,
			expectedAttribute: nil,
			expectsError:      true,
			expectedErrMsg:    "invalid symlink target for PCI Bus ID",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			mockSysfsPath := t.TempDir()
			mockSysfs := sysfsPath(mockSysfsPath)
			if test.mockSysfsSetup != nil {
				test.mockSysfsSetup(t, mockSysfs)
			}
			sysfs = mockSysfs
			t.Cleanup(func() {
				sysfs = sysfsPath(sysfsRoot)
				if err := os.RemoveAll(mockSysfsPath); err != nil {
					t.Errorf("Failed to clean up mock sysfs path %s: %v", mockSysfsPath, err)
				}
			})

			got, err := GetPCIeRootAttributeByPCIBusID(test.address)
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
			if !reflect.DeepEqual(got, *test.expectedAttribute) {
				t.Errorf("Expected attribute %v, got %v", test.expectedAttribute, got)
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
