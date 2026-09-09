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
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"k8s.io/utils/ptr"

	resourceapi "k8s.io/api/resource/v1"
)

func TestGetPCIeRootAttributePCIBusID(t *testing.T) {
	pcieRoot := "pci0000:00"
	intermediateBusID := "0000:01:00.0"
	pciBusID := "0000:02:00.0"
	expectedAttribute := DeviceAttribute{
		Name:  StandardDeviceAttributePCIeRoot,
		Value: resourceapi.DeviceAttribute{StringValue: ptr.To(pcieRoot)},
	}

	tests := map[string]struct {
		testMachineSetup  func(t *testing.T, testRootPath string)
		address           string
		expectedAttribute *DeviceAttribute
		expectsError      bool
		expectedErrMsg    string
	}{
		"valid": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				relDevicePath := filepath.Join("devices", pcieRoot, intermediateBusID, pciBusID)
				touchFile(t, filepath.Join(testRootPath, relDevicePath))
				busDir := filepath.Join(testRootPath, "bus", "pci", "devices")
				mkDirAll(t, busDir)
				createSymlink(t,
					filepath.Join("..", "..", "..", relDevicePath),
					filepath.Join(busDir, pciBusID),
				)
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
		"invalid symlink (invalid prefix)": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				relDevicePath := filepath.Join("devices", "invalid-pci-root", intermediateBusID, pciBusID)
				touchFile(t, filepath.Join(testRootPath, relDevicePath))
				busDir := filepath.Join(testRootPath, "bus", "pci", "devices")
				mkDirAll(t, busDir)
				createSymlink(t,
					filepath.Join("..", "..", "..", relDevicePath),
					filepath.Join(busDir, pciBusID),
				)
			},
			address:           pciBusID,
			expectedAttribute: nil,
			expectsError:      true,
			expectedErrMsg:    fmt.Sprintf("symlink target for PCI Bus ID %s is invalid: it must start with", pciBusID),
		},
		"invalid symlink (invalid suffix)": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				relDevicePath := filepath.Join("devices", pcieRoot, intermediateBusID, "0000:00:13.1") // different PCI Bus ID
				touchFile(t, filepath.Join(testRootPath, relDevicePath))
				busDir := filepath.Join(testRootPath, "bus", "pci", "devices")
				mkDirAll(t, busDir)
				createSymlink(t,
					filepath.Join("..", "..", "..", relDevicePath),
					filepath.Join(busDir, pciBusID),
				)
			},
			address:           pciBusID,
			expectedAttribute: nil,
			expectsError:      true,
			expectedErrMsg:    fmt.Sprintf("symlink target for PCI Bus ID %s is invalid: it must end with", pciBusID),
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			// per docs, the testing package ensures cleanup, so no need to do that ourselves
			testMachinePath := t.TempDir()
			if test.testMachineSetup != nil {
				test.testMachineSetup(t, testMachinePath)
			}
			got, err := GetPCIeRootAttributeByPCIBusID(test.address, WithFSFromRoot(testMachinePath))
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

func TestGetPCIBusIDAttribute(t *testing.T) {
	pciBusID := "0000:02:00.0"
	expectedAttribute := DeviceAttribute{
		Name:  StandardDeviceAttributePCIBusID,
		Value: resourceapi.DeviceAttribute{StringValue: ptr.To(pciBusID)},
	}

	tests := map[string]struct {
		address           string
		expectedAttribute *DeviceAttribute
		expectsError      bool
		expectedErrMsg    string
	}{
		"valid": {
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
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := GetPCIBusIDAttribute(test.address)
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

func mkDirAll(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0755); err != nil {
		t.Fatalf("Failed to create directory %s: %v", path, err)
	}
}

func touchFile(t *testing.T, path string) {
	t.Helper()
	mkDirAll(t, filepath.Dir(path))
	if _, err := os.Create(path); err != nil {
		t.Fatalf("Failed to create file %s: %v", path, err)
	}
}

func createSymlink(t *testing.T, target, link string) {
	t.Helper()
	if err := os.Symlink(target, link); err != nil {
		t.Fatalf("Failed to create symlink from %s to %s: %v", target, link, err)
	}
}
