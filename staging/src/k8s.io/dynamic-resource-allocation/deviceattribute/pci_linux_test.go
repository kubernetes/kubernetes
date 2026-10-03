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
	"sort"
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

func TestGetPCIeRootAttributeMapByCPUId(t *testing.T) {
	// https://wiki.osdev.org/PCI#Class_Codes
	PCIClassHostBridge := "0x060000"
	PCIClassPCItoPCIBridge := "0x060400"
	PCIClassInfiniBandtoPCIBridge := "0x060a00"
	PCIClassNotBridge := "0x050000" // Memory Controller

	createCPUDevice := func(t *testing.T, testRootPath string, cpuIDs ...int) {
		t.Helper()
		for _, cpuID := range cpuIDs {
			mkDirAll(t, filepath.Join(testRootPath, "devices", "system", "cpu", fmt.Sprintf("cpu%d", cpuID)))
		}
	}

	createPCIDevice := func(t *testing.T, testRootPath string, rootDirName, pciBusID string, pciClass, localCPUList *string) {
		t.Helper()

		relDevicePath := filepath.Join("devices", rootDirName, pciBusID)
		mkDirAll(t, filepath.Join(testRootPath, relDevicePath))
		if pciClass != nil {
			writeFile(t, filepath.Join(testRootPath, relDevicePath, "class"), *pciClass)
		}
		if localCPUList != nil {
			writeFile(t, filepath.Join(testRootPath, relDevicePath, "local_cpulist"), *localCPUList)
		}
		busDir := filepath.Join(testRootPath, "bus", "pci", "devices")
		mkDirAll(t, busDir)
		createSymlink(t,
			filepath.Join("..", "..", "..", relDevicePath),
			filepath.Join(busDir, pciBusID),
		)
	}

	tests := map[string]struct {
		testMachineSetup func(t *testing.T, testRootPath string)
		expectedMap      map[int]DeviceAttribute
		expectsError     bool
		expectedErrMsg   string
	}{
		"valid: with no PCI devices": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				createCPUDevice(t, testRootPath, 0)
			},
			expectedMap: map[int]DeviceAttribute{},
		},
		"valid: with no bridges": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				createCPUDevice(t, testRootPath, 0)
				createPCIDevice(t, testRootPath, "pci0000:00", "0000:00:00.0", new(PCIClassNotBridge), new("0"))
			},
			expectedMap: map[int]DeviceAttribute{},
		},
		"valid: ignores malformed PCIBusID format": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				createCPUDevice(t, testRootPath, 0)
				createPCIDevice(t, testRootPath, "pci0000:00", "invalid-pci-bus-id", new(PCIClassHostBridge), new("0"))
			},
			expectedMap: map[int]DeviceAttribute{},
		},
		"valid: with sole host bridge": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				createCPUDevice(t, testRootPath, 0, 1)
				createPCIDevice(t, testRootPath, "pci0000:00", "0000:00:00.0", new(PCIClassHostBridge), new("0-1"))
				createPCIDevice(t, testRootPath, "pci0000:00", "0000:00:01.0", new(PCIClassNotBridge), new("0-1"))
			},
			expectedMap: map[int]DeviceAttribute{
				0: {
					Name:  StandardDeviceAttributePCIeRoot,
					Value: resourceapi.DeviceAttribute{StringValues: sort.StringSlice([]string{"pci0000:00"})},
				},
				1: {
					Name:  StandardDeviceAttributePCIeRoot,
					Value: resourceapi.DeviceAttribute{StringValues: sort.StringSlice([]string{"pci0000:00"})},
				},
			},
		},
		"valid: with host and other bridges": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				createCPUDevice(t, testRootPath, 0, 1, 2, 3)
				createPCIDevice(t, testRootPath, "pci0000:00", "0000:00:00.0", new(PCIClassHostBridge), new("0-1"))
				createPCIDevice(t, testRootPath, "pci0001:00", "0001:00:00.0", new(PCIClassPCItoPCIBridge), new("0-1"))
				createPCIDevice(t, testRootPath, "pci0000:00", "0001:00:01.0", new(PCIClassNotBridge), new("0-1"))
				createPCIDevice(t, testRootPath, "pci0002:00", "0002:00:00.0", new(PCIClassInfiniBandtoPCIBridge), new("2-3"))
				createPCIDevice(t, testRootPath, "pci0000:00", "0002:00:01.0", new(PCIClassNotBridge), new("2-3"))
			},
			expectedMap: map[int]DeviceAttribute{
				0: {
					Name:  StandardDeviceAttributePCIeRoot,
					Value: resourceapi.DeviceAttribute{StringValues: sort.StringSlice([]string{"pci0000:00", "pci0001:00"})},
				},
				1: {
					Name:  StandardDeviceAttributePCIeRoot,
					Value: resourceapi.DeviceAttribute{StringValues: sort.StringSlice([]string{"pci0000:00", "pci0001:00"})},
				},
				2: {
					Name:  StandardDeviceAttributePCIeRoot,
					Value: resourceapi.DeviceAttribute{StringValues: sort.StringSlice([]string{"pci0002:00"})},
				},
				3: {
					Name:  StandardDeviceAttributePCIeRoot,
					Value: resourceapi.DeviceAttribute{StringValues: sort.StringSlice([]string{"pci0002:00"})},
				},
			},
		},
		"invalid: invalid PCI class(no class file)": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				createCPUDevice(t, testRootPath, 0)
				createPCIDevice(t, testRootPath, "pci0000:00", "0000:00:00.0", nil, new("0"))
			},
			expectsError:   true,
			expectedErrMsg: "failed to get PCIe bridges: failed to read PCI class for PCI Bus ID",
		},
		"invalid: invalid PCI class (empty)": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				createCPUDevice(t, testRootPath, 0)
				createPCIDevice(t, testRootPath, "pci0000:00", "0000:00:00.0", new(""), new("0"))
			},
			expectsError:   true,
			expectedErrMsg: "failed to get PCIe bridges: invalid PCI Class data for PCI Bus ID",
		},
		"invalid: invalid PCI class": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				createCPUDevice(t, testRootPath, 0)
				createPCIDevice(t, testRootPath, "pci0000:00", "0000:00:00.0", new("invalid-pci-class"), new("0"))
			},
			expectsError:   true,
			expectedErrMsg: "failed to get PCIe bridges: invalid PCI Class data for PCI Bus ID",
		},
		"invalid: CPU with invalid cpuId": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				// Setup a CPU path that matches glob but cannot be parsed as int (overflow)
				mkDirAll(t, filepath.Join(testRootPath, "devices", "system", "cpu", "cpu999999999999999999999999999999999"))
			},
			expectsError:   true,
			expectedErrMsg: "failed to get CPU IDs: failed to parse CPU ID from path",
		},
		"invalid: no local_cpulist": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				createCPUDevice(t, testRootPath, 0)
				createPCIDevice(t, testRootPath, "pci0000:00", "0000:01:00.0", new(PCIClassHostBridge), nil)
			},
			expectsError:   true,
			expectedErrMsg: "failed to read local_cpulist",
		},
		"invalid: malformed local_cpulist (empty)": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				createCPUDevice(t, testRootPath, 0)
				createPCIDevice(t, testRootPath, "pci0000:00", "0000:01:00.0", new(PCIClassHostBridge), new(""))
			},
			expectsError:   true,
			expectedErrMsg: "CPU list string is empty",
		},
		"invalid: malformed local_cpulist (non integer id)": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				createCPUDevice(t, testRootPath, 0)
				createPCIDevice(t, testRootPath, "pci0000:00", "0000:01:00.0", new(PCIClassHostBridge), new("a"))
			},
			expectsError:   true,
			expectedErrMsg: "invalid CPU ID in list",
		},
		"invalid: malformed local_cpulist (non integer range)": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				createCPUDevice(t, testRootPath, 0)
				createPCIDevice(t, testRootPath, "pci0000:00", "0000:01:00.0", new(PCIClassHostBridge), new("a-b"))
			},
			expectsError:   true,
			expectedErrMsg: "invalid CPU ID in list",
		},
		"invalid: malformed local_cpulist (invalid range)": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				createCPUDevice(t, testRootPath, 0)
				createCPUDevice(t, testRootPath, 1)
				createPCIDevice(t, testRootPath, "pci0000:00", "0000:00:00.0", new(PCIClassHostBridge), new("0-1-2"))
			},
			expectsError:   true,
			expectedErrMsg: "invalid CPU list format",
		},
		"invalid: malformed local_cpulist(reverse range)": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				createCPUDevice(t, testRootPath, 0)
				createPCIDevice(t, testRootPath, "pci0000:00", "0000:01:00.0", new(PCIClassHostBridge), new("3-2"))
			},
			expectsError:   true,
			expectedErrMsg: "invalid CPU range",
		},
		"invalid: malformed local_cpulist (empty part)": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				createCPUDevice(t, testRootPath, 0)
				createPCIDevice(t, testRootPath, "pci0000:00", "0000:00:00.0", new(PCIClassHostBridge), new("0,"))
			},
			expectsError:   true,
			expectedErrMsg: "invalid CPU list format: empty part in",
		},
		"invalid: local_cpulist with non-existent CPU IDs": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				createCPUDevice(t, testRootPath, 0)
				createPCIDevice(t, testRootPath, "pci0000:00", "0000:01:00.0", new(PCIClassHostBridge), new("0-1")) // CPU 1 does not exist
			},
			expectsError:   true,
			expectedErrMsg: "is not in the CPU ID set",
		},
		"invalid: local_cpulist with non-existent CPU IDs (no CPU)": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				createPCIDevice(t, testRootPath, "pci0000:00", "0000:01:00.0", new(PCIClassHostBridge), new("0")) // CPU 0 does not exist
			},
			expectsError:   true,
			expectedErrMsg: "is not in the CPU ID set",
		},
		"invalid: PCIeRoot resolution error": {
			testMachineSetup: func(t *testing.T, testRootPath string) {
				// resolvePCIeRoot requires targets to start with "devices/pci".
				createCPUDevice(t, testRootPath, 0)
				createPCIDevice(t, testRootPath, "not-a-pci-root", "0000:01:00.0", new(PCIClassHostBridge), new("0"))
			},
			expectsError:   true,
			expectedErrMsg: "failed to resolve PCIe Root Complex for PCI Bus ID",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			testMachinePath := t.TempDir()
			mkDirAll(t, filepath.Join(testMachinePath, "bus", "pci", "devices"))
			mkDirAll(t, filepath.Join(testMachinePath, "devices", "system", "cpu"))
			if test.testMachineSetup != nil {
				test.testMachineSetup(t, testMachinePath)
			}
			got, err := GetPCIeRootAttributeMapFromCPUId(WithFSFromRoot(testMachinePath))
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
			if !reflect.DeepEqual(got, test.expectedMap) {
				t.Errorf("Expected map %v, got %v", test.expectedMap, got)
			}
			// Additionally verify that the PCIe Root Complex values are sorted in the expected order
			for cpuID, attr := range got {
				if !sort.StringsAreSorted(attr.Value.StringValues) {
					t.Errorf("PCIe Root Complex values for CPU %d are not sorted: %v", cpuID, attr.Value.StringValues)
				}
			}
		})
	}
}

func TestParseCPUList(t *testing.T) {
	tests := map[string]struct {
		cpuListStr   string
		expectedList cpuList
		expectsError bool
	}{
		"valid single CPU": {
			cpuListStr:   "0",
			expectedList: cpuList{{start: 0, end: 0}},
			expectsError: false,
		},
		"valid multiple CPUs": {
			cpuListStr:   "0-3,5,7-9",
			expectedList: cpuList{{start: 0, end: 3}, {start: 5, end: 5}, {start: 7, end: 9}},
			expectsError: false,
		},
		"invalid format": {
			cpuListStr:   "invalid-cpu-list",
			expectedList: cpuList{},
			expectsError: true,
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got, err := parseCPUList(test.cpuListStr)
			if test.expectsError {
				if err == nil {
					t.Errorf("Expected error but got none")
					return
				}
				return
			}
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			if !reflect.DeepEqual(got, test.expectedList) {
				t.Errorf("Expected CPU list %v, got %v", test.expectedList, got)
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

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	mkDirAll(t, filepath.Dir(path))
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write file %s: %v", path, err)
	}
}

func createSymlink(t *testing.T, target, link string) {
	t.Helper()
	if err := os.Symlink(target, link); err != nil {
		t.Fatalf("Failed to create symlink from %s to %s: %v", target, link, err)
	}
}
