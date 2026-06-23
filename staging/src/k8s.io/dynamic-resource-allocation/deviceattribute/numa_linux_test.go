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

	resourceapi "k8s.io/api/resource/v1"
)

func TestGetNUMANodeAttributeByPCIBusID(t *testing.T) {
	pciBusID := "0000:02:00.0"
	numaNodePath := filepath.Join("bus", "pci", "devices", pciBusID, "numa_node")

	tests := map[string]struct {
		setup             func(t *testing.T, root string)
		address           string
		attrForm          AttributeForm
		expectedAttribute *DeviceAttribute
		expectsError      bool
		expectedErrMsg    string
	}{
		"scalar": {
			setup: func(t *testing.T, root string) {
				writeFile(t, filepath.Join(root, numaNodePath), "4\n")
			},
			address:  pciBusID,
			attrForm: ScalarAttribute,
			expectedAttribute: &DeviceAttribute{
				Name:  StandardDeviceAttributeNUMANode,
				Value: resourceapi.DeviceAttribute{IntValue: new(int64(4))},
			},
		},
		"list with equidistant nodes": {
			setup: func(t *testing.T, root string) {
				writeFile(t, filepath.Join(root, numaNodePath), "1\n")
				// node1 SLIT row: node0=12, node1(self)=10, node2=12.
				// Minimum non-self distance is 12 -> {0,2}; physical node first.
				// No cpulist/topology files, so the socket filter is inactive and
				// all minimum-distance nodes are kept.
				writeFile(t, filepath.Join(root, "devices", "system", "node", "node1", "distance"), "12 10 12\n")
			},
			address:  pciBusID,
			attrForm: ListAttribute,
			expectedAttribute: &DeviceAttribute{
				Name:  StandardDeviceAttributeNUMANode,
				Value: resourceapi.DeviceAttribute{IntValues: []int64{1, 0, 2}},
			},
		},
		"list single NUMA node": {
			setup: func(t *testing.T, root string) {
				writeFile(t, filepath.Join(root, numaNodePath), "0\n")
				// Only a self distance, so there are no equidistant peers.
				writeFile(t, filepath.Join(root, "devices", "system", "node", "node0", "distance"), "10\n")
			},
			address:  pciBusID,
			attrForm: ListAttribute,
			expectedAttribute: &DeviceAttribute{
				Name:  StandardDeviceAttributeNUMANode,
				Value: resourceapi.DeviceAttribute{IntValues: []int64{0}},
			},
		},
		"list with same-socket filter": {
			setup: func(t *testing.T, root string) {
				writeFile(t, filepath.Join(root, numaNodePath), "0\n")
				// node0 self=10; node1 and node2 both at minimum distance 12; node3 farther.
				writeFile(t, filepath.Join(root, "devices", "system", "node", "node0", "distance"), "10 12 12 20\n")
				// node0 and node1 are on socket 0; node2 is on socket 1, so node2 is filtered out.
				writeFile(t, filepath.Join(root, "devices", "system", "node", "node0", "cpulist"), "0\n")
				writeFile(t, filepath.Join(root, "devices", "system", "node", "node1", "cpulist"), "1\n")
				writeFile(t, filepath.Join(root, "devices", "system", "node", "node2", "cpulist"), "2\n")
				writeFile(t, filepath.Join(root, "devices", "system", "cpu", "cpu0", "topology", "physical_package_id"), "0\n")
				writeFile(t, filepath.Join(root, "devices", "system", "cpu", "cpu1", "topology", "physical_package_id"), "0\n")
				writeFile(t, filepath.Join(root, "devices", "system", "cpu", "cpu2", "topology", "physical_package_id"), "1\n")
			},
			address:  pciBusID,
			attrForm: ListAttribute,
			expectedAttribute: &DeviceAttribute{
				Name:  StandardDeviceAttributeNUMANode,
				Value: resourceapi.DeviceAttribute{IntValues: []int64{0, 1}},
			},
		},
		"no NUMA affinity is rejected": {
			setup: func(t *testing.T, root string) {
				writeFile(t, filepath.Join(root, numaNodePath), "-1\n")
			},
			address:        pciBusID,
			attrForm:       ListAttribute,
			expectsError:   true,
			expectedErrMsg: "no NUMA affinity",
		},
		"empty PCI Bus ID": {
			address:        "",
			attrForm:       ListAttribute,
			expectsError:   true,
			expectedErrMsg: "PCI Bus ID cannot be empty",
		},
		"missing numa_node": {
			address:        pciBusID,
			attrForm:       ScalarAttribute,
			expectsError:   true,
			expectedErrMsg: "failed to read NUMA node",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			if test.setup != nil {
				test.setup(t, root)
			}
			got, err := GetNUMANodeAttributeByPCIBusID(test.address, test.attrForm, WithFSFromRoot(root))
			if test.expectsError {
				if err == nil {
					t.Fatalf("expected error but got none")
				}
				if !strings.Contains(err.Error(), test.expectedErrMsg) {
					t.Errorf("expected error containing %q, got %q", test.expectedErrMsg, err.Error())
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(got, *test.expectedAttribute) {
				t.Errorf("expected %+v, got %+v", *test.expectedAttribute, got)
			}
		})
	}
}

func TestGetNUMANodeAttribute(t *testing.T) {
	t.Run("scalar", func(t *testing.T) {
		got, err := GetNUMANodeAttribute(4, ScalarAttribute)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		want := DeviceAttribute{
			Name:  StandardDeviceAttributeNUMANode,
			Value: resourceapi.DeviceAttribute{IntValue: new(int64(4))},
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("expected %+v, got %+v", want, got)
		}
		if got.Value.IntValues != nil {
			t.Errorf("expected nil IntValues for scalar attribute, got %v", got.Value.IntValues)
		}
	})

	t.Run("list with equidistant nodes", func(t *testing.T) {
		root := t.TempDir()
		writeFile(t, filepath.Join(root, "devices", "system", "node", "node1", "distance"), "12 10 12\n")
		got, err := GetNUMANodeAttribute(1, ListAttribute, WithFSFromRoot(root))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		want := DeviceAttribute{
			Name:  StandardDeviceAttributeNUMANode,
			Value: resourceapi.DeviceAttribute{IntValues: []int64{1, 0, 2}},
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("expected %+v, got %+v", want, got)
		}
		if got.Value.IntValue != nil {
			t.Errorf("expected nil IntValue for list attribute, got %v", *got.Value.IntValue)
		}
	})

	t.Run("no NUMA affinity is rejected", func(t *testing.T) {
		_, err := GetNUMANodeAttribute(-1, ScalarAttribute)
		if err == nil {
			t.Fatal("expected error for a device with no NUMA affinity, got none")
		}
		if !strings.Contains(err.Error(), "invalid NUMA node") {
			t.Errorf("expected error about invalid NUMA node, got %q", err.Error())
		}
	})
}

func TestGetNUMANodeForCPU(t *testing.T) {
	tests := map[string]struct {
		setup       func(t *testing.T, root string)
		cpuID       int
		wantNode    int
		expectsErr  bool
		errContains string
	}{
		"cpu on node1": {
			setup: func(t *testing.T, root string) {
				mkDirAll(t, filepath.Join(root, "devices", "system", "cpu", "cpu5", "node1"))
			},
			cpuID:    5,
			wantNode: 1,
		},
		"cpu on node0": {
			setup: func(t *testing.T, root string) {
				mkDirAll(t, filepath.Join(root, "devices", "system", "cpu", "cpu9", "node0"))
			},
			cpuID:    9,
			wantNode: 0,
		},
		"cpu not found": {
			setup:       func(t *testing.T, root string) {},
			cpuID:       99,
			expectsErr:  true,
			errContains: "not found in any NUMA node",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			if test.setup != nil {
				test.setup(t, root)
			}
			got, err := GetNUMANodeForCPU(test.cpuID, WithFSFromRoot(root))
			if test.expectsErr {
				if err == nil {
					t.Fatalf("expected error but got node %d", got)
				}
				if !strings.Contains(err.Error(), test.errContains) {
					t.Errorf("expected error containing %q, got %q", test.errContains, err.Error())
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != test.wantNode {
				t.Errorf("expected NUMA node %d, got %d", test.wantNode, got)
			}
		})
	}
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	mkDirAll(t, filepath.Dir(path))
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write file %s: %v", path, err)
	}
}
