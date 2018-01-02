/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package object

import (
	"testing"
	"time"

	"github.com/vmware/govmomi/vim25/types"
)

// VirtualMachine should implement the Reference interface.
var _ Reference = VirtualMachine{}

// pretty.Printf generated
var snapshot = &types.VirtualMachineSnapshotInfo{
	DynamicData:     types.DynamicData{},
	CurrentSnapshot: &types.ManagedObjectReference{Type: "VirtualMachineSnapshot", Value: "2-snapshot-11"},
	RootSnapshotList: []types.VirtualMachineSnapshotTree{
		{
			DynamicData:    types.DynamicData{},
			Snapshot:       types.ManagedObjectReference{Type: "VirtualMachineSnapshot", Value: "2-snapshot-1"},
			Vm:             types.ManagedObjectReference{Type: "VirtualMachine", Value: "2"},
			Name:           "root",
			Description:    "",
			Id:             1,
			CreateTime:     time.Now(),
			State:          "poweredOn",
			Quiesced:       false,
			BackupManifest: "",
			ChildSnapshotList: []types.VirtualMachineSnapshotTree{
				{
					DynamicData:       types.DynamicData{},
					Snapshot:          types.ManagedObjectReference{Type: "VirtualMachineSnapshot", Value: "2-snapshot-2"},
					Vm:                types.ManagedObjectReference{Type: "VirtualMachine", Value: "2"},
					Name:              "child",
					Description:       "",
					Id:                2,
					CreateTime:        time.Now(),
					State:             "poweredOn",
					Quiesced:          false,
					BackupManifest:    "",
					ChildSnapshotList: nil,
					ReplaySupported:   types.NewBool(false),
				},
				{
					DynamicData:    types.DynamicData{},
					Snapshot:       types.ManagedObjectReference{Type: "VirtualMachineSnapshot", Value: "2-snapshot-3"},
					Vm:             types.ManagedObjectReference{Type: "VirtualMachine", Value: "2"},
					Name:           "child",
					Description:    "",
					Id:             3,
					CreateTime:     time.Now(),
					State:          "poweredOn",
					Quiesced:       false,
					BackupManifest: "",
					ChildSnapshotList: []types.VirtualMachineSnapshotTree{
						{
							DynamicData:    types.DynamicData{},
							Snapshot:       types.ManagedObjectReference{Type: "VirtualMachineSnapshot", Value: "2-snapshot-9"},
							Vm:             types.ManagedObjectReference{Type: "VirtualMachine", Value: "2"},
							Name:           "grandkid",
							Description:    "",
							Id:             9,
							CreateTime:     time.Now(),
							State:          "poweredOn",
							Quiesced:       false,
							BackupManifest: "",
							ChildSnapshotList: []types.VirtualMachineSnapshotTree{
								{
									DynamicData:       types.DynamicData{},
									Snapshot:          types.ManagedObjectReference{Type: "VirtualMachineSnapshot", Value: "2-snapshot-10"},
									Vm:                types.ManagedObjectReference{Type: "VirtualMachine", Value: "2"},
									Name:              "great",
									Description:       "",
									Id:                10,
									CreateTime:        time.Now(),
									State:             "poweredOn",
									Quiesced:          false,
									BackupManifest:    "",
									ChildSnapshotList: nil,
									ReplaySupported:   types.NewBool(false),
								},
							},
							ReplaySupported: types.NewBool(false),
						},
					},
					ReplaySupported: types.NewBool(false),
				},
				{
					DynamicData:    types.DynamicData{},
					Snapshot:       types.ManagedObjectReference{Type: "VirtualMachineSnapshot", Value: "2-snapshot-5"},
					Vm:             types.ManagedObjectReference{Type: "VirtualMachine", Value: "2"},
					Name:           "voodoo",
					Description:    "",
					Id:             5,
					CreateTime:     time.Now(),
					State:          "poweredOn",
					Quiesced:       false,
					BackupManifest: "",
					ChildSnapshotList: []types.VirtualMachineSnapshotTree{
						{
							DynamicData:       types.DynamicData{},
							Snapshot:          types.ManagedObjectReference{Type: "VirtualMachineSnapshot", Value: "2-snapshot-11"},
							Vm:                types.ManagedObjectReference{Type: "VirtualMachine", Value: "2"},
							Name:              "child",
							Description:       "",
							Id:                11,
							CreateTime:        time.Now(),
							State:             "poweredOn",
							Quiesced:          false,
							BackupManifest:    "",
							ChildSnapshotList: nil,
							ReplaySupported:   types.NewBool(false),
						},
					},
					ReplaySupported: types.NewBool(false),
				},
				{
					DynamicData:    types.DynamicData{},
					Snapshot:       types.ManagedObjectReference{Type: "VirtualMachineSnapshot", Value: "2-snapshot-6"},
					Vm:             types.ManagedObjectReference{Type: "VirtualMachine", Value: "2"},
					Name:           "better",
					Description:    "",
					Id:             6,
					CreateTime:     time.Now(),
					State:          "poweredOn",
					Quiesced:       false,
					BackupManifest: "",
					ChildSnapshotList: []types.VirtualMachineSnapshotTree{
						{
							DynamicData:    types.DynamicData{},
							Snapshot:       types.ManagedObjectReference{Type: "VirtualMachineSnapshot", Value: "2-snapshot-7"},
							Vm:             types.ManagedObjectReference{Type: "VirtualMachine", Value: "2"},
							Name:           "best",
							Description:    "",
							Id:             7,
							CreateTime:     time.Now(),
							State:          "poweredOn",
							Quiesced:       false,
							BackupManifest: "",
							ChildSnapshotList: []types.VirtualMachineSnapshotTree{
								{
									DynamicData:       types.DynamicData{},
									Snapshot:          types.ManagedObjectReference{Type: "VirtualMachineSnapshot", Value: "2-snapshot-8"},
									Vm:                types.ManagedObjectReference{Type: "VirtualMachine", Value: "2"},
									Name:              "betterer",
									Description:       "",
									Id:                8,
									CreateTime:        time.Now(),
									State:             "poweredOn",
									Quiesced:          false,
									BackupManifest:    "",
									ChildSnapshotList: nil,
									ReplaySupported:   types.NewBool(false),
								},
							},
							ReplaySupported: types.NewBool(false),
						},
					},
					ReplaySupported: types.NewBool(false),
				},
			},
			ReplaySupported: types.NewBool(false),
		},
	},
}

func TestVirtualMachineSnapshotMap(t *testing.T) {
	m := make(snapshotMap)
	m.add("", snapshot.RootSnapshotList)

	tests := []struct {
		name   string
		expect int
	}{
		{"enoent", 0},
		{"root", 1},
		{"child", 3},
		{"root/child", 2},
		{"root/voodoo/child", 1},
		{"2-snapshot-6", 1},
	}

	for _, test := range tests {
		s := m[test.name]

		if len(s) != test.expect {
			t.Errorf("%s: %d != %d", test.name, len(s), test.expect)
		}
	}
}
