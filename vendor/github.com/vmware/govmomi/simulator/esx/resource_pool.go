/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package esx

import (
	"time"

	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

// ResourcePool is the default template for ResourcePool properties.
// Capture method:
//   govc pool.info "*" -dump
var ResourcePool = mo.ResourcePool{
	ManagedEntity: mo.ManagedEntity{
		ExtensibleManagedObject: mo.ExtensibleManagedObject{
			Self:           types.ManagedObjectReference{Type: "ResourcePool", Value: "ha-root-pool"},
			Value:          nil,
			AvailableField: nil,
		},
		Parent:              &types.ManagedObjectReference{Type: "ComputeResource", Value: "ha-compute-res"},
		CustomValue:         nil,
		OverallStatus:       "green",
		ConfigStatus:        "green",
		ConfigIssue:         nil,
		EffectiveRole:       []int32{-1},
		Permission:          nil,
		Name:                "Resources",
		DisabledMethod:      []string{"CreateVApp", "CreateChildVM_Task"},
		RecentTask:          nil,
		DeclaredAlarmState:  nil,
		TriggeredAlarmState: nil,
		AlarmActionsEnabled: (*bool)(nil),
		Tag:                 nil,
	},
	Summary: &types.ResourcePoolSummary{
		DynamicData: types.DynamicData{},
		Name:        "Resources",
		Config: types.ResourceConfigSpec{
			DynamicData:   types.DynamicData{},
			Entity:        &types.ManagedObjectReference{Type: "ResourcePool", Value: "ha-root-pool"},
			ChangeVersion: "",
			LastModified:  (*time.Time)(nil),
			CpuAllocation: types.ResourceAllocationInfo{
				DynamicData:           types.DynamicData{},
				Reservation:           types.NewInt64(4121),
				ExpandableReservation: types.NewBool(false),
				Limit: types.NewInt64(4121),
				Shares: &types.SharesInfo{
					DynamicData: types.DynamicData{},
					Shares:      9000,
					Level:       "custom",
				},
				OverheadLimit: nil,
			},
			MemoryAllocation: types.ResourceAllocationInfo{
				DynamicData:           types.DynamicData{},
				Reservation:           types.NewInt64(961),
				ExpandableReservation: types.NewBool(false),
				Limit: types.NewInt64(961),
				Shares: &types.SharesInfo{
					DynamicData: types.DynamicData{},
					Shares:      9000,
					Level:       "custom",
				},
				OverheadLimit: nil,
			},
		},
		Runtime: types.ResourcePoolRuntimeInfo{
			DynamicData: types.DynamicData{},
			Memory: types.ResourcePoolResourceUsage{
				DynamicData:          types.DynamicData{},
				ReservationUsed:      0,
				ReservationUsedForVm: 0,
				UnreservedForPool:    1007681536,
				UnreservedForVm:      1007681536,
				OverallUsage:         0,
				MaxUsage:             1007681536,
			},
			Cpu: types.ResourcePoolResourceUsage{
				DynamicData:          types.DynamicData{},
				ReservationUsed:      0,
				ReservationUsedForVm: 0,
				UnreservedForPool:    4121,
				UnreservedForVm:      4121,
				OverallUsage:         0,
				MaxUsage:             4121,
			},
			OverallStatus: "green",
		},
		QuickStats:         (*types.ResourcePoolQuickStats)(nil),
		ConfiguredMemoryMB: 0,
	},
	Runtime: types.ResourcePoolRuntimeInfo{
		DynamicData: types.DynamicData{},
		Memory: types.ResourcePoolResourceUsage{
			DynamicData:          types.DynamicData{},
			ReservationUsed:      0,
			ReservationUsedForVm: 0,
			UnreservedForPool:    1007681536,
			UnreservedForVm:      1007681536,
			OverallUsage:         0,
			MaxUsage:             1007681536,
		},
		Cpu: types.ResourcePoolResourceUsage{
			DynamicData:          types.DynamicData{},
			ReservationUsed:      0,
			ReservationUsedForVm: 0,
			UnreservedForPool:    4121,
			UnreservedForVm:      4121,
			OverallUsage:         0,
			MaxUsage:             4121,
		},
		OverallStatus: "green",
	},
	Owner:        types.ManagedObjectReference{Type: "ComputeResource", Value: "ha-compute-res"},
	ResourcePool: nil,
	Vm:           nil,
	Config: types.ResourceConfigSpec{
		DynamicData:   types.DynamicData{},
		Entity:        &types.ManagedObjectReference{Type: "ResourcePool", Value: "ha-root-pool"},
		ChangeVersion: "",
		LastModified:  (*time.Time)(nil),
		CpuAllocation: types.ResourceAllocationInfo{
			DynamicData:           types.DynamicData{},
			Reservation:           types.NewInt64(4121),
			ExpandableReservation: types.NewBool(false),
			Limit: types.NewInt64(4121),
			Shares: &types.SharesInfo{
				DynamicData: types.DynamicData{},
				Shares:      9000,
				Level:       "custom",
			},
			OverheadLimit: nil,
		},
		MemoryAllocation: types.ResourceAllocationInfo{
			DynamicData:           types.DynamicData{},
			Reservation:           types.NewInt64(961),
			ExpandableReservation: types.NewBool(false),
			Limit: types.NewInt64(961),
			Shares: &types.SharesInfo{
				DynamicData: types.DynamicData{},
				Shares:      9000,
				Level:       "custom",
			},
			OverheadLimit: nil,
		},
	},
	ChildConfiguration: nil,
}
