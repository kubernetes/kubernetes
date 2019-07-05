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

package vpx

import (
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

var RootFolder = mo.Folder{
	ManagedEntity: mo.ManagedEntity{
		ExtensibleManagedObject: mo.ExtensibleManagedObject{
			Self:           types.ManagedObjectReference{Type: "Folder", Value: "group-d1"},
			Value:          nil,
			AvailableField: nil,
		},
		Parent:        (*types.ManagedObjectReference)(nil),
		CustomValue:   nil,
		OverallStatus: "green",
		ConfigStatus:  "green",
		ConfigIssue:   nil,
		EffectiveRole: []int32{-1},
		Permission: []types.Permission{
			{
				DynamicData: types.DynamicData{},
				Entity:      &types.ManagedObjectReference{Type: "Folder", Value: "group-d1"},
				Principal:   "VSPHERE.LOCAL\\Administrator",
				Group:       false,
				RoleId:      -1,
				Propagate:   true,
			},
			{
				DynamicData: types.DynamicData{},
				Entity:      &types.ManagedObjectReference{Type: "Folder", Value: "group-d1"},
				Principal:   "VSPHERE.LOCAL\\Administrators",
				Group:       true,
				RoleId:      -1,
				Propagate:   true,
			},
		},
		Name:                "Datacenters",
		DisabledMethod:      nil,
		RecentTask:          nil,
		DeclaredAlarmState:  nil,
		AlarmActionsEnabled: (*bool)(nil),
		Tag:                 nil,
	},
	ChildType:   []string{"Folder", "Datacenter"},
	ChildEntity: nil,
}
