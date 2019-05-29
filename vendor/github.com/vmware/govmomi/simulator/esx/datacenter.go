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
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

// Datacenter is the default template for Datacenter properties.
// Capture method:
//   govc datacenter.info -dump
var Datacenter = mo.Datacenter{
	ManagedEntity: mo.ManagedEntity{
		ExtensibleManagedObject: mo.ExtensibleManagedObject{
			Self:           types.ManagedObjectReference{Type: "Datacenter", Value: "ha-datacenter"},
			Value:          nil,
			AvailableField: nil,
		},
		Parent:              (*types.ManagedObjectReference)(nil),
		CustomValue:         nil,
		OverallStatus:       "",
		ConfigStatus:        "",
		ConfigIssue:         nil,
		EffectiveRole:       nil,
		Permission:          nil,
		Name:                "ha-datacenter",
		DisabledMethod:      nil,
		RecentTask:          nil,
		DeclaredAlarmState:  nil,
		TriggeredAlarmState: nil,
		AlarmActionsEnabled: (*bool)(nil),
		Tag:                 nil,
	},
	VmFolder:        types.ManagedObjectReference{Type: "Folder", Value: "ha-folder-vm"},
	HostFolder:      types.ManagedObjectReference{Type: "Folder", Value: "ha-folder-host"},
	DatastoreFolder: types.ManagedObjectReference{Type: "Folder", Value: "ha-folder-datastore"},
	NetworkFolder:   types.ManagedObjectReference{Type: "Folder", Value: "ha-folder-network"},
	Datastore: []types.ManagedObjectReference{
		{Type: "Datastore", Value: "57089c25-85e3ccd4-17b6-000c29d0beb3"},
	},
	Network: []types.ManagedObjectReference{
		{Type: "Network", Value: "HaNetwork-VM Network"},
	},
	Configuration: types.DatacenterConfigInfo{},
}
