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

package simulator

import (
	"strings"

	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

// Create Datacenter Folders.
// Every Datacenter has 4 inventory Folders: Vm, Host, Datastore and Network.
// The ESX folder child types are limited to 1 type.
// The VC folders have additional child types, including nested folders.
func createDatacenterFolders(dc *mo.Datacenter, isVC bool) {
	folders := []struct {
		ref   *types.ManagedObjectReference
		name  string
		types []string
	}{
		{&dc.VmFolder, "vm", []string{"VirtualMachine", "VirtualApp", "Folder"}},
		{&dc.HostFolder, "host", []string{"ComputeResource", "Folder"}},
		{&dc.DatastoreFolder, "datastore", []string{"Datastore", "StoragePod", "Folder"}},
		{&dc.NetworkFolder, "network", []string{"Network", "DistributedVirtualSwitch", "Folder"}},
	}

	for _, f := range folders {
		folder := &Folder{}
		folder.Name = f.name

		if isVC {
			folder.ChildType = f.types
			e := Map.PutEntity(dc, folder)

			// propagate the generated morefs to Datacenter
			ref := e.Reference()
			f.ref.Type = ref.Type
			f.ref.Value = ref.Value
		} else {
			folder.ChildType = f.types[:1]
			folder.Self = *f.ref
			Map.PutEntity(dc, folder)
		}
	}

	net := Map.Get(dc.NetworkFolder).(*Folder)

	for _, ref := range esx.Datacenter.Network {
		// Add VM Network by default to each Datacenter
		network := &mo.Network{}
		network.Self = ref
		network.Name = strings.Split(ref.Value, "-")[1]
		network.Entity().Name = network.Name
		if isVC {
			network.Self.Value = "" // we want a different moid per-DC
		}

		net.putChild(network)
	}
}
