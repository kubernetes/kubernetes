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
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type Datacenter struct {
	mo.Datacenter

	isESX bool
}

// NewDatacenter creates a Datacenter and its child folders.
func NewDatacenter(f *Folder) *Datacenter {
	dc := &Datacenter{
		isESX: f.Self == esx.RootFolder.Self,
	}

	if dc.isESX {
		dc.Datacenter = esx.Datacenter
	}

	f.putChild(dc)

	dc.createFolders()

	return dc
}

// Create Datacenter Folders.
// Every Datacenter has 4 inventory Folders: Vm, Host, Datastore and Network.
// The ESX folder child types are limited to 1 type.
// The VC folders have additional child types, including nested folders.
func (dc *Datacenter) createFolders() {
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

		if dc.isESX {
			folder.ChildType = f.types[:1]
			folder.Self = *f.ref
			Map.PutEntity(dc, folder)
		} else {
			folder.ChildType = f.types
			e := Map.PutEntity(dc, folder)

			// propagate the generated morefs to Datacenter
			ref := e.Reference()
			f.ref.Type = ref.Type
			f.ref.Value = ref.Value
		}
	}

	net := Map.Get(dc.NetworkFolder).(*Folder)

	for _, ref := range esx.Datacenter.Network {
		// Add VM Network by default to each Datacenter
		network := &mo.Network{}
		network.Self = ref
		network.Name = strings.Split(ref.Value, "-")[1]
		network.Entity().Name = network.Name
		if !dc.isESX {
			network.Self.Value = "" // we want a different moid per-DC
		}

		net.putChild(network)
	}
}

func datacenterEventArgument(obj mo.Entity) *types.DatacenterEventArgument {
	dc, ok := obj.(*Datacenter)
	if !ok {
		dc = Map.getEntityDatacenter(obj)
	}
	return &types.DatacenterEventArgument{
		Datacenter:          dc.Self,
		EntityEventArgument: types.EntityEventArgument{Name: dc.Name},
	}
}

func (dc *Datacenter) PowerOnMultiVMTask(ctx *Context, req *types.PowerOnMultiVM_Task) soap.HasFault {
	task := CreateTask(dc, "powerOnMultiVM", func(_ *Task) (types.AnyType, types.BaseMethodFault) {
		if dc.isESX {
			return nil, new(types.NotImplemented)
		}

		for _, ref := range req.Vm {
			vm := Map.Get(ref).(*VirtualMachine)
			Map.WithLock(vm, func() {
				vm.PowerOnVMTask(ctx, &types.PowerOnVM_Task{})
			})
		}

		return nil, nil
	})

	return &methods.PowerOnMultiVM_TaskBody{
		Res: &types.PowerOnMultiVM_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (d *Datacenter) DestroyTask(req *types.Destroy_Task) soap.HasFault {
	task := CreateTask(d, "destroy", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		folders := []types.ManagedObjectReference{
			d.VmFolder,
			d.HostFolder,
		}

		for _, ref := range folders {
			if len(Map.Get(ref).(*Folder).ChildEntity) != 0 {
				return nil, &types.ResourceInUse{}
			}
		}

		Map.Get(*d.Parent).(*Folder).removeChild(d.Self)

		return nil, nil
	})

	return &methods.Destroy_TaskBody{
		Res: &types.Destroy_TaskResponse{
			Returnval: task.Run(),
		},
	}
}
