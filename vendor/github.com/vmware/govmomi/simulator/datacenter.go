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
	"log"
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
func NewDatacenter(ctx *Context, f *mo.Folder) *Datacenter {
	dc := &Datacenter{
		isESX: f.Self == esx.RootFolder.Self,
	}

	if dc.isESX {
		dc.Datacenter = esx.Datacenter
	}

	folderPutChild(ctx, f, dc)

	dc.createFolders(ctx)

	return dc
}

func (dc *Datacenter) RenameTask(ctx *Context, r *types.Rename_Task) soap.HasFault {
	return RenameTask(ctx, dc, r)
}

// Create Datacenter Folders.
// Every Datacenter has 4 inventory Folders: Vm, Host, Datastore and Network.
// The ESX folder child types are limited to 1 type.
// The VC folders have additional child types, including nested folders.
func (dc *Datacenter) createFolders(ctx *Context) {
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
			ctx.Map.PutEntity(dc, folder)
		} else {
			folder.ChildType = f.types
			e := ctx.Map.PutEntity(dc, folder)

			// propagate the generated morefs to Datacenter
			ref := e.Reference()
			f.ref.Type = ref.Type
			f.ref.Value = ref.Value
		}
	}

	net := ctx.Map.Get(dc.NetworkFolder).(*Folder)

	for _, ref := range esx.Datacenter.Network {
		// Add VM Network by default to each Datacenter
		network := &mo.Network{}
		network.Self = ref
		network.Name = strings.Split(ref.Value, "-")[1]
		network.Entity().Name = network.Name
		if !dc.isESX {
			network.Self.Value = "" // we want a different moid per-DC
		}

		folderPutChild(ctx, &net.Folder, network)
	}
}

func (dc *Datacenter) defaultNetwork() []types.ManagedObjectReference {
	return dc.Network[:1] // VM Network
}

// folder returns the Datacenter folder that can contain the given object type
func (dc *Datacenter) folder(obj mo.Entity) *mo.Folder {
	folders := []types.ManagedObjectReference{
		dc.VmFolder,
		dc.HostFolder,
		dc.DatastoreFolder,
		dc.NetworkFolder,
	}
	otype := getManagedObject(obj).Type()
	rtype := obj.Reference().Type

	for i := range folders {
		folder, _ := asFolderMO(Map.Get(folders[i]))
		for _, kind := range folder.ChildType {
			if rtype == kind {
				return folder
			}
			if f, ok := otype.FieldByName(kind); ok && f.Anonymous {
				return folder
			}
		}
	}

	log.Panicf("failed to find folder for type=%s", rtype)
	return nil
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

		// Return per-VM tasks, structured as:
		// thisTask.result - DC level task
		//    +- []Attempted
		//        +- subTask.result - VM level powerOn task result
		//        +- ...
		res := types.ClusterPowerOnVmResult{}
		res.Attempted = []types.ClusterAttemptedVmInfo{}

		for _, ref := range req.Vm {
			vm := ctx.Map.Get(ref).(*VirtualMachine)

			// This task creates multiple subtasks which violates the assumption
			// of 1:1 Context:Task, which results in data races in objects
			// like the Simulator.Event manager. This is the minimum context
			// required for the PowerOnVMTask to complete.
			taskCtx := &Context{
				Context: ctx.Context,
				Session: ctx.Session,
				Map:     ctx.Map,
			}

			// NOTE: Simulator does not actually perform any specific host-level placement
			// (equivalent to vSphere DRS).
			taskCtx.WithLock(vm, func() {
				vmTaskBody := vm.PowerOnVMTask(taskCtx, &types.PowerOnVM_Task{}).(*methods.PowerOnVM_TaskBody)
				res.Attempted = append(res.Attempted, types.ClusterAttemptedVmInfo{Vm: ref, Task: &vmTaskBody.Res.Returnval})
			})
		}

		return res, nil
	})

	return &methods.PowerOnMultiVM_TaskBody{
		Res: &types.PowerOnMultiVM_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}

func (d *Datacenter) DestroyTask(ctx *Context, req *types.Destroy_Task) soap.HasFault {
	task := CreateTask(d, "destroy", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		folders := []types.ManagedObjectReference{
			d.VmFolder,
			d.HostFolder,
		}

		for _, ref := range folders {
			f, _ := asFolderMO(ctx.Map.Get(ref))
			if len(f.ChildEntity) != 0 {
				return nil, &types.ResourceInUse{}
			}
		}

		p, _ := asFolderMO(ctx.Map.Get(*d.Parent))
		folderRemoveChild(ctx, p, d.Self)

		return nil, nil
	})

	return &methods.Destroy_TaskBody{
		Res: &types.Destroy_TaskResponse{
			Returnval: task.Run(ctx),
		},
	}
}
