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
	"fmt"
	"os"
	"path"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type VirtualMachineSnapshot struct {
	mo.VirtualMachineSnapshot
}

func (v *VirtualMachineSnapshot) createSnapshotFiles() types.BaseMethodFault {
	vm := Map.Get(v.Vm).(*VirtualMachine)

	snapshotDirectory := vm.Config.Files.SnapshotDirectory
	if snapshotDirectory == "" {
		snapshotDirectory = vm.Config.Files.VmPathName
	}

	index := 1
	for {
		fileName := fmt.Sprintf("%s-Snapshot%d.vmsn", vm.Name, index)
		f, err := vm.createFile(snapshotDirectory, fileName, false)
		if err != nil {
			switch err.(type) {
			case *types.FileAlreadyExists:
				index++
				continue
			default:
				return err
			}
		}

		_ = f.Close()

		p, _ := parseDatastorePath(snapshotDirectory)
		vm.useDatastore(p.Datastore)
		datastorePath := object.DatastorePath{
			Datastore: p.Datastore,
			Path:      path.Join(p.Path, fileName),
		}

		dataLayoutKey := vm.addFileLayoutEx(datastorePath, 0)
		vm.addSnapshotLayout(v.Self, dataLayoutKey)
		vm.addSnapshotLayoutEx(v.Self, dataLayoutKey, -1)

		return nil
	}
}

func (v *VirtualMachineSnapshot) removeSnapshotFiles(ctx *Context) types.BaseMethodFault {
	// TODO: also remove delta disks that were created when snapshot was taken

	vm := Map.Get(v.Vm).(*VirtualMachine)

	for idx, sLayout := range vm.Layout.Snapshot {
		if sLayout.Key == v.Self {
			vm.Layout.Snapshot = append(vm.Layout.Snapshot[:idx], vm.Layout.Snapshot[idx+1:]...)
			break
		}
	}

	for idx, sLayoutEx := range vm.LayoutEx.Snapshot {
		if sLayoutEx.Key == v.Self {
			for _, file := range vm.LayoutEx.File {
				if file.Key == sLayoutEx.DataKey || file.Key == sLayoutEx.MemoryKey {
					p, fault := parseDatastorePath(file.Name)
					if fault != nil {
						return fault
					}

					host := Map.Get(*vm.Runtime.Host).(*HostSystem)
					datastore := Map.FindByName(p.Datastore, host.Datastore).(*Datastore)
					dFilePath := path.Join(datastore.Info.GetDatastoreInfo().Url, p.Path)

					_ = os.Remove(dFilePath)
				}
			}

			vm.LayoutEx.Snapshot = append(vm.LayoutEx.Snapshot[:idx], vm.LayoutEx.Snapshot[idx+1:]...)
		}
	}

	vm.RefreshStorageInfo(ctx, nil)

	return nil
}

func (v *VirtualMachineSnapshot) RemoveSnapshotTask(ctx *Context, req *types.RemoveSnapshot_Task) soap.HasFault {
	task := CreateTask(v, "removeSnapshot", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		var changes []types.PropertyChange

		vm := Map.Get(v.Vm).(*VirtualMachine)
		Map.WithLock(vm, func() {
			if vm.Snapshot.CurrentSnapshot != nil && *vm.Snapshot.CurrentSnapshot == req.This {
				parent := findParentSnapshotInTree(vm.Snapshot.RootSnapshotList, req.This)
				changes = append(changes, types.PropertyChange{Name: "snapshot.currentSnapshot", Val: parent})
			}

			rootSnapshots := removeSnapshotInTree(vm.Snapshot.RootSnapshotList, req.This, req.RemoveChildren)
			changes = append(changes, types.PropertyChange{Name: "snapshot.rootSnapshotList", Val: rootSnapshots})

			if len(rootSnapshots) == 0 {
				changes = []types.PropertyChange{
					{Name: "snapshot", Val: nil},
				}
			}

			Map.Get(req.This).(*VirtualMachineSnapshot).removeSnapshotFiles(ctx)

			Map.Update(vm, changes)
		})

		Map.Remove(req.This)

		return nil, nil
	})

	return &methods.RemoveSnapshot_TaskBody{
		Res: &types.RemoveSnapshot_TaskResponse{
			Returnval: task.Run(),
		},
	}
}

func (v *VirtualMachineSnapshot) RevertToSnapshotTask(req *types.RevertToSnapshot_Task) soap.HasFault {
	task := CreateTask(v, "revertToSnapshot", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		vm := Map.Get(v.Vm).(*VirtualMachine)

		Map.WithLock(vm, func() {
			Map.Update(vm, []types.PropertyChange{
				{Name: "snapshot.currentSnapshot", Val: v.Self},
			})
		})

		return nil, nil
	})

	return &methods.RevertToSnapshot_TaskBody{
		Res: &types.RevertToSnapshot_TaskResponse{
			Returnval: task.Run(),
		},
	}
}
