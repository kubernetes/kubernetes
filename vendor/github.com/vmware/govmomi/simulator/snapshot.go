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
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type VirtualMachineSnapshot struct {
	mo.VirtualMachineSnapshot
}

func (v *VirtualMachineSnapshot) RemoveSnapshotTask(req *types.RemoveSnapshot_Task) soap.HasFault {
	task := CreateTask(v, "removeSnapshot", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		Map.Remove(req.This)

		vm := Map.Get(v.Vm).(*VirtualMachine)
		Map.WithLock(vm, func() {
			if vm.Snapshot.CurrentSnapshot != nil && *vm.Snapshot.CurrentSnapshot == req.This {
				parent := findParentSnapshotInTree(vm.Snapshot.RootSnapshotList, req.This)
				vm.Snapshot.CurrentSnapshot = parent
			}

			vm.Snapshot.RootSnapshotList = removeSnapshotInTree(vm.Snapshot.RootSnapshotList, req.This, req.RemoveChildren)

			if len(vm.Snapshot.RootSnapshotList) == 0 {
				vm.Snapshot = nil
			}
		})

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

		Map.WithLock(vm, func() { vm.Snapshot.CurrentSnapshot = &v.Self })

		return nil, nil
	})

	return &methods.RevertToSnapshot_TaskBody{
		Res: &types.RevertToSnapshot_TaskResponse{
			Returnval: task.Run(),
		},
	}
}
