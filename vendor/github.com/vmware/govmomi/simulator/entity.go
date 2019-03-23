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

func RenameTask(e mo.Entity, r *types.Rename_Task) soap.HasFault {
	task := CreateTask(e, "rename", func(t *Task) (types.AnyType, types.BaseMethodFault) {
		obj := Map.Get(r.This).(mo.Entity).Entity()

		if parent, ok := Map.Get(*obj.Parent).(*Folder); ok {
			if Map.FindByName(r.NewName, parent.ChildEntity) != nil {
				return nil, &types.InvalidArgument{InvalidProperty: "name"}
			}
		}

		obj.Name = r.NewName

		return nil, nil
	})

	return &methods.Rename_TaskBody{
		Res: &types.Rename_TaskResponse{
			Returnval: task.Run(),
		},
	}
}
