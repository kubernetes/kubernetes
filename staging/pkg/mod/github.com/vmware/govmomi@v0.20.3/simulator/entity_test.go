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
	"testing"

	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

func TestRename(t *testing.T) {
	m := VPX()
	m.Datacenter = 2
	m.Folder = 2

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := m.Service.NewServer()
	defer s.Close()

	dc := Map.Any("Datacenter").(*Datacenter)
	vmFolder := Map.Get(dc.VmFolder).(*Folder)

	f1 := Map.Get(vmFolder.ChildEntity[0]).(*Folder) // "F1"

	id := vmFolder.CreateFolder(&types.CreateFolder{
		This: vmFolder.Reference(),
		Name: "F2",
	}).(*methods.CreateFolderBody).Res.Returnval

	f2 := Map.Get(id).(*Folder) // "F2"

	states := []types.TaskInfoState{types.TaskInfoStateError, types.TaskInfoStateSuccess}
	name := f1.Name

	for _, expect := range states {
		id = f2.RenameTask(&types.Rename_Task{
			This:    f2.Reference(),
			NewName: name,
		}).(*methods.Rename_TaskBody).Res.Returnval

		task := Map.Get(id).(*Task)

		if task.Info.State != expect {
			t.Errorf("state=%s", task.Info.State)
		}

		name = name + "-uniq"
	}
}
