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

	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type addWaterTask struct {
	*mo.Folder

	fault types.BaseMethodFault
}

func (a *addWaterTask) Run(task *Task) (types.AnyType, types.BaseMethodFault) {
	return nil, a.fault
}

func TestNewTask(t *testing.T) {
	f := &mo.Folder{}
	Map.NewEntity(f)

	add := &addWaterTask{f, nil}
	task := NewTask(add)
	info := &task.Info

	if info.Name != "AddWater_Task" {
		t.Errorf("name=%s", info.Name)
	}

	if info.DescriptionId != "Folder.addWater" {
		t.Errorf("descriptionId=%s", info.DescriptionId)
	}

	task.Run()

	if info.State != types.TaskInfoStateSuccess {
		t.Fail()
	}

	add.fault = &types.ManagedObjectNotFound{}

	task.Run()

	if info.State != types.TaskInfoStateError {
		t.Fail()
	}
}
