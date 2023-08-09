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
	"reflect"
	"strings"
	"time"

	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

const vTaskSuffix = "_Task" // vmomi suffix
const sTaskSuffix = "Task"  // simulator suffix (avoiding golint warning)

type Task struct {
	mo.Task

	Execute func(*Task) (types.AnyType, types.BaseMethodFault)
}

func NewTask(runner TaskRunner) *Task {
	ref := runner.Reference()
	name := reflect.TypeOf(runner).Elem().Name()
	name = strings.Replace(name, "VM", "Vm", 1) // "VM" for the type to make go-lint happy, but "Vm" for the vmodl ID
	return CreateTask(ref, name, runner.Run)
}

func CreateTask(e mo.Reference, name string, run func(*Task) (types.AnyType, types.BaseMethodFault)) *Task {
	ref := e.Reference()
	id := name

	if strings.HasSuffix(id, sTaskSuffix) {
		id = id[:len(id)-len(sTaskSuffix)]
		name = id + vTaskSuffix
	}

	task := &Task{
		Execute: run,
	}

	task.Self = Map.newReference(task)
	task.Info.Key = task.Self.Value
	task.Info.Task = task.Self
	task.Info.Name = ucFirst(name)
	task.Info.DescriptionId = fmt.Sprintf("%s.%s", ref.Type, id)
	task.Info.Entity = &ref
	task.Info.EntityName = ref.Value
	task.Info.Reason = &types.TaskReasonUser{UserName: "vcsim"} // TODO: Context.Session.User
	task.Info.QueueTime = time.Now()
	task.Info.State = types.TaskInfoStateQueued

	Map.Put(task)

	return task
}

type TaskRunner interface {
	mo.Reference

	Run(*Task) (types.AnyType, types.BaseMethodFault)
}

func (t *Task) Run() types.ManagedObjectReference {
	now := time.Now()

	Map.Update(t, []types.PropertyChange{
		{Name: "info.startTime", Val: now},
		{Name: "info.state", Val: types.TaskInfoStateRunning},
	})

	res, err := t.Execute(t)
	state := types.TaskInfoStateSuccess
	var fault interface{}
	if err != nil {
		state = types.TaskInfoStateError
		fault = types.LocalizedMethodFault{
			Fault:            err,
			LocalizedMessage: fmt.Sprintf("%T", err),
		}
	}

	now = time.Now()

	Map.Update(t, []types.PropertyChange{
		{Name: "info.completeTime", Val: now},
		{Name: "info.state", Val: state},
		{Name: "info.result", Val: res},
		{Name: "info.error", Val: fault},
	})

	return t.Self
}
