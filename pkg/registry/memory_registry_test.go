/*
Copyright 2014 Google Inc. All rights reserved.

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
package registry

import (
	"testing"

	. "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestListTasksEmpty(t *testing.T) {
	registry := MakeMemoryRegistry()
	tasks, err := registry.ListTasks(nil)
	expectNoError(t, err)
	if len(tasks) != 0 {
		t.Errorf("Unexpected task list: %#v", tasks)
	}
}

func TestMemoryListTasks(t *testing.T) {
	registry := MakeMemoryRegistry()
	registry.CreateTask("machine", Task{JSONBase: JSONBase{ID: "foo"}})
	tasks, err := registry.ListTasks(nil)
	expectNoError(t, err)
	if len(tasks) != 1 || tasks[0].ID != "foo" {
		t.Errorf("Unexpected task list: %#v", tasks)
	}
}

func TestMemorySetGetTasks(t *testing.T) {
	registry := MakeMemoryRegistry()
	expectedTask := Task{JSONBase: JSONBase{ID: "foo"}}
	registry.CreateTask("machine", expectedTask)
	task, err := registry.GetTask("foo")
	expectNoError(t, err)
	if expectedTask.ID != task.ID {
		t.Errorf("Unexpected task, expected %#v, actual %#v", expectedTask, task)
	}
}

func TestMemorySetUpdateGetTasks(t *testing.T) {
	registry := MakeMemoryRegistry()
	oldTask := Task{JSONBase: JSONBase{ID: "foo"}}
	expectedTask := Task{
		JSONBase: JSONBase{
			ID: "foo",
		},
		DesiredState: TaskState{
			Host: "foo.com",
		},
	}
	registry.CreateTask("machine", oldTask)
	registry.UpdateTask(expectedTask)
	task, err := registry.GetTask("foo")
	expectNoError(t, err)
	if expectedTask.ID != task.ID || task.DesiredState.Host != expectedTask.DesiredState.Host {
		t.Errorf("Unexpected task, expected %#v, actual %#v", expectedTask, task)
	}
}

func TestMemorySetDeleteGetTasks(t *testing.T) {
	registry := MakeMemoryRegistry()
	expectedTask := Task{JSONBase: JSONBase{ID: "foo"}}
	registry.CreateTask("machine", expectedTask)
	registry.DeleteTask("foo")
	task, err := registry.GetTask("foo")
	expectNoError(t, err)
	if task != nil {
		t.Errorf("Unexpected task: %#v", task)
	}
}

func TestListControllersEmpty(t *testing.T) {
	registry := MakeMemoryRegistry()
	tasks, err := registry.ListControllers()
	expectNoError(t, err)
	if len(tasks) != 0 {
		t.Errorf("Unexpected task list: %#v", tasks)
	}
}

func TestMemoryListControllers(t *testing.T) {
	registry := MakeMemoryRegistry()
	registry.CreateController(ReplicationController{JSONBase: JSONBase{ID: "foo"}})
	tasks, err := registry.ListControllers()
	expectNoError(t, err)
	if len(tasks) != 1 || tasks[0].ID != "foo" {
		t.Errorf("Unexpected task list: %#v", tasks)
	}
}

func TestMemorySetGetControllers(t *testing.T) {
	registry := MakeMemoryRegistry()
	expectedController := ReplicationController{JSONBase: JSONBase{ID: "foo"}}
	registry.CreateController(expectedController)
	task, err := registry.GetController("foo")
	expectNoError(t, err)
	if expectedController.ID != task.ID {
		t.Errorf("Unexpected task, expected %#v, actual %#v", expectedController, task)
	}
}

func TestMemorySetUpdateGetControllers(t *testing.T) {
	registry := MakeMemoryRegistry()
	oldController := ReplicationController{JSONBase: JSONBase{ID: "foo"}}
	expectedController := ReplicationController{
		JSONBase: JSONBase{
			ID: "foo",
		},
		DesiredState: ReplicationControllerState{
			Replicas: 2,
		},
	}
	registry.CreateController(oldController)
	registry.UpdateController(expectedController)
	task, err := registry.GetController("foo")
	expectNoError(t, err)
	if expectedController.ID != task.ID || task.DesiredState.Replicas != expectedController.DesiredState.Replicas {
		t.Errorf("Unexpected task, expected %#v, actual %#v", expectedController, task)
	}
}

func TestMemorySetDeleteGetControllers(t *testing.T) {
	registry := MakeMemoryRegistry()
	expectedController := ReplicationController{JSONBase: JSONBase{ID: "foo"}}
	registry.CreateController(expectedController)
	registry.DeleteController("foo")
	task, err := registry.GetController("foo")
	expectNoError(t, err)
	if task != nil {
		t.Errorf("Unexpected task: %#v", task)
	}
}
