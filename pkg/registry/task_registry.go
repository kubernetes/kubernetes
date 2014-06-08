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
	"encoding/json"
	"fmt"
	"net/url"

	. "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

// TaskRegistryStorage implements the RESTStorage interface in terms of a TaskRegistry
type TaskRegistryStorage struct {
	registry      TaskRegistry
	containerInfo client.ContainerInfo
	scheduler     Scheduler
}

func MakeTaskRegistryStorage(registry TaskRegistry, containerInfo client.ContainerInfo, scheduler Scheduler) apiserver.RESTStorage {
	return &TaskRegistryStorage{
		registry:      registry,
		containerInfo: containerInfo,
		scheduler:     scheduler,
	}
}

// LabelMatch tests to see if a Task's labels map contains 'key' mapping to 'value'
func LabelMatch(task Task, queryKey, queryValue string) bool {
	for key, value := range task.Labels {
		if queryKey == key && queryValue == value {
			return true
		}
	}
	return false
}

// LabelMatch tests to see if a Task's labels map contains all key/value pairs in 'labelQuery'
func LabelsMatch(task Task, labelQuery *map[string]string) bool {
	if labelQuery == nil {
		return true
	}
	for key, value := range *labelQuery {
		if !LabelMatch(task, key, value) {
			return false
		}
	}
	return true
}

func (storage *TaskRegistryStorage) List(url *url.URL) (interface{}, error) {
	var result TaskList
	var query *map[string]string
	if url != nil {
		queryMap := client.DecodeLabelQuery(url.Query().Get("labels"))
		query = &queryMap
	}
	tasks, err := storage.registry.ListTasks(query)
	if err == nil {
		result = TaskList{
			Items: tasks,
		}
	}
	result.Kind = "cluster#taskList"
	return result, err
}

func (storage *TaskRegistryStorage) Get(id string) (interface{}, error) {
	task, err := storage.registry.GetTask(id)
	if err != nil {
		return task, err
	}
	info, err := storage.containerInfo.GetContainerInfo(task.CurrentState.Host, id)
	if err != nil {
		return task, err
	}
	task.CurrentState.Info = info
	task.Kind = "cluster#task"
	return task, err
}

func (storage *TaskRegistryStorage) Delete(id string) error {
	return storage.registry.DeleteTask(id)
}

func (storage *TaskRegistryStorage) Extract(body string) (interface{}, error) {
	task := Task{}
	err := json.Unmarshal([]byte(body), &task)
	return task, err
}

func (storage *TaskRegistryStorage) Create(task interface{}) error {
	taskObj := task.(Task)
	if len(taskObj.ID) == 0 {
		return fmt.Errorf("ID is unspecified: %#v", task)
	}
	machine, err := storage.scheduler.Schedule(taskObj)
	if err != nil {
		return err
	}
	return storage.registry.CreateTask(machine, taskObj)
}

func (storage *TaskRegistryStorage) Update(task interface{}) error {
	return storage.registry.UpdateTask(task.(Task))
}
