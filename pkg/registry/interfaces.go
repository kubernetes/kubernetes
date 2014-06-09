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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// TaskRegistry is an interface implemented by things that know how to store Task objects
type TaskRegistry interface {
	// ListTasks obtains a list of tasks that match query.
	// Query may be nil in which case all tasks are returned.
	ListTasks(query *map[string]string) ([]api.Pod, error)
	// Get a specific task
	GetTask(taskId string) (*api.Pod, error)
	// Create a task based on a specification, schedule it onto a specific machine.
	CreateTask(machine string, task api.Pod) error
	// Update an existing task
	UpdateTask(task api.Pod) error
	// Delete an existing task
	DeleteTask(taskId string) error
}

// ControllerRegistry is an interface for things that know how to store Controllers
type ControllerRegistry interface {
	ListControllers() ([]api.ReplicationController, error)
	GetController(controllerId string) (*api.ReplicationController, error)
	CreateController(controller api.ReplicationController) error
	UpdateController(controller api.ReplicationController) error
	DeleteController(controllerId string) error
}
