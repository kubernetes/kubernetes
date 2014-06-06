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
	"log"

	"github.com/coreos/go-etcd/etcd"

	. "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// TODO: Need to add a reconciler loop that makes sure that things in tasks are reflected into
//       kubelet (and vice versa)

// EtcdClient is an injectable interface for testing.
type EtcdClient interface {
	AddChild(key, data string, ttl uint64) (*etcd.Response, error)
	Get(key string, sort, recursive bool) (*etcd.Response, error)
	Set(key, value string, ttl uint64) (*etcd.Response, error)
	Create(key, value string, ttl uint64) (*etcd.Response, error)
	Delete(key string, recursive bool) (*etcd.Response, error)
	// I'd like to use directional channels here (e.g. <-chan) but this interface mimics
	// the etcd client interface which doesn't, and it doesn't seem worth it to wrap the api.
	Watch(prefix string, waitIndex uint64, recursive bool, receiver chan *etcd.Response, stop chan bool) (*etcd.Response, error)
}

// EtcdRegistry is an implementation of both ControllerRegistry and TaskRegistry which is backed with etcd.
type EtcdRegistry struct {
	etcdClient      EtcdClient
	machines        []string
	manifestFactory ManifestFactory
}

// MakeEtcdRegistry creates an etcd registry.
// 'client' is the connection to etcd
// 'machines' is the list of machines
// 'scheduler' is the scheduling algorithm to use.
func MakeEtcdRegistry(client EtcdClient, machines []string) *EtcdRegistry {
	registry := &EtcdRegistry{
		etcdClient: client,
		machines:   machines,
	}
	registry.manifestFactory = &BasicManifestFactory{
		serviceRegistry: registry,
	}
	return registry
}

func makeTaskKey(machine, taskID string) string {
	return "/registry/hosts/" + machine + "/tasks/" + taskID
}

func (registry *EtcdRegistry) ListTasks(query *map[string]string) ([]Task, error) {
	tasks := []Task{}
	for _, machine := range registry.machines {
		machineTasks, err := registry.listTasksForMachine(machine)
		if err != nil {
			return tasks, err
		}
		for _, task := range machineTasks {
			if LabelsMatch(task, query) {
				tasks = append(tasks, task)
			}
		}
	}
	return tasks, nil
}

func (registry *EtcdRegistry) listEtcdNode(key string) ([]*etcd.Node, error) {
	result, err := registry.etcdClient.Get(key, false, true)
	if err != nil {
		nodes := make([]*etcd.Node, 0)
		if isEtcdNotFound(err) {
			return nodes, nil
		} else {
			return nodes, err
		}
	}
	return result.Node.Nodes, nil
}

func (registry *EtcdRegistry) listTasksForMachine(machine string) ([]Task, error) {
	tasks := []Task{}
	key := "/registry/hosts/" + machine + "/tasks"
	nodes, err := registry.listEtcdNode(key)
	for _, node := range nodes {
		task := Task{}
		err = json.Unmarshal([]byte(node.Value), &task)
		if err != nil {
			return tasks, err
		}
		task.CurrentState.Host = machine
		tasks = append(tasks, task)
	}
	return tasks, err
}

func (registry *EtcdRegistry) GetTask(taskID string) (*Task, error) {
	task, _, err := registry.findTask(taskID)
	return &task, err
}

func makeContainerKey(machine string) string {
	return "/registry/hosts/" + machine + "/kubelet"
}

func (registry *EtcdRegistry) loadManifests(machine string) ([]ContainerManifest, error) {
	var manifests []ContainerManifest
	response, err := registry.etcdClient.Get(makeContainerKey(machine), false, false)

	if err != nil {
		if isEtcdNotFound(err) {
			err = nil
			manifests = []ContainerManifest{}
		}
	} else {
		err = json.Unmarshal([]byte(response.Node.Value), &manifests)
	}
	return manifests, err
}

func (registry *EtcdRegistry) updateManifests(machine string, manifests []ContainerManifest) error {
	containerData, err := json.Marshal(manifests)
	if err != nil {
		return err
	}
	_, err = registry.etcdClient.Set(makeContainerKey(machine), string(containerData), 0)
	return err
}

func (registry *EtcdRegistry) CreateTask(machineIn string, task Task) error {
	taskOut, machine, err := registry.findTask(task.ID)
	if err == nil {
		return fmt.Errorf("A task named %s already exists on %s (%#v)", task.ID, machine, taskOut)
	}
	return registry.runTask(task, machineIn)
}

func (registry *EtcdRegistry) runTask(task Task, machine string) error {
	manifests, err := registry.loadManifests(machine)
	if err != nil {
		return err
	}

	key := makeTaskKey(machine, task.ID)
	data, err := json.Marshal(task)
	if err != nil {
		return err
	}
	_, err = registry.etcdClient.Create(key, string(data), 0)

	manifest, err := registry.manifestFactory.MakeManifest(machine, task)
	if err != nil {
		return err
	}
	manifests = append(manifests, manifest)
	return registry.updateManifests(machine, manifests)
}

func (registry *EtcdRegistry) UpdateTask(task Task) error {
	return fmt.Errorf("Unimplemented!")
}

func (registry *EtcdRegistry) DeleteTask(taskID string) error {
	_, machine, err := registry.findTask(taskID)
	if err != nil {
		return err
	}
	return registry.deleteTaskFromMachine(machine, taskID)
}

func (registry *EtcdRegistry) deleteTaskFromMachine(machine, taskID string) error {
	manifests, err := registry.loadManifests(machine)
	if err != nil {
		return err
	}
	newManifests := make([]ContainerManifest, 0)
	found := false
	for _, manifest := range manifests {
		if manifest.Id != taskID {
			newManifests = append(newManifests, manifest)
		} else {
			found = true
		}
	}
	if !found {
		// This really shouldn't happen, it indicates something is broken, and likely
		// there is a lost task somewhere.
		// However it is "deleted" so log it and move on
		log.Printf("Couldn't find: %s in %#v", taskID, manifests)
	}
	if err = registry.updateManifests(machine, newManifests); err != nil {
		return err
	}
	key := makeTaskKey(machine, taskID)
	_, err = registry.etcdClient.Delete(key, true)
	return err
}

func (registry *EtcdRegistry) getTaskForMachine(machine, taskID string) (Task, error) {
	key := makeTaskKey(machine, taskID)
	result, err := registry.etcdClient.Get(key, false, false)
	if err != nil {
		if isEtcdNotFound(err) {
			return Task{}, fmt.Errorf("Not found (%#v).", err)
		} else {
			return Task{}, err
		}
	}
	if result.Node == nil || len(result.Node.Value) == 0 {
		return Task{}, fmt.Errorf("no nodes field: %#v", result)
	}
	task := Task{}
	err = json.Unmarshal([]byte(result.Node.Value), &task)
	task.CurrentState.Host = machine
	return task, err
}

func (registry *EtcdRegistry) findTask(taskID string) (Task, string, error) {
	for _, machine := range registry.machines {
		task, err := registry.getTaskForMachine(machine, taskID)
		if err == nil {
			return task, machine, nil
		}
	}
	return Task{}, "", fmt.Errorf("Task not found %s", taskID)
}

func isEtcdNotFound(err error) bool {
	if err == nil {
		return false
	}
	switch err.(type) {
	case *etcd.EtcdError:
		etcdError := err.(*etcd.EtcdError)
		if etcdError == nil {
			return false
		}
		if etcdError.ErrorCode == 100 {
			return true
		}
	}
	return false
}

func (registry *EtcdRegistry) ListControllers() ([]ReplicationController, error) {
	var controllers []ReplicationController
	key := "/registry/controllers"
	nodes, err := registry.listEtcdNode(key)
	for _, node := range nodes {
		var controller ReplicationController
		err = json.Unmarshal([]byte(node.Value), &controller)
		if err != nil {
			return controllers, err
		}
		controllers = append(controllers, controller)
	}
	return controllers, nil
}

func makeControllerKey(id string) string {
	return "/registry/controllers/" + id
}

func (registry *EtcdRegistry) GetController(controllerID string) (*ReplicationController, error) {
	var controller ReplicationController
	key := makeControllerKey(controllerID)
	result, err := registry.etcdClient.Get(key, false, false)
	if err != nil {
		if isEtcdNotFound(err) {
			return nil, fmt.Errorf("Controller %s not found", controllerID)
		} else {
			return nil, err
		}
	}
	if result.Node == nil || len(result.Node.Value) == 0 {
		return nil, fmt.Errorf("no nodes field: %#v", result)
	}
	err = json.Unmarshal([]byte(result.Node.Value), &controller)
	return &controller, err
}

func (registry *EtcdRegistry) CreateController(controller ReplicationController) error {
	// TODO : check for existence here and error.
	return registry.UpdateController(controller)
}

func (registry *EtcdRegistry) UpdateController(controller ReplicationController) error {
	controllerData, err := json.Marshal(controller)
	if err != nil {
		return err
	}
	key := makeControllerKey(controller.ID)
	_, err = registry.etcdClient.Set(key, string(controllerData), 0)
	return err
}

func (registry *EtcdRegistry) DeleteController(controllerID string) error {
	key := makeControllerKey(controllerID)
	_, err := registry.etcdClient.Delete(key, false)
	return err
}

func makeServiceKey(name string) string {
	return "/registry/services/specs/" + name
}

func (registry *EtcdRegistry) ListServices() (ServiceList, error) {
	nodes, err := registry.listEtcdNode("/registry/services/specs")
	if err != nil {
		return ServiceList{}, err
	}

	var services []Service
	for _, node := range nodes {
		var svc Service
		err := json.Unmarshal([]byte(node.Value), &svc)
		if err != nil {
			return ServiceList{}, err
		}
		services = append(services, svc)
	}
	return ServiceList{Items: services}, nil
}

func (registry *EtcdRegistry) CreateService(svc Service) error {
	key := makeServiceKey(svc.ID)
	data, err := json.Marshal(svc)
	if err != nil {
		return err
	}
	_, err = registry.etcdClient.Set(key, string(data), 0)
	return err
}

func (registry *EtcdRegistry) GetService(name string) (*Service, error) {
	key := makeServiceKey(name)
	response, err := registry.etcdClient.Get(key, false, false)
	if err != nil {
		if isEtcdNotFound(err) {
			return nil, fmt.Errorf("Service %s was not found.", name)
		} else {
			return nil, err
		}
	}
	var svc Service
	err = json.Unmarshal([]byte(response.Node.Value), &svc)
	if err != nil {
		return nil, err
	}
	return &svc, err
}

func (registry *EtcdRegistry) DeleteService(name string) error {
	key := makeServiceKey(name)
	_, err := registry.etcdClient.Delete(key, true)
	if err != nil {
		return err
	}
	key = "/registry/services/endpoints/" + name
	_, err = registry.etcdClient.Delete(key, true)
	return err
}

func (registry *EtcdRegistry) UpdateService(svc Service) error {
	return registry.CreateService(svc)
}

func (registry *EtcdRegistry) UpdateEndpoints(e Endpoints) error {
	data, err := json.Marshal(e)
	if err != nil {
		return err
	}
	_, err = registry.etcdClient.Set("/registry/services/endpoints/"+e.Name, string(data), 0)
	return err
}
