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

package client

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// FakeClient implements Interface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type FakeClient struct {
	// FakeClient by default keeps a simple list of the methods that have been called.
	Actions []string
}

func (client *FakeClient) ListPods(selector labels.Selector) (api.PodList, error) {
	client.Actions = append(client.Actions, "list-pods")
	return api.PodList{}, nil
}

func (client *FakeClient) GetPod(name string) (api.Pod, error) {
	client.Actions = append(client.Actions, "get-pod")
	return api.Pod{}, nil
}

func (client *FakeClient) DeletePod(name string) error {
	client.Actions = append(client.Actions, "delete-pod")
	return nil
}

func (client *FakeClient) CreatePod(pod api.Pod) (api.Pod, error) {
	client.Actions = append(client.Actions, "create-pod")
	return api.Pod{}, nil
}

func (client *FakeClient) UpdatePod(pod api.Pod) (api.Pod, error) {
	client.Actions = append(client.Actions, "update-pod")
	return api.Pod{}, nil
}

func (client *FakeClient) ListReplicationControllers(selector labels.Selector) (api.ReplicationControllerList, error) {
	client.Actions = append(client.Actions, "list-controllers")
	return api.ReplicationControllerList{}, nil
}

func (client *FakeClient) GetReplicationController(name string) (api.ReplicationController, error) {
	client.Actions = append(client.Actions, "get-controller")
	return api.ReplicationController{}, nil
}

func (client *FakeClient) CreateReplicationController(controller api.ReplicationController) (api.ReplicationController, error) {
	client.Actions = append(client.Actions, "create-controller")
	return api.ReplicationController{}, nil
}

func (client *FakeClient) UpdateReplicationController(controller api.ReplicationController) (api.ReplicationController, error) {
	client.Actions = append(client.Actions, "update-controller")
	return api.ReplicationController{}, nil
}

func (client *FakeClient) DeleteReplicationController(controller string) error {
	client.Actions = append(client.Actions, "delete-controller")
	return nil
}

func (client *FakeClient) WatchReplicationControllers(label, field labels.Selector, resourceVersion uint64) (watch.Interface, error) {
	client.Actions = append(client.Actions, "watch-controllers")
	return watch.NewFake(), nil
}

func (client *FakeClient) GetService(name string) (api.Service, error) {
	client.Actions = append(client.Actions, "get-controller")
	return api.Service{}, nil
}

func (client *FakeClient) CreateService(controller api.Service) (api.Service, error) {
	client.Actions = append(client.Actions, "create-service")
	return api.Service{}, nil
}

func (client *FakeClient) UpdateService(controller api.Service) (api.Service, error) {
	client.Actions = append(client.Actions, "update-service")
	return api.Service{}, nil
}

func (client *FakeClient) DeleteService(controller string) error {
	client.Actions = append(client.Actions, "delete-service")
	return nil
}
