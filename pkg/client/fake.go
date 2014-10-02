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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

type FakeAction struct {
	Action string
	Value  interface{}
}

// Fake implements Interface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type Fake struct {
	// Fake by default keeps a simple list of the methods that have been called.
	Actions       []FakeAction
	Pods          api.PodList
	Ctrl          api.ReplicationController
	ServiceList   api.ServiceList
	EndpointsList api.EndpointsList
	Minions       api.MinionList
	Err           error
	Watch         watch.Interface
}

func (c *Fake) ListPods(ctx api.Context, selector labels.Selector) (*api.PodList, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "list-pods"})
	return api.Scheme.CopyOrDie(&c.Pods).(*api.PodList), nil
}

func (c *Fake) GetPod(ctx api.Context, name string) (*api.Pod, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "get-pod", Value: name})
	return &api.Pod{}, nil
}

func (c *Fake) DeletePod(ctx api.Context, name string) error {
	c.Actions = append(c.Actions, FakeAction{Action: "delete-pod", Value: name})
	return nil
}

func (c *Fake) CreatePod(ctx api.Context, pod *api.Pod) (*api.Pod, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "create-pod"})
	return &api.Pod{}, nil
}

func (c *Fake) UpdatePod(ctx api.Context, pod *api.Pod) (*api.Pod, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "update-pod", Value: pod.ID})
	return &api.Pod{}, nil
}

func (c *Fake) ListReplicationControllers(ctx api.Context, selector labels.Selector) (*api.ReplicationControllerList, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "list-controllers"})
	return &api.ReplicationControllerList{}, nil
}

func (c *Fake) GetReplicationController(ctx api.Context, name string) (*api.ReplicationController, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "get-controller", Value: name})
	return api.Scheme.CopyOrDie(&c.Ctrl).(*api.ReplicationController), nil
}

func (c *Fake) CreateReplicationController(ctx api.Context, controller *api.ReplicationController) (*api.ReplicationController, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "create-controller", Value: controller})
	return &api.ReplicationController{}, nil
}

func (c *Fake) UpdateReplicationController(ctx api.Context, controller *api.ReplicationController) (*api.ReplicationController, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "update-controller", Value: controller})
	return &api.ReplicationController{}, nil
}

func (c *Fake) DeleteReplicationController(ctx api.Context, controller string) error {
	c.Actions = append(c.Actions, FakeAction{Action: "delete-controller", Value: controller})
	return nil
}

func (c *Fake) WatchReplicationControllers(ctx api.Context, label, field labels.Selector, resourceVersion uint64) (watch.Interface, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "watch-controllers", Value: resourceVersion})
	return c.Watch, nil
}

func (c *Fake) ListServices(ctx api.Context, selector labels.Selector) (*api.ServiceList, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "list-services"})
	return &c.ServiceList, c.Err
}

func (c *Fake) GetService(ctx api.Context, name string) (*api.Service, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "get-service", Value: name})
	return &api.Service{}, nil
}

func (c *Fake) CreateService(ctx api.Context, service *api.Service) (*api.Service, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "create-service", Value: service})
	return &api.Service{}, nil
}

func (c *Fake) UpdateService(ctx api.Context, service *api.Service) (*api.Service, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "update-service", Value: service})
	return &api.Service{}, nil
}

func (c *Fake) DeleteService(ctx api.Context, service string) error {
	c.Actions = append(c.Actions, FakeAction{Action: "delete-service", Value: service})
	return nil
}

func (c *Fake) WatchServices(ctx api.Context, label, field labels.Selector, resourceVersion uint64) (watch.Interface, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "watch-services", Value: resourceVersion})
	return c.Watch, c.Err
}

func (c *Fake) ListEndpoints(ctx api.Context, selector labels.Selector) (*api.EndpointsList, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "list-endpoints"})
	return api.Scheme.CopyOrDie(&c.EndpointsList).(*api.EndpointsList), c.Err
}

func (c *Fake) GetEndpoints(ctx api.Context, name string) (*api.Endpoints, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "get-endpoints"})
	return &api.Endpoints{}, nil
}

func (c *Fake) WatchEndpoints(ctx api.Context, label, field labels.Selector, resourceVersion uint64) (watch.Interface, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "watch-endpoints", Value: resourceVersion})
	return c.Watch, c.Err
}

func (c *Fake) ServerVersion() (*version.Info, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "get-version", Value: nil})
	versionInfo := version.Get()
	return &versionInfo, nil
}

func (c *Fake) ListMinions() (*api.MinionList, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "list-minions", Value: nil})
	return &c.Minions, nil
}
