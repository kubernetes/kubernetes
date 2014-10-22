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
	Events        api.EventList
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
	c.Actions = append(c.Actions, FakeAction{Action: "update-pod", Value: pod.Name})
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

func (c *Fake) WatchReplicationControllers(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
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

func (c *Fake) WatchServices(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
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

func (c *Fake) WatchEndpoints(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
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

func (c *Fake) CreateMinion(minion *api.Minion) (*api.Minion, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "create-minion", Value: minion})
	return &api.Minion{}, nil
}

func (c *Fake) DeleteMinion(id string) error {
	c.Actions = append(c.Actions, FakeAction{Action: "delete-minion", Value: id})
	return nil
}

// CreateEvent makes a new event. Returns the copy of the event the server returns, or an error.
func (c *Fake) CreateEvent(event *api.Event) (*api.Event, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "get-event", Value: event.Name})
	return &api.Event{}, nil
}

// ListEvents returns a list of events matching the selectors.
func (c *Fake) ListEvents(label, field labels.Selector) (*api.EventList, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "list-events"})
	return &c.Events, nil
}

// GetEvent returns the given event, or an error.
func (c *Fake) GetEvent(id string) (*api.Event, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "get-event", Value: id})
	return &api.Event{}, nil
}

// WatchEvents starts watching for events matching the given selectors.
func (c *Fake) WatchEvents(label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "watch-events", Value: resourceVersion})
	return c.Watch, c.Err
}
