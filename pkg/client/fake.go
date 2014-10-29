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
	Actions       []FakeAction
	PodsList      api.PodList
	Ctrl          api.ReplicationController
	ServiceList   api.ServiceList
	EndpointsList api.EndpointsList
	MinionsList   api.MinionList
	EventsList    api.EventList
	Err           error
	Watch         watch.Interface
}

func (c *Fake) ReplicationControllers(namespace string) ReplicationControllerInterface {
	return &FakeReplicationControllers{Fake: c, Namespace: namespace}
}

func (c *Fake) Minions() MinionInterface {
	return &FakeMinions{Fake: c}
}

func (c *Fake) Events() EventInterface {
	return &FakeEvents{Fake: c}
}

func (c *Fake) Endpoints(namespace string) EndpointsInterface {
	return &FakeEndpoints{Fake: c, Namespace: namespace}
}

func (c *Fake) Pods(namespace string) PodInterface {
	return &FakePods{Fake: c, Namespace: namespace}
}

func (c *Fake) Services(namespace string) ServiceInterface {
	return &FakeServices{Fake: c, Namespace: namespace}
}

func (c *Fake) ServerVersion() (*version.Info, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "get-version", Value: nil})
	versionInfo := version.Get()
	return &versionInfo, nil
}

func (c *Fake) ServerAPIVersions() (*version.APIVersions, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "get-apiversions", Value: nil})
	return &version.APIVersions{Versions: []string{"v1beta1", "v1beta2"}}, nil
}
