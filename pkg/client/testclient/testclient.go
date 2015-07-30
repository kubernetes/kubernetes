/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package testclient

import (
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/registered"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// NewSimpleFake returns a client that will respond with the provided objects
func NewSimpleFake(objects ...runtime.Object) *Fake {
	o := NewObjects(api.Scheme, api.Scheme)
	for _, obj := range objects {
		if err := o.Add(obj); err != nil {
			panic(err)
		}
	}
	return &Fake{ReactFn: ObjectReaction(o, latest.RESTMapper)}
}

type FakeAction struct {
	Action string
	Value  interface{}
}

// ReactionFunc is a function that returns an object or error for a given Action
type ReactionFunc func(FakeAction) (runtime.Object, error)

// Fake implements client.Interface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type Fake struct {
	sync.RWMutex
	actions []FakeAction
	err     error

	Watch watch.Interface
	// ReactFn is an optional function that will be invoked with the provided action
	// and return a response. It can implement scenario specific behavior. The type
	// of object returned must match the expected type from the caller (even if nil).
	ReactFn ReactionFunc
}

// Invokes records the provided FakeAction and then invokes the ReactFn (if provided).
// obj is expected to be of the same type a normal call would return.
func (c *Fake) Invokes(action FakeAction, obj runtime.Object) (runtime.Object, error) {
	c.Lock()
	defer c.Unlock()

	c.actions = append(c.actions, action)
	if c.ReactFn != nil {
		return c.ReactFn(action)
	}
	return obj, c.err
}

// ClearActions clears the history of actions called on the fake client
func (c *Fake) ClearActions() {
	c.Lock()
	c.Unlock()

	c.actions = make([]FakeAction, 0)
}

// Actions returns a chronologically ordered slice fake actions called on the fake client
func (c *Fake) Actions() []FakeAction {
	c.RLock()
	defer c.RUnlock()
	fa := make([]FakeAction, len(c.actions))
	copy(fa, c.actions)
	return fa
}

// SetErr sets the error to return for client calls
func (c *Fake) SetErr(err error) {
	c.Lock()
	defer c.Unlock()

	c.err = err
}

// Err returns any a client error or nil
func (c *Fake) Err() error {
	c.RLock()
	c.RUnlock()

	return c.err
}

func (c *Fake) LimitRanges(namespace string) client.LimitRangeInterface {
	return &FakeLimitRanges{Fake: c, Namespace: namespace}
}

func (c *Fake) ResourceQuotas(namespace string) client.ResourceQuotaInterface {
	return &FakeResourceQuotas{Fake: c, Namespace: namespace}
}

func (c *Fake) ReplicationControllers(namespace string) client.ReplicationControllerInterface {
	return &FakeReplicationControllers{Fake: c, Namespace: namespace}
}

func (c *Fake) Nodes() client.NodeInterface {
	return &FakeNodes{Fake: c}
}

func (c *Fake) Events(namespace string) client.EventInterface {
	return &FakeEvents{Fake: c}
}

func (c *Fake) Endpoints(namespace string) client.EndpointsInterface {
	return &FakeEndpoints{Fake: c, Namespace: namespace}
}

func (c *Fake) PersistentVolumes() client.PersistentVolumeInterface {
	return &FakePersistentVolumes{Fake: c}
}

func (c *Fake) PersistentVolumeClaims(namespace string) client.PersistentVolumeClaimInterface {
	return &FakePersistentVolumeClaims{Fake: c, Namespace: namespace}
}

func (c *Fake) Pods(namespace string) client.PodInterface {
	return &FakePods{Fake: c, Namespace: namespace}
}

func (c *Fake) PodTemplates(namespace string) client.PodTemplateInterface {
	return &FakePodTemplates{Fake: c, Namespace: namespace}
}

func (c *Fake) Services(namespace string) client.ServiceInterface {
	return &FakeServices{Fake: c, Namespace: namespace}
}

func (c *Fake) ServiceAccounts(namespace string) client.ServiceAccountsInterface {
	return &FakeServiceAccounts{Fake: c, Namespace: namespace}
}

func (c *Fake) Secrets(namespace string) client.SecretsInterface {
	return &FakeSecrets{Fake: c, Namespace: namespace}
}

func (c *Fake) Namespaces() client.NamespaceInterface {
	return &FakeNamespaces{Fake: c}
}

func (c *Fake) ServerVersion() (*version.Info, error) {
	c.Invokes(FakeAction{Action: "get-version", Value: nil}, nil)
	versionInfo := version.Get()
	return &versionInfo, nil
}

func (c *Fake) ServerAPIVersions() (*api.APIVersions, error) {
	c.Invokes(FakeAction{Action: "get-apiversions", Value: nil}, nil)
	return &api.APIVersions{Versions: registered.RegisteredVersions}, nil
}

func (c *Fake) ComponentStatuses() client.ComponentStatusInterface {
	return &FakeComponentStatuses{Fake: c}
}
