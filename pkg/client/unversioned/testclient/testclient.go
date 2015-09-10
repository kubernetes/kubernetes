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
	"fmt"
	"sync"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/registered"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/pkg/watch"
)

// NewSimpleFake returns a client that will respond with the provided objects
func NewSimpleFake(objects ...runtime.Object) *Fake {
	o := NewObjects(api.Scheme, api.Scheme)
	for _, obj := range objects {
		if err := o.Add(obj); err != nil {
			panic(err)
		}
	}

	fakeClient := &Fake{}
	fakeClient.AddReactor("*", "*", ObjectReaction(o, latest.RESTMapper))

	return fakeClient
}

// Fake implements client.Interface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type Fake struct {
	sync.RWMutex
	actions []Action // these may be castable to other types, but "Action" is the minimum

	// ReactionChain is the list of reactors that will be attempted for every request in the order they are tried
	ReactionChain []Reactor
	// WatchReactionChain is the list of watch reactors that will be attempted for every request in the order they are tried
	WatchReactionChain []WatchReactor
}

// Reactor is an interface to allow the composition of reaction functions.
type Reactor interface {
	// Handles indicates whether or not this Reactor deals with a given action
	Handles(action Action) bool
	// React handles the action and returns results.  It may choose to delegate by indicated handled=false
	React(action Action) (handled bool, ret runtime.Object, err error)
}

// WatchReactor is an interface to allow the composition of watch functions.
type WatchReactor interface {
	// Handles indicates whether or not this Reactor deals with a given action
	Handles(action Action) bool
	// React handles a watch action and returns results.  It may choose to delegate by indicated handled=false
	React(action Action) (handled bool, ret watch.Interface, err error)
}

// ReactionFunc is a function that returns an object or error for a given Action.  If "handled" is false,
// then the test client will continue ignore the results and continue to the next ReactionFunc
type ReactionFunc func(action Action) (handled bool, ret runtime.Object, err error)

// WatchReactionFunc is a function that returns a watch interface.  If "handled" is false,
// then the test client will continue ignore the results and continue to the next ReactionFunc
type WatchReactionFunc func(action Action) (handled bool, ret watch.Interface, err error)

// AddReactor appends a reactor to the end of the chain
func (c *Fake) AddReactor(verb, resource string, reaction ReactionFunc) {
	c.ReactionChain = append(c.ReactionChain, &SimpleReactor{verb, resource, reaction})
}

// PrependReactor adds a reactor to the beginning of the chain
func (c *Fake) PrependReactor(verb, resource string, reaction ReactionFunc) {
	newChain := make([]Reactor, 0, len(c.ReactionChain)+1)
	newChain[0] = &SimpleReactor{verb, resource, reaction}
	newChain = append(newChain, c.ReactionChain...)
}

// AddWatchReactor appends a reactor to the end of the chain
func (c *Fake) AddWatchReactor(resource string, reaction WatchReactionFunc) {
	c.WatchReactionChain = append(c.WatchReactionChain, &SimpleWatchReactor{resource, reaction})
}

// Invokes records the provided Action and then invokes the ReactFn (if provided).
// defaultReturnObj is expected to be of the same type a normal call would return.
func (c *Fake) Invokes(action Action, defaultReturnObj runtime.Object) (runtime.Object, error) {
	c.Lock()
	defer c.Unlock()

	c.actions = append(c.actions, action)
	for _, reactor := range c.ReactionChain {
		if !reactor.Handles(action) {
			continue
		}

		handled, ret, err := reactor.React(action)
		if !handled {
			continue
		}

		return ret, err
	}

	return defaultReturnObj, nil
}

// InvokesWatch records the provided Action and then invokes the ReactFn (if provided).
func (c *Fake) InvokesWatch(action Action) (watch.Interface, error) {
	c.Lock()
	defer c.Unlock()

	c.actions = append(c.actions, action)
	for _, reactor := range c.WatchReactionChain {
		if !reactor.Handles(action) {
			continue
		}

		handled, ret, err := reactor.React(action)
		if !handled {
			continue
		}

		return ret, err
	}

	return nil, fmt.Errorf("unhandled watch: %#v", action)
}

// ClearActions clears the history of actions called on the fake client
func (c *Fake) ClearActions() {
	c.Lock()
	c.Unlock()

	c.actions = make([]Action, 0)
}

// Actions returns a chronologically ordered slice fake actions called on the fake client
func (c *Fake) Actions() []Action {
	c.RLock()
	defer c.RUnlock()
	fa := make([]Action, len(c.actions))
	copy(fa, c.actions)
	return fa
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
	return &FakeEvents{Fake: c, Namespace: namespace}
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

func (c *Fake) Experimental() client.ExperimentalInterface {
	return &FakeExperimental{c}
}

func (c *Fake) ServerVersion() (*version.Info, error) {
	action := ActionImpl{}
	action.Verb = "get"
	action.Resource = "version"

	c.Invokes(action, nil)
	versionInfo := version.Get()
	return &versionInfo, nil
}

func (c *Fake) ServerAPIVersions() (*api.APIVersions, error) {
	action := ActionImpl{}
	action.Verb = "get"
	action.Resource = "apiversions"

	c.Invokes(action, nil)
	return &api.APIVersions{Versions: registered.RegisteredVersions}, nil
}

func (c *Fake) ComponentStatuses() client.ComponentStatusInterface {
	return &FakeComponentStatuses{Fake: c}
}

type FakeExperimental struct {
	*Fake
}

func (c *FakeExperimental) DaemonSets(namespace string) client.DaemonSetInterface {
	return &FakeDaemonSets{Fake: c, Namespace: namespace}
}

func (c *FakeExperimental) HorizontalPodAutoscalers(namespace string) client.HorizontalPodAutoscalerInterface {
	return &FakeHorizontalPodAutoscalers{Fake: c, Namespace: namespace}
}

func (c *FakeExperimental) Deployments(namespace string) client.DeploymentInterface {
	return &FakeDeployments{Fake: c, Namespace: namespace}
}

func (c *FakeExperimental) Scales(namespace string) client.ScaleInterface {
	return &FakeScales{Fake: c, Namespace: namespace}
}
