/*
Copyright 2016 The Kubernetes Authors.

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

package testing

import (
	"fmt"
	"sync"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	restclient "k8s.io/client-go/rest"
)

// Fake implements client.Interface. Meant to be embedded into a struct to get
// a default implementation. This makes faking out just the method you want to
// test easier.
type Fake struct {
	sync.RWMutex
	actions []Action // these may be castable to other types, but "Action" is the minimum

	// ReactionChain is the list of reactors that will be attempted for every
	// request in the order they are tried.
	ReactionChain []Reactor
	// WatchReactionChain is the list of watch reactors that will be attempted
	// for every request in the order they are tried.
	WatchReactionChain []WatchReactor
	// ProxyReactionChain is the list of proxy reactors that will be attempted
	// for every request in the order they are tried.
	ProxyReactionChain []ProxyReactor

	Resources []*metav1.APIResourceList
}

// Reactor is an interface to allow the composition of reaction functions.
type Reactor interface {
	// Handles indicates whether or not this Reactor deals with a given
	// action.
	Handles(action Action) bool
	// React handles the action and returns results.  It may choose to
	// delegate by indicated handled=false.
	React(action Action) (handled bool, ret runtime.Object, err error)
}

// WatchReactor is an interface to allow the composition of watch functions.
type WatchReactor interface {
	// Handles indicates whether or not this Reactor deals with a given
	// action.
	Handles(action Action) bool
	// React handles a watch action and returns results.  It may choose to
	// delegate by indicating handled=false.
	React(action Action) (handled bool, ret watch.Interface, err error)
}

// ProxyReactor is an interface to allow the composition of proxy get
// functions.
type ProxyReactor interface {
	// Handles indicates whether or not this Reactor deals with a given
	// action.
	Handles(action Action) bool
	// React handles a watch action and returns results.  It may choose to
	// delegate by indicating handled=false.
	React(action Action) (handled bool, ret restclient.ResponseWrapper, err error)
}

// ReactionFunc is a function that returns an object or error for a given
// Action.  If "handled" is false, then the test client will ignore the
// results and continue to the next ReactionFunc.  A ReactionFunc can describe
// reactions on subresources by testing the result of the action's
// GetSubresource() method.
type ReactionFunc func(action Action) (handled bool, ret runtime.Object, err error)

// WatchReactionFunc is a function that returns a watch interface.  If
// "handled" is false, then the test client will ignore the results and
// continue to the next ReactionFunc.
type WatchReactionFunc func(action Action) (handled bool, ret watch.Interface, err error)

// ProxyReactionFunc is a function that returns a ResponseWrapper interface
// for a given Action.  If "handled" is false, then the test client will
// ignore the results and continue to the next ProxyReactionFunc.
type ProxyReactionFunc func(action Action) (handled bool, ret restclient.ResponseWrapper, err error)

// AddReactor appends a reactor to the end of the chain.
func (c *Fake) AddReactor(verb, resource string, reaction ReactionFunc) {
	c.ReactionChain = append(c.ReactionChain, &SimpleReactor{verb, resource, reaction})
}

// PrependReactor adds a reactor to the beginning of the chain.
func (c *Fake) PrependReactor(verb, resource string, reaction ReactionFunc) {
	c.ReactionChain = append([]Reactor{&SimpleReactor{verb, resource, reaction}}, c.ReactionChain...)
}

// AddWatchReactor appends a reactor to the end of the chain.
func (c *Fake) AddWatchReactor(resource string, reaction WatchReactionFunc) {
	c.Lock()
	defer c.Unlock()
	c.WatchReactionChain = append(c.WatchReactionChain, &SimpleWatchReactor{resource, reaction})
}

// PrependWatchReactor adds a reactor to the beginning of the chain.
func (c *Fake) PrependWatchReactor(resource string, reaction WatchReactionFunc) {
	c.Lock()
	defer c.Unlock()
	c.WatchReactionChain = append([]WatchReactor{&SimpleWatchReactor{resource, reaction}}, c.WatchReactionChain...)
}

// AddProxyReactor appends a reactor to the end of the chain.
func (c *Fake) AddProxyReactor(resource string, reaction ProxyReactionFunc) {
	c.ProxyReactionChain = append(c.ProxyReactionChain, &SimpleProxyReactor{resource, reaction})
}

// PrependProxyReactor adds a reactor to the beginning of the chain.
func (c *Fake) PrependProxyReactor(resource string, reaction ProxyReactionFunc) {
	c.ProxyReactionChain = append([]ProxyReactor{&SimpleProxyReactor{resource, reaction}}, c.ProxyReactionChain...)
}

// Invokes records the provided Action and then invokes the ReactionFunc that
// handles the action if one exists. defaultReturnObj is expected to be of the
// same type a normal call would return.
func (c *Fake) Invokes(action Action, defaultReturnObj runtime.Object) (runtime.Object, error) {
	c.Lock()
	defer c.Unlock()

	actionCopy := action.DeepCopy()
	c.actions = append(c.actions, action.DeepCopy())
	for _, reactor := range c.ReactionChain {
		if !reactor.Handles(actionCopy) {
			continue
		}

		handled, ret, err := reactor.React(actionCopy)
		if !handled {
			continue
		}

		return ret, err
	}

	return defaultReturnObj, nil
}

// InvokesWatch records the provided Action and then invokes the ReactionFunc
// that handles the action if one exists.
func (c *Fake) InvokesWatch(action Action) (watch.Interface, error) {
	c.Lock()
	defer c.Unlock()

	actionCopy := action.DeepCopy()
	c.actions = append(c.actions, action.DeepCopy())
	for _, reactor := range c.WatchReactionChain {
		if !reactor.Handles(actionCopy) {
			continue
		}

		handled, ret, err := reactor.React(actionCopy)
		if !handled {
			continue
		}

		return ret, err
	}

	return nil, fmt.Errorf("unhandled watch: %#v", action)
}

// InvokesProxy records the provided Action and then invokes the ReactionFunc
// that handles the action if one exists.
func (c *Fake) InvokesProxy(action Action) restclient.ResponseWrapper {
	c.Lock()
	defer c.Unlock()

	actionCopy := action.DeepCopy()
	c.actions = append(c.actions, action.DeepCopy())
	for _, reactor := range c.ProxyReactionChain {
		if !reactor.Handles(actionCopy) {
			continue
		}

		handled, ret, err := reactor.React(actionCopy)
		if !handled || err != nil {
			continue
		}

		return ret
	}

	return nil
}

// ClearActions clears the history of actions called on the fake client.
func (c *Fake) ClearActions() {
	c.Lock()
	defer c.Unlock()

	c.actions = make([]Action, 0)
}

// Actions returns a chronologically ordered slice fake actions called on the
// fake client.
func (c *Fake) Actions() []Action {
	c.RLock()
	defer c.RUnlock()
	fa := make([]Action, len(c.actions))
	copy(fa, c.actions)
	return fa
}
