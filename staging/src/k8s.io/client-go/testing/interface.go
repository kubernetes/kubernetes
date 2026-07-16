/*
Copyright 2021 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	restclient "k8s.io/client-go/rest"
)

type FakeClient interface {
	// Tracker gives access to the ObjectTracker internal to the fake client.
	Tracker() ObjectTracker

	// AddReactor appends a reactor to the end of the chain.
	AddReactor(verb, resource string, reaction ReactionFunc)

	// PrependReactor adds a reactor to the beginning of the chain.
	PrependReactor(verb, resource string, reaction ReactionFunc)

	// AddWatchReactor appends a reactor to the end of the chain.
	AddWatchReactor(resource string, reaction WatchReactionFunc)

	// PrependWatchReactor adds a reactor to the beginning of the chain.
	PrependWatchReactor(resource string, reaction WatchReactionFunc)

	// AddProxyReactor appends a reactor to the end of the chain.
	AddProxyReactor(resource string, reaction ProxyReactionFunc)

	// PrependProxyReactor adds a reactor to the beginning of the chain.
	PrependProxyReactor(resource string, reaction ProxyReactionFunc)

	// Invokes records the provided Action and then invokes the ReactionFunc that
	// handles the action if one exists. defaultReturnObj is expected to be of the
	// same type a normal call would return.
	Invokes(action Action, defaultReturnObj runtime.Object) (runtime.Object, error)

	// InvokesWatch records the provided Action and then invokes the ReactionFunc
	// that handles the action if one exists.
	InvokesWatch(action Action) (watch.Interface, error)

	// InvokesProxy records the provided Action and then invokes the ReactionFunc
	// that handles the action if one exists.
	InvokesProxy(action Action) restclient.ResponseWrapper

	// ClearActions clears the history of actions called on the fake client.
	ClearActions()

	// Actions returns a chronologically ordered slice fake actions called on the
	// fake client.
	Actions() []Action
}
