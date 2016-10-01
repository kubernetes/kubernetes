/*
Copyright 2014 The Kubernetes Authors.

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

package genericapiserver

import (
	"fmt"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/client/restclient"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
)

// PostStartHookFunc is a function that is called after the server has started.
// It must properly handle cases like:
//  1. asynchronous start in multiple API server processes
//  2. conflicts between the different processes all trying to perform the same action
//  3. partially complete work (API server crashes while running your hook)
//  4. API server access **BEFORE** your hook has completed
// Think of it like a mini-controller that is super privileged and gets to run in-process
// If you use this feature, tag @deads2k on github who has promised to review code for anyone's PostStartHook
// until it becomes easier to use.
type PostStartHookFunc func(context PostStartHookContext) error

// PostStartHookContext provides information about this API server to a PostStartHookFunc
type PostStartHookContext struct {
	// LoopbackClientConfig is a config for a privileged loopback connection to the API server
	LoopbackClientConfig *restclient.Config
}

// PostStartHookProvider is an interface in addition to provide a post start hook for the api server
type PostStartHookProvider interface {
	PostStartHook() (string, PostStartHookFunc, error)
}

// AddPostStartHook allows you to add a PostStartHook.
func (s *GenericAPIServer) AddPostStartHook(name string, hook PostStartHookFunc) error {
	if len(name) == 0 {
		return fmt.Errorf("missing name")
	}
	if hook == nil {
		return nil
	}

	s.postStartHookLock.Lock()
	defer s.postStartHookLock.Unlock()

	if s.postStartHooksCalled {
		return fmt.Errorf("unable to add %q because PostStartHooks have already been called", name)
	}
	if s.postStartHooks == nil {
		s.postStartHooks = map[string]PostStartHookFunc{}
	}
	if _, exists := s.postStartHooks[name]; exists {
		return fmt.Errorf("unable to add %q because it is already registered", name)
	}

	s.postStartHooks[name] = hook

	return nil
}

// RunPostStartHooks runs the PostStartHooks for the server
func (s *GenericAPIServer) RunPostStartHooks() {
	s.postStartHookLock.Lock()
	defer s.postStartHookLock.Unlock()
	s.postStartHooksCalled = true

	context := PostStartHookContext{LoopbackClientConfig: s.LoopbackClientConfig}

	for hookName, hook := range s.postStartHooks {
		go runPostStartHook(hookName, hook, context)
	}
}

func runPostStartHook(name string, hook PostStartHookFunc, context PostStartHookContext) {
	var err error
	func() {
		// don't let the hook *accidentally* panic and kill the server
		defer utilruntime.HandleCrash()
		err = hook(context)
	}()
	// if the hook intentionally wants to kill server, let it.
	if err != nil {
		glog.Fatalf("PostStartHook %q failed: %v", name, err)
	}
}
