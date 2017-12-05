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

package server

import (
	"errors"
	"fmt"
	"net/http"

	"github.com/golang/glog"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/apiserver/pkg/server/types"
)

type postStartHookEntry struct {
	hook types.PostStartHookFunc

	// done will be closed when the postHook is finished
	done chan struct{}
}

type preShutdownHookEntry struct {
	hook types.PreShutdownHookFunc
}

// AddPostStartHook allows you to add a PostStartHook.
func (s *GenericAPIServer) AddPostStartHook(name string, hook types.PostStartHookFunc) error {
	if len(name) == 0 {
		return fmt.Errorf("missing name")
	}
	if hook == nil {
		return nil
	}
	if s.disabledPostStartHooks.Has(name) {
		return nil
	}

	s.postStartHookLock.Lock()
	defer s.postStartHookLock.Unlock()

	if s.postStartHooksCalled {
		return fmt.Errorf("unable to add %q because PostStartHooks have already been called", name)
	}
	if _, exists := s.postStartHooks[name]; exists {
		return fmt.Errorf("unable to add %q because it is already registered", name)
	}

	// done is closed when the poststarthook is finished.  This is used by the health check to be able to indicate
	// that the poststarthook is finished
	done := make(chan struct{})
	s.AddHealthzChecks(postStartHookHealthz{name: "poststarthook/" + name, done: done})
	s.postStartHooks[name] = postStartHookEntry{hook: hook, done: done}

	return nil
}

// AddPostStartHookOrDie allows you to add a PostStartHook, but dies on failure
func (s *GenericAPIServer) AddPostStartHookOrDie(name string, hook types.PostStartHookFunc) {
	if err := s.AddPostStartHook(name, hook); err != nil {
		glog.Fatalf("Error registering PostStartHook %q: %v", name, err)
	}
}

// AddPreShutdownHook allows you to add a PreShutdownHook.
func (s *GenericAPIServer) AddPreShutdownHook(name string, hook types.PreShutdownHookFunc) error {
	if len(name) == 0 {
		return fmt.Errorf("missing name")
	}
	if hook == nil {
		return nil
	}

	s.preShutdownHookLock.Lock()
	defer s.preShutdownHookLock.Unlock()

	if s.preShutdownHooksCalled {
		return fmt.Errorf("unable to add %q because PreShutdownHooks have already been called", name)
	}
	if _, exists := s.preShutdownHooks[name]; exists {
		return fmt.Errorf("unable to add %q because it is already registered", name)
	}

	s.preShutdownHooks[name] = preShutdownHookEntry{hook: hook}

	return nil
}

// AddPreShutdownHookOrDie allows you to add a PostStartHook, but dies on failure
func (s *GenericAPIServer) AddPreShutdownHookOrDie(name string, hook types.PreShutdownHookFunc) {
	if err := s.AddPreShutdownHook(name, hook); err != nil {
		glog.Fatalf("Error registering PreShutdownHook %q: %v", name, err)
	}
}

// RunPostStartHooks runs the PostStartHooks for the server
func (s *GenericAPIServer) RunPostStartHooks(stopCh <-chan struct{}) {
	s.postStartHookLock.Lock()
	defer s.postStartHookLock.Unlock()
	s.postStartHooksCalled = true

	context := types.PostStartHookContext{
		LoopbackClientConfig: s.LoopbackClientConfig,
		StopCh:               stopCh,
	}

	for hookName, hookEntry := range s.postStartHooks {
		go runPostStartHook(hookName, hookEntry, context)
	}
}

// RunPreShutdownHooks runs the PreShutdownHooks for the server
func (s *GenericAPIServer) RunPreShutdownHooks() error {
	var errorList []error

	s.preShutdownHookLock.Lock()
	defer s.preShutdownHookLock.Unlock()
	s.preShutdownHooksCalled = true

	for hookName, hookEntry := range s.preShutdownHooks {
		if err := runPreShutdownHook(hookName, hookEntry); err != nil {
			errorList = append(errorList, err)
		}
	}
	return utilerrors.NewAggregate(errorList)
}

// isPostStartHookRegistered checks whether a given PostStartHook is registered
func (s *GenericAPIServer) isPostStartHookRegistered(name string) bool {
	s.postStartHookLock.Lock()
	defer s.postStartHookLock.Unlock()
	_, exists := s.postStartHooks[name]
	return exists
}

func runPostStartHook(name string, entry postStartHookEntry, context types.PostStartHookContext) {
	var err error
	func() {
		// don't let the hook *accidentally* panic and kill the server
		defer utilruntime.HandleCrash()
		err = entry.hook(context)
	}()
	// if the hook intentionally wants to kill server, let it.
	if err != nil {
		glog.Fatalf("PostStartHook %q failed: %v", name, err)
	}
	close(entry.done)
}

func runPreShutdownHook(name string, entry preShutdownHookEntry) error {
	var err error
	func() {
		// don't let the hook *accidentally* panic and kill the server
		defer utilruntime.HandleCrash()
		err = entry.hook()
	}()
	if err != nil {
		return fmt.Errorf("PreShutdownHook %q failed: %v", name, err)
	}
	return nil
}

// postStartHookHealthz implements a healthz check for poststarthooks.  It will return a "hookNotFinished"
// error until the poststarthook is finished.
type postStartHookHealthz struct {
	name string

	// done will be closed when the postStartHook is finished
	done chan struct{}
}

var _ healthz.HealthzChecker = postStartHookHealthz{}

func (h postStartHookHealthz) Name() string {
	return h.name
}

var hookNotFinished = errors.New("not finished")

func (h postStartHookHealthz) Check(req *http.Request) error {
	select {
	case <-h.done:
		return nil
	default:
		return hookNotFinished
	}
}
