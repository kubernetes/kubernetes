/*
Copyright 2015 The Kubernetes Authors.

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

package kubelet

import (
	"errors"
	"fmt"
	"sync"
	"time"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

type runtimeState struct {
	sync.RWMutex
	lastBaseRuntimeSync      time.Time
	baseRuntimeSyncThreshold time.Duration
	networkError             error
	runtimeError             error
	storageError             error
	cidr                     string
	healthChecks             []*healthCheck
	rtHandlers               []kubecontainer.RuntimeHandler
}

// A health check function should be efficient and not rely on external
// components (e.g., container runtime).
type healthCheckFnType func() (bool, error)

type healthCheck struct {
	name string
	fn   healthCheckFnType
}

func (s *runtimeState) addHealthCheck(name string, f healthCheckFnType) {
	s.Lock()
	defer s.Unlock()
	s.healthChecks = append(s.healthChecks, &healthCheck{name: name, fn: f})
}

func (s *runtimeState) setRuntimeSync(t time.Time) {
	s.Lock()
	defer s.Unlock()
	s.lastBaseRuntimeSync = t
}

func (s *runtimeState) setNetworkState(err error) {
	s.Lock()
	defer s.Unlock()
	s.networkError = err
}

func (s *runtimeState) setRuntimeState(err error) {
	s.Lock()
	defer s.Unlock()
	s.runtimeError = err
}

func (s *runtimeState) setRuntimeHandlers(rtHandlers []kubecontainer.RuntimeHandler) {
	s.Lock()
	defer s.Unlock()
	s.rtHandlers = rtHandlers
}

func (s *runtimeState) runtimeHandlers() []kubecontainer.RuntimeHandler {
	s.RLock()
	defer s.RUnlock()
	return s.rtHandlers
}

func (s *runtimeState) setStorageState(err error) {
	s.Lock()
	defer s.Unlock()
	s.storageError = err
}

func (s *runtimeState) setPodCIDR(cidr string) {
	s.Lock()
	defer s.Unlock()
	s.cidr = cidr
}

func (s *runtimeState) podCIDR() string {
	s.RLock()
	defer s.RUnlock()
	return s.cidr
}

func (s *runtimeState) runtimeErrors() error {
	s.RLock()
	defer s.RUnlock()
	errs := []error{}
	if s.lastBaseRuntimeSync.IsZero() {
		errs = append(errs, errors.New("container runtime status check may not have completed yet"))
	} else if !s.lastBaseRuntimeSync.Add(s.baseRuntimeSyncThreshold).After(time.Now()) {
		errs = append(errs, errors.New("container runtime is down"))
	}
	for _, hc := range s.healthChecks {
		if ok, err := hc.fn(); !ok {
			errs = append(errs, fmt.Errorf("%s is not healthy: %v", hc.name, err))
		}
	}
	if s.runtimeError != nil {
		errs = append(errs, s.runtimeError)
	}

	return utilerrors.NewAggregate(errs)
}

func (s *runtimeState) networkErrors() error {
	s.RLock()
	defer s.RUnlock()
	errs := []error{}
	if s.networkError != nil {
		errs = append(errs, s.networkError)
	}
	return utilerrors.NewAggregate(errs)
}

func (s *runtimeState) storageErrors() error {
	s.RLock()
	defer s.RUnlock()
	errs := []error{}
	if s.storageError != nil {
		errs = append(errs, s.storageError)
	}
	return utilerrors.NewAggregate(errs)
}

func newRuntimeState(runtimeSyncThreshold time.Duration) *runtimeState {
	return &runtimeState{
		lastBaseRuntimeSync:      time.Time{},
		baseRuntimeSyncThreshold: runtimeSyncThreshold,
		networkError:             ErrNetworkUnknown,
	}
}
