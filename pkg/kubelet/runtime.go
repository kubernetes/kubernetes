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
	"fmt"
	"sync"
	"time"
)

type runtimeState struct {
	sync.RWMutex
	lastBaseRuntimeSync      time.Time
	baseRuntimeSyncThreshold time.Duration
	networkError             error
	cidr                     string
	healthChecks             []*healthCheck
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

func (s *runtimeState) runtimeErrors() []string {
	s.RLock()
	defer s.RUnlock()
	var ret []string
	if !s.lastBaseRuntimeSync.Add(s.baseRuntimeSyncThreshold).After(time.Now()) {
		ret = append(ret, "container runtime is down")
	}
	for _, hc := range s.healthChecks {
		if ok, err := hc.fn(); !ok {
			ret = append(ret, fmt.Sprintf("%s is not healthy: %v", hc.name, err))
		}
	}

	return ret
}

func (s *runtimeState) networkErrors() []string {
	s.RLock()
	defer s.RUnlock()
	var ret []string
	if s.networkError != nil {
		ret = append(ret, s.networkError.Error())
	}
	return ret
}

func newRuntimeState(
	runtimeSyncThreshold time.Duration,
) *runtimeState {
	return &runtimeState{
		lastBaseRuntimeSync:      time.Time{},
		baseRuntimeSyncThreshold: runtimeSyncThreshold,
		networkError:             fmt.Errorf("network state unknown"),
	}
}
