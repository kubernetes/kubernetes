/*
Copyright 2017 The Kubernetes Authors.

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

package configuration

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
)

const (
	defaultInterval         = 1 * time.Second
	defaultFailureThreshold = 5
)

type getFunc func() (runtime.Object, error)

// When running, poller calls `get` every `interval`. If `get` is
// successful, `Ready()` returns ready and `configuration()` returns the
// `mergedConfiguration`; if `get` has failed more than `failureThreshold ` times,
// `Ready()` returns not ready and `configuration()` returns nil configuration.
// In an HA setup, the poller is consistent only if the `get` is
// doing consistent read.
type poller struct {
	// a function to consistently read the latest configuration
	get getFunc
	// consistent read interval
	interval time.Duration
	// if the number of consecutive read failure equals or exceeds the failureThreshold , the
	// configuration is regarded as not ready.
	failureThreshold int
	// number of consecutive failures so far.
	failures int
	// if the configuration is regarded as ready.
	ready               bool
	mergedConfiguration runtime.Object
	// lock much be hold when reading ready or mergedConfiguration
	lock sync.RWMutex
}

func newPoller(get getFunc) *poller {
	return &poller{
		get:              get,
		interval:         defaultInterval,
		failureThreshold: defaultFailureThreshold,
	}
}

func (a *poller) notReady() {
	a.lock.Lock()
	defer a.lock.Unlock()
	a.ready = false
}

func (a *poller) configuration() (runtime.Object, error) {
	a.lock.RLock()
	defer a.lock.RUnlock()
	if !a.ready {
		return nil, fmt.Errorf("configuration is not ready")
	}
	return a.mergedConfiguration, nil
}

func (a *poller) setConfigurationAndReady(value runtime.Object) {
	a.lock.Lock()
	defer a.lock.Unlock()
	a.mergedConfiguration = value
	a.ready = true
}

func (a *poller) Run(stopCh <-chan struct{}) {
	go wait.Until(a.sync, a.interval, stopCh)
}

func (a *poller) sync() {
	configuration, err := a.get()
	if err != nil {
		a.failures++
		if a.failures >= a.failureThreshold {
			a.notReady()
		}
		return
	}
	a.failures = 0
	a.setConfigurationAndReady(configuration)
}
