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
	defaultInterval             = 1 * time.Second
	defaultFailureThreshold     = 5
	defaultBootstrapRetries     = 5
	defaultBootstrapGraceperiod = 5 * time.Second
)

var (
	ErrNotReady = fmt.Errorf("configuration is not ready")
	ErrDisabled = fmt.Errorf("disabled")
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
	// read-only
	interval time.Duration
	// if the number of consecutive read failure equals or exceeds the failureThreshold , the
	// configuration is regarded as not ready.
	// read-only
	failureThreshold int
	// number of consecutive failures so far.
	failures int
	// If the poller has passed the bootstrap phase. The poller is considered
	// bootstrapped either bootstrapGracePeriod after the first call of
	// configuration(), or when setConfigurationAndReady() is called, whichever
	// comes first.
	bootstrapped bool
	// configuration() retries bootstrapRetries times if poller is not bootstrapped
	// read-only
	bootstrapRetries int
	// Grace period for bootstrapping
	// read-only
	bootstrapGracePeriod time.Duration
	once                 sync.Once
	// if the configuration is regarded as ready.
	ready               bool
	mergedConfiguration runtime.Object
	lastErr             error
	// lock must be hold when reading/writing the data fields of poller.
	lock sync.RWMutex
}

func newPoller(get getFunc) *poller {
	p := poller{
		get:                  get,
		interval:             defaultInterval,
		failureThreshold:     defaultFailureThreshold,
		bootstrapRetries:     defaultBootstrapRetries,
		bootstrapGracePeriod: defaultBootstrapGraceperiod,
	}
	return &p
}

func (a *poller) lastError(err error) {
	a.lock.Lock()
	defer a.lock.Unlock()
	a.lastErr = err
}

func (a *poller) notReady() {
	a.lock.Lock()
	defer a.lock.Unlock()
	a.ready = false
}

func (a *poller) bootstrapping() {
	// bootstrapGracePeriod is read-only, so no lock is required
	timer := time.NewTimer(a.bootstrapGracePeriod)
	go func() {
		defer timer.Stop()
		<-timer.C
		a.lock.Lock()
		defer a.lock.Unlock()
		a.bootstrapped = true
	}()
}

// If the poller is not bootstrapped yet, the configuration() gets a few chances
// to retry. This hides transient failures during system startup.
func (a *poller) configuration() (runtime.Object, error) {
	a.once.Do(a.bootstrapping)
	a.lock.RLock()
	defer a.lock.RUnlock()
	retries := 1
	if !a.bootstrapped {
		retries = a.bootstrapRetries
	}
	for count := 0; count < retries; count++ {
		if count > 0 {
			a.lock.RUnlock()
			time.Sleep(a.interval)
			a.lock.RLock()
		}
		if a.ready {
			return a.mergedConfiguration, nil
		}
	}
	if a.lastErr != nil {
		return nil, a.lastErr
	}
	return nil, ErrNotReady
}

func (a *poller) setConfigurationAndReady(value runtime.Object) {
	a.lock.Lock()
	defer a.lock.Unlock()
	a.bootstrapped = true
	a.mergedConfiguration = value
	a.ready = true
	a.lastErr = nil
}

func (a *poller) Run(stopCh <-chan struct{}) {
	go wait.Until(a.sync, a.interval, stopCh)
}

func (a *poller) sync() {
	configuration, err := a.get()
	if err != nil {
		a.failures++
		a.lastError(err)
		if a.failures >= a.failureThreshold {
			a.notReady()
		}
		return
	}
	a.failures = 0
	a.setConfigurationAndReady(configuration)
}
