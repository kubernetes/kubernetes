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

package server

import (
	"fmt"
	"net/http"
	"sync"
	"time"

	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/utils/clock"
)

// healthMux is an interface describing the methods InstallHandler requires.
type healthMux interface {
	Handle(pattern string, handler http.Handler)
}

type healthCheckRegistry struct {
	path            string
	lock            sync.Mutex
	checks          []healthz.HealthChecker
	checksInstalled bool
	clock           clock.Clock
}

func (reg *healthCheckRegistry) addHealthChecks(checks ...healthz.HealthChecker) error {
	return reg.addDelayedHealthChecks(0, checks...)
}

func (reg *healthCheckRegistry) addDelayedHealthChecks(delay time.Duration, checks ...healthz.HealthChecker) error {
	if delay > 0 && reg.clock == nil {
		return fmt.Errorf("nil clock in healthCheckRegistry for %s endpoint", reg.path)
	}
	reg.lock.Lock()
	defer reg.lock.Unlock()
	if reg.checksInstalled {
		return fmt.Errorf("unable to add because the %s endpoint has already been created", reg.path)
	}
	if delay > 0 {
		for _, check := range checks {
			reg.checks = append(reg.checks, delayedHealthCheck(check, reg.clock, delay))
		}
	} else {
		reg.checks = append(reg.checks, checks...)
	}
	return nil
}

func (reg *healthCheckRegistry) installHandler(mux healthMux) {
	reg.installHandlerWithHealthyFunc(mux, nil)
}

func (reg *healthCheckRegistry) installHandlerWithHealthyFunc(mux healthMux, firstTimeHealthy func()) {
	reg.lock.Lock()
	defer reg.lock.Unlock()
	reg.checksInstalled = true
	healthz.InstallPathHandlerWithHealthyFunc(mux, reg.path, firstTimeHealthy, reg.checks...)
}

// AddHealthChecks adds HealthCheck(s) to health endpoints (healthz, livez, readyz) but
// configures the liveness grace period to be zero, which means we expect this health check
// to immediately indicate that the apiserver is unhealthy.
func (s *GenericAPIServer) AddHealthChecks(checks ...healthz.HealthChecker) error {
	// we opt for a delay of zero here, because this entrypoint adds generic health checks
	// and not health checks which are specifically related to kube-apiserver boot-sequences.
	return s.addHealthChecks(0, checks...)
}

// AddBootSequenceHealthChecks adds health checks to the old healthz endpoint (for backwards compatibility reasons)
// as well as livez and readyz. The livez grace period is defined by the value of the
// command-line flag --livez-grace-period; before the grace period elapses, the livez health checks
// will default to healthy. One may want to set a grace period in order to prevent the kubelet from restarting
// the kube-apiserver due to long-ish boot sequences. Readyz health checks, on the other hand, have no grace period,
// since readyz should fail until boot fully completes.
func (s *GenericAPIServer) AddBootSequenceHealthChecks(checks ...healthz.HealthChecker) error {
	return s.addHealthChecks(s.livezGracePeriod, checks...)
}

// addHealthChecks adds health checks to healthz, livez, and readyz. The delay passed in will set
// a corresponding grace period on livez.
func (s *GenericAPIServer) addHealthChecks(livezGracePeriod time.Duration, checks ...healthz.HealthChecker) error {
	if err := s.healthzRegistry.addHealthChecks(checks...); err != nil {
		return err
	}
	if err := s.livezRegistry.addDelayedHealthChecks(livezGracePeriod, checks...); err != nil {
		return err
	}
	return s.readyzRegistry.addHealthChecks(checks...)
}

// AddReadyzChecks allows you to add a HealthCheck to readyz.
func (s *GenericAPIServer) AddReadyzChecks(checks ...healthz.HealthChecker) error {
	return s.readyzRegistry.addHealthChecks(checks...)
}

// AddLivezChecks allows you to add a HealthCheck to livez.
func (s *GenericAPIServer) AddLivezChecks(delay time.Duration, checks ...healthz.HealthChecker) error {
	return s.livezRegistry.addDelayedHealthChecks(delay, checks...)
}

// addReadyzShutdownCheck is a convenience function for adding a readyz shutdown check, so
// that we can register that the api-server is no longer ready while we attempt to gracefully
// shutdown.
func (s *GenericAPIServer) addReadyzShutdownCheck(stopCh <-chan struct{}) error {
	return s.AddReadyzChecks(healthz.NewShutdownHealthz(stopCh))
}

// installHealthz creates the healthz endpoint for this server
func (s *GenericAPIServer) installHealthz() {
	s.healthzRegistry.installHandler(s.Handler.NonGoRestfulMux)
}

// installReadyz creates the readyz endpoint for this server.
func (s *GenericAPIServer) installReadyz() {
	s.readyzRegistry.installHandlerWithHealthyFunc(s.Handler.NonGoRestfulMux, func() {
		// note: installHandlerWithHealthyFunc guarantees that this is called only once
		s.lifecycleSignals.HasBeenReady.Signal()
	})
}

// installLivez creates the livez endpoint for this server.
func (s *GenericAPIServer) installLivez() {
	s.livezRegistry.installHandler(s.Handler.NonGoRestfulMux)
}

// delayedHealthCheck wraps a health check which will not fail until the explicitly defined delay has elapsed. This
// is intended for use primarily for livez health checks.
func delayedHealthCheck(check healthz.HealthChecker, clock clock.Clock, delay time.Duration) healthz.HealthChecker {
	return delayedLivezCheck{
		check,
		clock.Now().Add(delay),
		clock,
	}
}

type delayedLivezCheck struct {
	check      healthz.HealthChecker
	startCheck time.Time
	clock      clock.Clock
}

func (c delayedLivezCheck) Name() string {
	return c.check.Name()
}

func (c delayedLivezCheck) Check(req *http.Request) error {
	if c.clock.Now().After(c.startCheck) {
		return c.check.Check(req)
	}
	return nil
}
