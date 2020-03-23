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
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apiserver/pkg/server/healthz"
)

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
	s.healthzLock.Lock()
	defer s.healthzLock.Unlock()
	if s.healthzChecksInstalled {
		return fmt.Errorf("unable to add because the healthz endpoint has already been created")
	}
	s.healthzChecks = append(s.healthzChecks, checks...)
	return s.addLivezChecks(livezGracePeriod, checks...)
}

// addReadyzChecks allows you to add a HealthCheck to readyz.
func (s *GenericAPIServer) addReadyzChecks(checks ...healthz.HealthChecker) error {
	s.readyzLock.Lock()
	defer s.readyzLock.Unlock()
	if s.readyzChecksInstalled {
		return fmt.Errorf("unable to add because the readyz endpoint has already been created")
	}
	s.readyzChecks = append(s.readyzChecks, checks...)
	return nil
}

// addLivezChecks allows you to add a HealthCheck to livez. It will also automatically add a check to readyz,
// since we want to avoid being ready when we are not live.
func (s *GenericAPIServer) addLivezChecks(delay time.Duration, checks ...healthz.HealthChecker) error {
	s.livezLock.Lock()
	defer s.livezLock.Unlock()
	if s.livezChecksInstalled {
		return fmt.Errorf("unable to add because the livez endpoint has already been created")
	}
	for _, check := range checks {
		s.livezChecks = append(s.livezChecks, delayedHealthCheck(check, s.livezClock, delay))
	}
	return s.addReadyzChecks(checks...)
}

// addReadyzShutdownCheck is a convenience function for adding a readyz shutdown check, so
// that we can register that the api-server is no longer ready while we attempt to gracefully
// shutdown.
func (s *GenericAPIServer) addReadyzShutdownCheck(stopCh <-chan struct{}) error {
	return s.addReadyzChecks(shutdownCheck{stopCh})
}

// installHealthz creates the healthz endpoint for this server
func (s *GenericAPIServer) installHealthz() {
	s.healthzLock.Lock()
	defer s.healthzLock.Unlock()
	s.healthzChecksInstalled = true
	healthz.InstallHandler(s.Handler.NonGoRestfulMux, s.healthzChecks...)
}

// installReadyz creates the readyz endpoint for this server.
func (s *GenericAPIServer) installReadyz() {
	s.readyzLock.Lock()
	defer s.readyzLock.Unlock()
	s.readyzChecksInstalled = true
	healthz.InstallReadyzHandler(s.Handler.NonGoRestfulMux, s.readyzChecks...)
}

// installLivez creates the livez endpoint for this server.
func (s *GenericAPIServer) installLivez() {
	s.livezLock.Lock()
	defer s.livezLock.Unlock()
	s.livezChecksInstalled = true
	healthz.InstallLivezHandler(s.Handler.NonGoRestfulMux, s.livezChecks...)
}

// shutdownCheck fails if the embedded channel is closed. This is intended to allow for graceful shutdown sequences
// for the apiserver.
type shutdownCheck struct {
	StopCh <-chan struct{}
}

func (shutdownCheck) Name() string {
	return "shutdown"
}

func (c shutdownCheck) Check(req *http.Request) error {
	select {
	case <-c.StopCh:
		return fmt.Errorf("process is shutting down")
	default:
	}
	return nil
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
