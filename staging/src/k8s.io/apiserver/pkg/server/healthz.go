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

// AddHealthzCheck adds HealthzCheck(s) to both healthz and readyz. All healthz checks
// are automatically added to readyz, since we want to avoid the situation where the
// apiserver is ready but not live.
func (s *GenericAPIServer) AddHealthzChecks(checks ...healthz.HealthzChecker) error {
	return s.AddDelayedHealthzChecks(0, checks...)
}

// AddReadyzChecks allows you to add a HealthzCheck to readyz.
func (s *GenericAPIServer) AddReadyzChecks(checks ...healthz.HealthzChecker) error {
	s.readyzLock.Lock()
	defer s.readyzLock.Unlock()
	return s.addReadyzChecks(checks...)
}

// addReadyzChecks allows you to add a HealthzCheck to readyz.
// premise: readyzLock has been obtained
func (s *GenericAPIServer) addReadyzChecks(checks ...healthz.HealthzChecker) error {
	if s.readyzChecksInstalled {
		return fmt.Errorf("unable to add because the readyz endpoint has already been created")
	}

	s.readyzChecks = append(s.readyzChecks, checks...)
	return nil
}

// installHealthz creates the healthz endpoint for this server
func (s *GenericAPIServer) installHealthz() {
	s.healthzLock.Lock()
	defer s.healthzLock.Unlock()
	s.healthzChecksInstalled = true

	healthz.InstallHandler(s.Handler.NonGoRestfulMux, s.healthzChecks...)
}

// installReadyz creates the readyz endpoint for this server.
func (s *GenericAPIServer) installReadyz(stopCh <-chan struct{}) {
	s.readyzLock.Lock()
	defer s.readyzLock.Unlock()
	s.addReadyzChecks(shutdownCheck{stopCh})

	s.readyzChecksInstalled = true

	healthz.InstallReadyzHandler(s.Handler.NonGoRestfulMux, s.readyzChecks...)
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

// AddDelayedHealthzChecks adds a health check to both healthz and readyz. The delay parameter
// allows you to set the grace period for healthz checks, which will return healthy while
// grace period has not yet elapsed. One may want to set a grace period in order to prevent
// the kubelet from restarting the kube-apiserver due to long-ish boot sequences. Readyz health
// checks have no grace period, since we want readyz to fail while boot has not completed.
func (s *GenericAPIServer) AddDelayedHealthzChecks(delay time.Duration, checks ...healthz.HealthzChecker) error {
	s.healthzLock.Lock()
	defer s.healthzLock.Unlock()
	if s.healthzChecksInstalled {
		return fmt.Errorf("unable to add because the healthz endpoint has already been created")
	}
	for _, check := range checks {
		s.healthzChecks = append(s.healthzChecks, delayedHealthCheck(check, s.healthzClock, s.maxStartupSequenceDuration))
	}

	s.readyzLock.Lock()
	defer s.readyzLock.Unlock()
	return s.addReadyzChecks(checks...)
}

// delayedHealthCheck wraps a health check which will not fail until the explicitly defined delay has elapsed.
func delayedHealthCheck(check healthz.HealthzChecker, clock clock.Clock, delay time.Duration) healthz.HealthzChecker {
	return delayedHealthzCheck{
		check,
		clock.Now().Add(delay),
		clock,
	}
}

type delayedHealthzCheck struct {
	check      healthz.HealthzChecker
	startCheck time.Time
	clock      clock.Clock
}

func (c delayedHealthzCheck) Name() string {
	return c.check.Name()
}

func (c delayedHealthzCheck) Check(req *http.Request) error {
	if c.clock.Now().After(c.startCheck) {
		return c.check.Check(req)
	}
	return nil
}
