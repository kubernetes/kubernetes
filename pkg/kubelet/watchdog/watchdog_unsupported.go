//go:build !linux
// +build !linux

/*
Copyright 2024 The Kubernetes Authors.

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

package watchdog

import "k8s.io/apiserver/pkg/server/healthz"

type healthCheckerUnsupported struct{}

var _ HealthChecker = &healthCheckerUnsupported{}

type Option func(*healthCheckerUnsupported)

func WithExtendedCheckers(checkers []healthz.HealthChecker) Option {
	return nil
}

// NewHealthChecker creates a fake one here
func NewHealthChecker(_ syncLoopHealthChecker, _ ...Option) (HealthChecker, error) {
	return &healthCheckerUnsupported{}, nil
}

func (ow *healthCheckerUnsupported) Start() {
	return
}
