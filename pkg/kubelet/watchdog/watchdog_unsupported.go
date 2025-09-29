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

import (
	"context"

	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/klog/v2"
)

type healthCheckerUnsupported struct{}

var _ HealthChecker = &healthCheckerUnsupported{}

type Option func(*healthCheckerUnsupported)

func WithExtendedCheckers([]healthz.HealthChecker) Option {
	return nil
}

// NewHealthChecker creates a fake one here
func NewHealthChecker(klog.Logger, ...Option) (HealthChecker, error) {
	return &healthCheckerUnsupported{}, nil
}

func (hc *healthCheckerUnsupported) SetHealthCheckers(syncLoop syncLoopHealthChecker, checkers []healthz.HealthChecker) {
}

func (ow *healthCheckerUnsupported) Start(context.Context) {
	return
}
