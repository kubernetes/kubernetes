/*
Copyright 2021 The Kubernetes Authors.

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

package controller

import (
	"net/http"

	"k8s.io/controller-manager/pkg/healthz"
)

// Interface defines the base of a controller managed by a controller manager
type Interface interface {
	// Name returns the canonical name of the controller.
	Name() string
}

// Debuggable defines a controller that allows the controller manager
// to expose a debugging handler for the controller
//
// If a controller implements Debuggable, and the returned handler is
// not nil, the controller manager can mount the handler during startup.
type Debuggable interface {
	// DebuggingHandler returns a Handler that expose debugging information
	// for the controller, or nil if a debugging handler is not desired.
	//
	// The handler will be accessible at "/debug/controllers/{controllerName}/".
	DebuggingHandler() http.Handler
}

// HealthCheckable defines a controller that allows the controller manager
// to include it in the health checks.
//
// If a controller implements HealthCheckable, and the returned check
// is not nil, the controller manager can expose the check to the
// /healthz endpoint.
type HealthCheckable interface {
	// HealthChecker returns a UnnamedHealthChecker that the controller manager
	// can choose to mount on the /healthz endpoint, or nil if no custom
	// health check is desired.
	HealthChecker() healthz.UnnamedHealthChecker
}
