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

// Package reconcilers a noop based reconciler
package reconcilers

import (
	"net"

	"k8s.io/kubernetes/pkg/api"
)

// NoneEndpointReconciler allows for the endpoint reconciler to be disabled
type noneEndpointReconciler struct{}

// NewNoneEndpointReconciler creates a new EndpointReconciler that reconciles based on a
// nothing. It is a no-op.
func NewNoneEndpointReconciler() EndpointReconciler {
	return &noneEndpointReconciler{}
}

// ReconcileEndpoints noop reconcile
func (r *noneEndpointReconciler) ReconcileEndpoints(serviceName string, ip net.IP, endpointPorts []api.EndpointPort, reconcilePorts bool) error {
	return nil
}

// StopReconciling noop reconcile
func (r *noneEndpointReconciler) StopReconciling(serviceName string, ip net.IP, endpointPorts []api.EndpointPort) error {
	return nil
}
