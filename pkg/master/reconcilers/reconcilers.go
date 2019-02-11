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

// Package reconcilers Endpoint Reconcilers for the apiserver
package reconcilers

import (
	corev1 "k8s.io/api/core/v1"
	"net"
)

// EndpointReconciler knows how to reconcile the endpoints for the apiserver service.
type EndpointReconciler interface {
	// ReconcileEndpoints sets the endpoints for the given apiserver service (ro or rw).
	// ReconcileEndpoints expects that the endpoints objects it manages will all be
	// managed only by ReconcileEndpoints; therefore, to understand this, you need only
	// understand the requirements.
	//
	// Requirements:
	//  * All apiservers MUST use the same ports for their {rw, ro} services.
	//  * All apiservers MUST use ReconcileEndpoints and only ReconcileEndpoints to manage the
	//      endpoints for their {rw, ro} services.
	//  * ReconcileEndpoints is called periodically from all apiservers.
	ReconcileEndpoints(serviceName string, ip net.IP, endpointPorts []corev1.EndpointPort, reconcilePorts bool) error
	StopReconciling(serviceName string, ip net.IP, endpointPorts []corev1.EndpointPort) error
}

// Type the reconciler type
type Type string

const (
	// MasterCountReconcilerType will select the original reconciler
	MasterCountReconcilerType Type = "master-count"
	// LeaseEndpointReconcilerType will select a storage based reconciler
	LeaseEndpointReconcilerType = "lease"
	// NoneEndpointReconcilerType will turn off the endpoint reconciler
	NoneEndpointReconcilerType = "none"
)

// Types an array of reconciler types
type Types []Type

// AllTypes export all reconcilers
var AllTypes = Types{
	MasterCountReconcilerType,
	LeaseEndpointReconcilerType,
	NoneEndpointReconcilerType,
}

// Names returns a slice of all the reconciler names
func (t Types) Names() []string {
	strs := make([]string, len(t))
	for i, v := range t {
		strs[i] = string(v)
	}
	return strs
}
