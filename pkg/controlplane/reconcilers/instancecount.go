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

// Package reconcilers master count based reconciler
package reconcilers

import (
	"net"
	"sync"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

// masterCountEndpointReconciler reconciles endpoints based on a specified expected number of
// masters. masterCountEndpointReconciler implements EndpointReconciler.
type masterCountEndpointReconciler struct {
	masterCount           int
	epAdapter             *EndpointsAdapter
	stopReconcilingCalled bool
	reconcilingLock       sync.Mutex
}

// NewMasterCountEndpointReconciler creates a new EndpointReconciler that reconciles based on a
// specified expected number of masters.
func NewMasterCountEndpointReconciler(masterCount int, epAdapter *EndpointsAdapter) EndpointReconciler {
	return &masterCountEndpointReconciler{
		masterCount: masterCount,
		epAdapter:   epAdapter,
	}
}

// ReconcileEndpoints sets the endpoints for the given apiserver service (ro or rw).
// ReconcileEndpoints expects that the endpoints objects it manages will all be
// managed only by ReconcileEndpoints; therefore, to understand this, you need only
// understand the requirements and the body of this function.
//
// Requirements:
//   - All apiservers MUST use the same ports for their {rw, ro} services.
//   - All apiservers MUST use ReconcileEndpoints and only ReconcileEndpoints to manage the
//     endpoints for their {rw, ro} services.
//   - All apiservers MUST know and agree on the number of apiservers expected
//     to be running (c.masterCount).
//   - ReconcileEndpoints is called periodically from all apiservers.
func (r *masterCountEndpointReconciler) ReconcileEndpoints(ip net.IP, endpointPorts []corev1.EndpointPort, reconcilePorts bool) error {
	r.reconcilingLock.Lock()
	defer r.reconcilingLock.Unlock()

	if r.stopReconcilingCalled {
		return nil
	}

	endpointIPs, err := r.epAdapter.Get()
	if err != nil {
		return err
	}

	// We *always* add our own IP address.
	ipStr := ip.String()
	endpointIPs.Insert(ipStr)

	// If we want M IPs and have N where N>M, then remove the (N-M) IPs immediately
	// following our own in the list (wrapping back around to the start if necessary).
	// Given the requirements stated at the top of this function, this should cause
	// the list of IP addresses to become eventually correct.
	if len(endpointIPs) > r.masterCount {
		sortedIPs := sets.List(endpointIPs)
		for i := range sortedIPs {
			if sortedIPs[i] == ipStr {
				for len(endpointIPs) > r.masterCount {
					// wrap around if necessary.
					remove := (i + 1) % len(sortedIPs)
					endpointIPs.Delete(sortedIPs[remove])
				}
				break
			}
		}
	}

	return r.epAdapter.Sync(endpointIPs, endpointPorts, reconcilePorts)
}

func (r *masterCountEndpointReconciler) RemoveEndpoints(ip net.IP, endpointPorts []corev1.EndpointPort) error {
	r.reconcilingLock.Lock()
	defer r.reconcilingLock.Unlock()

	endpointIPs, err := r.epAdapter.Get()
	if err != nil {
		return err
	}

	ipStr := ip.String()
	if len(endpointIPs) == 0 || !endpointIPs.Has(ipStr) {
		// Nothing to do
		return nil
	}

	endpointIPs.Delete(ipStr)
	return r.epAdapter.Sync(endpointIPs, endpointPorts, false)
}

func (r *masterCountEndpointReconciler) StopReconciling() {
	r.reconcilingLock.Lock()
	defer r.reconcilingLock.Unlock()
	r.stopReconcilingCalled = true
}

func (r *masterCountEndpointReconciler) Destroy() {
}
