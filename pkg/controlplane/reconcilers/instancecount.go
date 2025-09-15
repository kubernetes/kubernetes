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
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"
	endpointsv1 "k8s.io/kubernetes/pkg/api/v1/endpoints"
)

// masterCountEndpointReconciler reconciles endpoints based on a specified expected number of
// masters. masterCountEndpointReconciler implements EndpointReconciler.
type masterCountEndpointReconciler struct {
	masterCount           int
	epAdapter             EndpointsAdapter
	stopReconcilingCalled bool
	reconcilingLock       sync.Mutex
}

// NewMasterCountEndpointReconciler creates a new EndpointReconciler that reconciles based on a
// specified expected number of masters.
func NewMasterCountEndpointReconciler(masterCount int, epAdapter EndpointsAdapter) EndpointReconciler {
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
func (r *masterCountEndpointReconciler) ReconcileEndpoints(serviceName string, ip net.IP, endpointPorts []corev1.EndpointPort, reconcilePorts bool) error {
	r.reconcilingLock.Lock()
	defer r.reconcilingLock.Unlock()

	if r.stopReconcilingCalled {
		return nil
	}

	e, err := r.epAdapter.Get(metav1.NamespaceDefault, serviceName, metav1.GetOptions{})
	if err != nil {
		e = &corev1.Endpoints{
			ObjectMeta: metav1.ObjectMeta{
				Name:      serviceName,
				Namespace: metav1.NamespaceDefault,
			},
		}
	}

	// Don't use the EndpointSliceMirroring controller to mirror this to
	// EndpointSlices. This may change in the future.
	skipMirrorChanged := setSkipMirrorTrue(e)

	if errors.IsNotFound(err) {
		// Simply create non-existing endpoints for the service.
		e.Subsets = []corev1.EndpointSubset{{
			Addresses: []corev1.EndpointAddress{{IP: ip.String()}},
			Ports:     endpointPorts,
		}}
		_, err = r.epAdapter.Create(metav1.NamespaceDefault, e)
		return err
	}

	// First, determine if the endpoint is in the format we expect (one
	// subset, ports matching endpointPorts, N IP addresses).
	formatCorrect, ipCorrect, portsCorrect := checkEndpointSubsetFormat(e, ip.String(), endpointPorts, r.masterCount, reconcilePorts)
	if !formatCorrect {
		// Something is egregiously wrong, just re-make the endpoints record.
		e.Subsets = []corev1.EndpointSubset{{
			Addresses: []corev1.EndpointAddress{{IP: ip.String()}},
			Ports:     endpointPorts,
		}}
		klog.Warningf("Resetting endpoints for master service %q to %#v", serviceName, e)
		_, err = r.epAdapter.Update(metav1.NamespaceDefault, e)
		return err
	}

	if !skipMirrorChanged && ipCorrect && portsCorrect {
		return r.epAdapter.EnsureEndpointSliceFromEndpoints(metav1.NamespaceDefault, e)
	}
	if !ipCorrect {
		// We *always* add our own IP address.
		e.Subsets[0].Addresses = append(e.Subsets[0].Addresses, corev1.EndpointAddress{IP: ip.String()})

		// Lexicographic order is retained by this step.
		e.Subsets = endpointsv1.RepackSubsets(e.Subsets)

		// If too many IP addresses, remove the ones lexicographically after our
		// own IP address.  Given the requirements stated at the top of
		// this function, this should cause the list of IP addresses to
		// become eventually correct.
		if addrs := &e.Subsets[0].Addresses; len(*addrs) > r.masterCount {
			// addrs is a pointer because we're going to mutate it.
			for i, addr := range *addrs {
				if addr.IP == ip.String() {
					for len(*addrs) > r.masterCount {
						// wrap around if necessary.
						remove := (i + 1) % len(*addrs)
						*addrs = append((*addrs)[:remove], (*addrs)[remove+1:]...)
					}
					break
				}
			}
		}
	}
	if !portsCorrect {
		// Reset ports.
		e.Subsets[0].Ports = endpointPorts
	}
	klog.Warningf("Resetting endpoints for master service %q to %v", serviceName, e)
	_, err = r.epAdapter.Update(metav1.NamespaceDefault, e)
	return err
}

func (r *masterCountEndpointReconciler) RemoveEndpoints(serviceName string, ip net.IP, endpointPorts []corev1.EndpointPort) error {
	r.reconcilingLock.Lock()
	defer r.reconcilingLock.Unlock()

	e, err := r.epAdapter.Get(metav1.NamespaceDefault, serviceName, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			// Endpoint doesn't exist
			return nil
		}
		return err
	}

	if len(e.Subsets) == 0 {
		// no action is needed to remove the endpoint
		return nil
	}
	// Remove our IP from the list of addresses
	new := []corev1.EndpointAddress{}
	for _, addr := range e.Subsets[0].Addresses {
		if addr.IP != ip.String() {
			new = append(new, addr)
		}
	}
	e.Subsets[0].Addresses = new
	e.Subsets = endpointsv1.RepackSubsets(e.Subsets)
	err = retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		_, err := r.epAdapter.Update(metav1.NamespaceDefault, e)
		return err
	})
	return err
}

func (r *masterCountEndpointReconciler) StopReconciling() {
	r.reconcilingLock.Lock()
	defer r.reconcilingLock.Unlock()
	r.stopReconcilingCalled = true
}

func (r *masterCountEndpointReconciler) Destroy() {
}

// Determine if the endpoint is in the format ReconcileEndpoints expects.
//
// Return values:
//   - formatCorrect is true if exactly one subset is found.
//   - ipCorrect is true when current master's IP is found and the number
//     of addresses is less than or equal to the master count.
//   - portsCorrect is true when endpoint ports exactly match provided ports.
//     portsCorrect is only evaluated when reconcilePorts is set to true.
func checkEndpointSubsetFormat(e *corev1.Endpoints, ip string, ports []corev1.EndpointPort, count int, reconcilePorts bool) (formatCorrect bool, ipCorrect bool, portsCorrect bool) {
	if len(e.Subsets) != 1 {
		return false, false, false
	}
	sub := &e.Subsets[0]
	portsCorrect = true
	if reconcilePorts {
		if len(sub.Ports) != len(ports) {
			portsCorrect = false
		}
		for i, port := range ports {
			if len(sub.Ports) <= i || port != sub.Ports[i] {
				portsCorrect = false
				break
			}
		}
	}
	for _, addr := range sub.Addresses {
		if addr.IP == ip {
			ipCorrect = len(sub.Addresses) <= count
			break
		}
	}
	return true, ipCorrect, portsCorrect
}
