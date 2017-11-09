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

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/endpoints"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
)

// masterCountEndpointReconciler reconciles endpoints based on a specified expected number of
// masters. masterCountEndpointReconciler implements EndpointReconciler.
type masterCountEndpointReconciler struct {
	masterCount           int
	endpointClient        coreclient.EndpointsGetter
	stopReconcilingCalled bool
	reconcilingLock       sync.Mutex
}

// NewMasterCountEndpointReconciler creates a new EndpointReconciler that reconciles based on a
// specified expected number of masters.
func NewMasterCountEndpointReconciler(masterCount int, endpointClient coreclient.EndpointsGetter) EndpointReconciler {
	return &masterCountEndpointReconciler{
		masterCount:    masterCount,
		endpointClient: endpointClient,
	}
}

// ReconcileEndpoints sets the endpoints for the given apiserver service (ro or rw).
// ReconcileEndpoints expects that the endpoints objects it manages will all be
// managed only by ReconcileEndpoints; therefore, to understand this, you need only
// understand the requirements and the body of this function.
//
// Requirements:
//  * All apiservers MUST use the same ports for their {rw, ro} services.
//  * All apiservers MUST use ReconcileEndpoints and only ReconcileEndpoints to manage the
//      endpoints for their {rw, ro} services.
//  * All apiservers MUST know and agree on the number of apiservers expected
//      to be running (c.masterCount).
//  * ReconcileEndpoints is called periodically from all apiservers.
func (r *masterCountEndpointReconciler) ReconcileEndpoints(serviceName string, ip net.IP, endpointPorts []api.EndpointPort, reconcilePorts bool) error {
	r.reconcilingLock.Lock()
	defer r.reconcilingLock.Unlock()

	if r.stopReconcilingCalled {
		return nil
	}

	e, err := r.endpointClient.Endpoints(metav1.NamespaceDefault).Get(serviceName, metav1.GetOptions{})
	if err != nil {
		e = &api.Endpoints{
			ObjectMeta: metav1.ObjectMeta{
				Name:      serviceName,
				Namespace: metav1.NamespaceDefault,
			},
		}
	}
	if errors.IsNotFound(err) {
		// Simply create non-existing endpoints for the service.
		e.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: ip.String()}},
			Ports:     endpointPorts,
		}}
		_, err = r.endpointClient.Endpoints(metav1.NamespaceDefault).Create(e)
		return err
	}

	// First, determine if the endpoint is in the format we expect (one
	// subset, ports matching endpointPorts, N IP addresses).
	formatCorrect, ipCorrect, portsCorrect := checkEndpointSubsetFormat(e, ip.String(), endpointPorts, r.masterCount, reconcilePorts)
	if !formatCorrect {
		// Something is egregiously wrong, just re-make the endpoints record.
		e.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: ip.String()}},
			Ports:     endpointPorts,
		}}
		glog.Warningf("Resetting endpoints for master service %q to %#v", serviceName, e)
		_, err = r.endpointClient.Endpoints(metav1.NamespaceDefault).Update(e)
		return err
	}
	if ipCorrect && portsCorrect {
		return nil
	}
	if !ipCorrect {
		// We *always* add our own IP address.
		e.Subsets[0].Addresses = append(e.Subsets[0].Addresses, api.EndpointAddress{IP: ip.String()})

		// Lexicographic order is retained by this step.
		e.Subsets = endpoints.RepackSubsets(e.Subsets)

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
	glog.Warningf("Resetting endpoints for master service %q to %v", serviceName, e)
	_, err = r.endpointClient.Endpoints(metav1.NamespaceDefault).Update(e)
	return err
}

func (r *masterCountEndpointReconciler) StopReconciling(serviceName string, ip net.IP, endpointPorts []api.EndpointPort) error {
	r.reconcilingLock.Lock()
	defer r.reconcilingLock.Unlock()
	r.stopReconcilingCalled = true

	e, err := r.endpointClient.Endpoints(metav1.NamespaceDefault).Get(serviceName, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			// Endpoint doesn't exist
			return nil
		}
		return err
	}

	// Remove our IP from the list of addresses
	new := []api.EndpointAddress{}
	for _, addr := range e.Subsets[0].Addresses {
		if addr.IP != ip.String() {
			new = append(new, addr)
		}
	}
	e.Subsets[0].Addresses = new
	e.Subsets = endpoints.RepackSubsets(e.Subsets)
	err = retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		_, err := r.endpointClient.Endpoints(metav1.NamespaceDefault).Update(e)
		return err
	})
	return err
}

// Determine if the endpoint is in the format ReconcileEndpoints expects.
//
// Return values:
// * formatCorrect is true if exactly one subset is found.
// * ipCorrect is true when current master's IP is found and the number
//     of addresses is less than or equal to the master count.
// * portsCorrect is true when endpoint ports exactly match provided ports.
//     portsCorrect is only evaluated when reconcilePorts is set to true.
func checkEndpointSubsetFormat(e *api.Endpoints, ip string, ports []api.EndpointPort, count int, reconcilePorts bool) (formatCorrect bool, ipCorrect bool, portsCorrect bool) {
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

// GetMasterServiceUpdateIfNeeded sets service attributes for the
//     given apiserver service.
// * GetMasterServiceUpdateIfNeeded expects that the service object it
//     manages will be managed only by GetMasterServiceUpdateIfNeeded;
//     therefore, to understand this, you need only understand the
//     requirements and the body of this function.
// * GetMasterServiceUpdateIfNeeded ensures that the correct ports are
//     are set.
//
// Requirements:
// * All apiservers MUST use GetMasterServiceUpdateIfNeeded and only
//     GetMasterServiceUpdateIfNeeded to manage service attributes
// * updateMasterService is called periodically from all apiservers.
func GetMasterServiceUpdateIfNeeded(svc *api.Service, servicePorts []api.ServicePort, serviceType api.ServiceType) (s *api.Service, updated bool) {
	// Determine if the service is in the format we expect
	// (servicePorts are present and service type matches)
	formatCorrect := checkServiceFormat(svc, servicePorts, serviceType)
	if formatCorrect {
		return svc, false
	}
	svc.Spec.Ports = servicePorts
	svc.Spec.Type = serviceType
	return svc, true
}

// Determine if the service is in the correct format
// GetMasterServiceUpdateIfNeeded expects (servicePorts are correct
// and service type matches).
func checkServiceFormat(s *api.Service, ports []api.ServicePort, serviceType api.ServiceType) (formatCorrect bool) {
	if s.Spec.Type != serviceType {
		return false
	}
	if len(ports) != len(s.Spec.Ports) {
		return false
	}
	for i, port := range ports {
		if port != s.Spec.Ports[i] {
			return false
		}
	}
	return true
}
