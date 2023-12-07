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

package reconcilers

/*
Original Source:
https://github.com/openshift/origin/blob/bb340c5dd5ff72718be86fb194dedc0faed7f4c7/pkg/cmd/server/election/lease_endpoint_reconciler.go
*/

import (
	"fmt"
	"net"
	"path"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/klog/v2"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kruntime "k8s.io/apimachinery/pkg/runtime"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	storagefactory "k8s.io/apiserver/pkg/storage/storagebackend/factory"
	endpointsv1 "k8s.io/kubernetes/pkg/api/v1/endpoints"
)

// Leases is an interface which assists in managing the set of active masters
type Leases interface {
	// ListLeases retrieves a list of the current master IPs
	ListLeases() ([]string, error)

	// UpdateLease adds or refreshes a master's lease
	UpdateLease(ip string) error

	// RemoveLease removes a master's lease
	RemoveLease(ip string) error

	// Destroy cleans up everything on shutdown.
	Destroy()
}

type storageLeases struct {
	storage   storage.Interface
	destroyFn func()
	baseKey   string
	leaseTime time.Duration
}

var _ Leases = &storageLeases{}

// ListLeases retrieves a list of the current master IPs from storage
func (s *storageLeases) ListLeases() ([]string, error) {
	ipInfoList := &corev1.EndpointsList{}
	storageOpts := storage.ListOptions{
		ResourceVersion:      "0",
		ResourceVersionMatch: metav1.ResourceVersionMatchNotOlderThan,
		Predicate:            storage.Everything,
		Recursive:            true,
	}
	if err := s.storage.GetList(apirequest.NewDefaultContext(), s.baseKey, storageOpts, ipInfoList); err != nil {
		return nil, err
	}

	ipList := make([]string, 0, len(ipInfoList.Items))
	for _, ip := range ipInfoList.Items {
		if len(ip.Subsets) > 0 && len(ip.Subsets[0].Addresses) > 0 && len(ip.Subsets[0].Addresses[0].IP) > 0 {
			ipList = append(ipList, ip.Subsets[0].Addresses[0].IP)
		}
	}

	klog.V(6).Infof("Current master IPs listed in storage are %v", ipList)

	return ipList, nil
}

// UpdateLease resets the TTL on a master IP in storage
// UpdateLease will create a new key if it doesn't exist.
func (s *storageLeases) UpdateLease(ip string) error {
	key := path.Join(s.baseKey, ip)
	return s.storage.GuaranteedUpdate(apirequest.NewDefaultContext(), key, &corev1.Endpoints{}, true, nil, func(input kruntime.Object, respMeta storage.ResponseMeta) (kruntime.Object, *uint64, error) {
		// just make sure we've got the right IP set, and then refresh the TTL
		existing := input.(*corev1.Endpoints)
		existing.Subsets = []corev1.EndpointSubset{
			{
				Addresses: []corev1.EndpointAddress{{IP: ip}},
			},
		}

		// leaseTime needs to be in seconds
		leaseTime := uint64(s.leaseTime / time.Second)

		// NB: GuaranteedUpdate does not perform the store operation unless
		// something changed between load and store (not including resource
		// version), meaning we can't refresh the TTL without actually
		// changing a field.
		existing.Generation++

		klog.V(6).Infof("Resetting TTL on master IP %q listed in storage to %v", ip, leaseTime)

		return existing, &leaseTime, nil
	}, nil)
}

// RemoveLease removes the lease on a master IP in storage
func (s *storageLeases) RemoveLease(ip string) error {
	key := path.Join(s.baseKey, ip)
	return s.storage.Delete(apirequest.NewDefaultContext(), key, &corev1.Endpoints{}, nil, rest.ValidateAllObjectFunc, nil)
}

func (s *storageLeases) Destroy() {
	s.destroyFn()
}

// NewLeases creates a new etcd-based Leases implementation.
func NewLeases(config *storagebackend.ConfigForResource, baseKey string, leaseTime time.Duration) (Leases, error) {
	// note that newFunc, newListFunc and resourcePrefix
	// can be left blank unless the storage.Watch method is used
	leaseStorage, destroyFn, err := storagefactory.Create(*config, nil, nil, "")
	if err != nil {
		return nil, fmt.Errorf("error creating storage factory: %v", err)
	}
	var once sync.Once
	return &storageLeases{
		storage:   leaseStorage,
		destroyFn: func() { once.Do(destroyFn) },
		baseKey:   baseKey,
		leaseTime: leaseTime,
	}, nil
}

type leaseEndpointReconciler struct {
	epAdapter             EndpointsAdapter
	masterLeases          Leases
	stopReconcilingCalled atomic.Bool
	reconcilingLock       sync.Mutex
}

// NewLeaseEndpointReconciler creates a new LeaseEndpoint reconciler
func NewLeaseEndpointReconciler(epAdapter EndpointsAdapter, masterLeases Leases) EndpointReconciler {
	return &leaseEndpointReconciler{
		epAdapter:    epAdapter,
		masterLeases: masterLeases,
	}
}

// ReconcileEndpoints lists keys in a special etcd directory.
// Each key is expected to have a TTL of R+n, where R is the refresh interval
// at which this function is called, and n is some small value.  If an
// apiserver goes down, it will fail to refresh its key's TTL and the key will
// expire. ReconcileEndpoints will notice that the endpoints object is
// different from the directory listing, and update the endpoints object
// accordingly.
func (r *leaseEndpointReconciler) ReconcileEndpoints(serviceName string, ip net.IP, endpointPorts []corev1.EndpointPort, reconcilePorts bool) error {
	// reconcile endpoints only if apiserver was not shutdown
	if r.stopReconcilingCalled.Load() {
		return nil
	}

	// Ensure that there will be no race condition with the RemoveEndpoints.
	r.reconcilingLock.Lock()
	defer r.reconcilingLock.Unlock()

	// Refresh the TTL on our key, independently of whether any error or
	// update conflict happens below. This makes sure that at least some of
	// the masters will add our endpoint.
	if err := r.masterLeases.UpdateLease(ip.String()); err != nil {
		return err
	}

	return r.doReconcile(serviceName, endpointPorts, reconcilePorts)
}

// doReconcile can be called from ReconcileEndpoints() or RemoveEndpoints().
// it is NOT SAFE to call it from multiple goroutines.
func (r *leaseEndpointReconciler) doReconcile(serviceName string, endpointPorts []corev1.EndpointPort, reconcilePorts bool) error {
	e, err := r.epAdapter.Get(corev1.NamespaceDefault, serviceName, metav1.GetOptions{})
	shouldCreate := false
	if err != nil {
		if !errors.IsNotFound(err) {
			return err
		}

		// there are no endpoints and we should stop reconciling
		if r.stopReconcilingCalled.Load() {
			return nil
		}

		shouldCreate = true
		e = &corev1.Endpoints{
			ObjectMeta: metav1.ObjectMeta{
				Name:      serviceName,
				Namespace: corev1.NamespaceDefault,
			},
		}
	}

	// ... and the list of master IP keys from etcd
	masterIPs, err := r.masterLeases.ListLeases()
	if err != nil {
		return err
	}

	// Since we just refreshed our own key, assume that zero endpoints
	// returned from storage indicates an issue or invalid state, and thus do
	// not update the endpoints list based on the result.
	// If the controller was ordered to stop and is this is the last apiserver
	// we keep going to remove our endpoint before shutting down.
	if !r.stopReconcilingCalled.Load() && len(masterIPs) == 0 {
		return fmt.Errorf("no API server IP addresses were listed in storage, refusing to erase all endpoints for the kubernetes Service")
	}

	// Don't use the EndpointSliceMirroring controller to mirror this to
	// EndpointSlices. This may change in the future.
	skipMirrorChanged := setSkipMirrorTrue(e)

	// Next, we compare the current list of endpoints with the list of master IP keys
	formatCorrect, ipCorrect, portsCorrect := checkEndpointSubsetFormatWithLease(e, masterIPs, endpointPorts, reconcilePorts)
	if !skipMirrorChanged && formatCorrect && ipCorrect && portsCorrect {
		return r.epAdapter.EnsureEndpointSliceFromEndpoints(corev1.NamespaceDefault, e)
	}

	if !formatCorrect {
		// Something is egregiously wrong, just re-make the endpoints record.
		e.Subsets = []corev1.EndpointSubset{{
			Addresses: []corev1.EndpointAddress{},
			Ports:     endpointPorts,
		}}
	}

	if !formatCorrect || !ipCorrect {
		// repopulate the addresses according to the expected IPs from etcd
		e.Subsets[0].Addresses = make([]corev1.EndpointAddress, len(masterIPs))
		for ind, ip := range masterIPs {
			e.Subsets[0].Addresses[ind] = corev1.EndpointAddress{IP: ip}
		}

		// Lexicographic order is retained by this step.
		e.Subsets = endpointsv1.RepackSubsets(e.Subsets)
	}

	if len(e.Subsets) != 0 && !portsCorrect {
		// Reset ports.
		e.Subsets[0].Ports = endpointPorts
	}

	klog.Warningf("Resetting endpoints for master service %q to %v", serviceName, masterIPs)
	if shouldCreate {
		if _, err = r.epAdapter.Create(corev1.NamespaceDefault, e); errors.IsAlreadyExists(err) {
			err = nil
		}
	} else {
		_, err = r.epAdapter.Update(corev1.NamespaceDefault, e)
	}
	return err
}

// checkEndpointSubsetFormatWithLease determines if the endpoint is in the
// format ReconcileEndpoints expects when the controller is using leases.
//
// Return values:
//   - formatCorrect is true if exactly one subset is found.
//   - ipsCorrect when the addresses in the endpoints match the expected addresses list
//   - portsCorrect is true when endpoint ports exactly match provided ports.
//     portsCorrect is only evaluated when reconcilePorts is set to true.
func checkEndpointSubsetFormatWithLease(e *corev1.Endpoints, expectedIPs []string, ports []corev1.EndpointPort, reconcilePorts bool) (formatCorrect bool, ipsCorrect bool, portsCorrect bool) {
	if len(e.Subsets) != 1 {
		return false, false, false
	}
	sub := &e.Subsets[0]
	portsCorrect = true
	if reconcilePorts {
		if len(sub.Ports) != len(ports) {
			portsCorrect = false
		} else {
			for i, port := range ports {
				if port != sub.Ports[i] {
					portsCorrect = false
					break
				}
			}
		}
	}

	ipsCorrect = true
	if len(sub.Addresses) != len(expectedIPs) {
		ipsCorrect = false
	} else {
		// check the actual content of the addresses
		// present addrs is used as a set (the keys) and to indicate if a
		// value was already found (the values)
		presentAddrs := make(map[string]bool, len(expectedIPs))
		for _, ip := range expectedIPs {
			presentAddrs[ip] = false
		}

		// uniqueness is assumed amongst all Addresses.
		for _, addr := range sub.Addresses {
			if alreadySeen, ok := presentAddrs[addr.IP]; alreadySeen || !ok {
				ipsCorrect = false
				break
			}

			presentAddrs[addr.IP] = true
		}
	}

	return true, ipsCorrect, portsCorrect
}

func (r *leaseEndpointReconciler) RemoveEndpoints(serviceName string, ip net.IP, endpointPorts []corev1.EndpointPort) error {
	// Ensure that there will be no race condition with the ReconcileEndpoints.
	r.reconcilingLock.Lock()
	defer r.reconcilingLock.Unlock()

	if err := r.masterLeases.RemoveLease(ip.String()); err != nil {
		return err
	}

	return r.doReconcile(serviceName, endpointPorts, true)
}

func (r *leaseEndpointReconciler) StopReconciling() {
	r.stopReconcilingCalled.Store(true)
}

func (r *leaseEndpointReconciler) Destroy() {
	r.masterLeases.Destroy()
}
