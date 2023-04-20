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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	storagefactory "k8s.io/apiserver/pkg/storage/storagebackend/factory"
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
	epAdapter             *EndpointsAdapter
	masterLeases          Leases
	stopReconcilingCalled atomic.Bool
	reconcilingLock       sync.Mutex
}

// NewLeaseEndpointReconciler creates a new LeaseEndpoint reconciler
func NewLeaseEndpointReconciler(epAdapter *EndpointsAdapter, masterLeases Leases) EndpointReconciler {
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
func (r *leaseEndpointReconciler) ReconcileEndpoints(ip net.IP, endpointPorts []corev1.EndpointPort, reconcilePorts bool) error {
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

	return r.doReconcile(endpointPorts, reconcilePorts)
}

// doReconcile can be called from ReconcileEndpoints() or RemoveEndpoints().
// it is NOT SAFE to call it from multiple goroutines.
func (r *leaseEndpointReconciler) doReconcile(endpointPorts []corev1.EndpointPort, reconcilePorts bool) error {
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

	return r.epAdapter.Sync(sets.New(masterIPs...), endpointPorts, reconcilePorts)
}

func (r *leaseEndpointReconciler) RemoveEndpoints(ip net.IP, endpointPorts []corev1.EndpointPort) error {
	// Ensure that there will be no race condition with the ReconcileEndpoints.
	r.reconcilingLock.Lock()
	defer r.reconcilingLock.Unlock()

	if err := r.masterLeases.RemoveLease(ip.String()); err != nil {
		return err
	}

	return r.doReconcile(endpointPorts, true)
}

func (r *leaseEndpointReconciler) StopReconciling() {
	r.stopReconcilingCalled.Store(true)
}

func (r *leaseEndpointReconciler) Destroy() {
	r.masterLeases.Destroy()
}
