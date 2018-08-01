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
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kruntime "k8s.io/apimachinery/pkg/runtime"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/kubernetes/pkg/api/endpoints"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// Leases is an interface which assists in managing the set of active masters
type Leases interface {
	// ListLeases retrieves a list of the current master IPs
	ListLeases() ([]string, error)

	// UpdateLease adds or refreshes a master's lease
	UpdateLease(ip string) error

	// RemoveLease removes a master's lease
	RemoveLease(ip string) error
}

type storageLeases struct {
	storage   storage.Interface
	baseKey   string
	leaseTime time.Duration
}

var _ Leases = &storageLeases{}

// ListLeases retrieves a list of the current master IPs from storage
func (s *storageLeases) ListLeases() ([]string, error) {
	ipInfoList := &api.EndpointsList{}
	if err := s.storage.List(apirequest.NewDefaultContext(), s.baseKey, "0", storage.Everything, ipInfoList); err != nil {
		return nil, err
	}

	ipList := make([]string, len(ipInfoList.Items))
	for i, ip := range ipInfoList.Items {
		ipList[i] = ip.Subsets[0].Addresses[0].IP
	}

	glog.V(6).Infof("Current master IPs listed in storage are %v", ipList)

	return ipList, nil
}

// UpdateLease resets the TTL on a master IP in storage
func (s *storageLeases) UpdateLease(ip string) error {
	key := path.Join(s.baseKey, ip)
	return s.storage.GuaranteedUpdate(apirequest.NewDefaultContext(), key, &api.Endpoints{}, true, nil, func(input kruntime.Object, respMeta storage.ResponseMeta) (kruntime.Object, *uint64, error) {
		// just make sure we've got the right IP set, and then refresh the TTL
		existing := input.(*api.Endpoints)
		existing.Subsets = []api.EndpointSubset{
			{
				Addresses: []api.EndpointAddress{{IP: ip}},
			},
		}

		// leaseTime needs to be in seconds
		leaseTime := uint64(s.leaseTime / time.Second)

		// NB: GuaranteedUpdate does not perform the store operation unless
		// something changed between load and store (not including resource
		// version), meaning we can't refresh the TTL without actually
		// changing a field.
		existing.Generation++

		glog.V(6).Infof("Resetting TTL on master IP %q listed in storage to %v", ip, leaseTime)

		return existing, &leaseTime, nil
	})
}

// RemoveLease removes the lease on a master IP in storage
func (s *storageLeases) RemoveLease(ip string) error {
	return s.storage.Delete(apirequest.NewDefaultContext(), s.baseKey+"/"+ip, &api.Endpoints{}, nil)
}

// NewLeases creates a new etcd-based Leases implementation.
func NewLeases(storage storage.Interface, baseKey string, leaseTime time.Duration) Leases {
	return &storageLeases{
		storage:   storage,
		baseKey:   baseKey,
		leaseTime: leaseTime,
	}
}

type leaseEndpointReconciler struct {
	endpointStorage       rest.StandardStorage
	masterLeases          Leases
	stopReconcilingCalled bool
	reconcilingLock       sync.Mutex
}

// NewLeaseEndpointReconciler creates a new LeaseEndpoint reconciler
func NewLeaseEndpointReconciler(endpointStorage rest.StandardStorage, masterLeases Leases) EndpointReconciler {
	return &leaseEndpointReconciler{
		endpointStorage:       endpointStorage,
		masterLeases:          masterLeases,
		stopReconcilingCalled: false,
	}
}

// ReconcileEndpoints lists keys in a special etcd directory.
// Each key is expected to have a TTL of R+n, where R is the refresh interval
// at which this function is called, and n is some small value.  If an
// apiserver goes down, it will fail to refresh its key's TTL and the key will
// expire. ReconcileEndpoints will notice that the endpoints object is
// different from the directory listing, and update the endpoints object
// accordingly.
func (r *leaseEndpointReconciler) ReconcileEndpoints(serviceName string, ip net.IP, endpointPorts []api.EndpointPort, reconcilePorts bool) error {
	r.reconcilingLock.Lock()
	defer r.reconcilingLock.Unlock()

	if r.stopReconcilingCalled {
		return nil
	}

	// Refresh the TTL on our key, independently of whether any error or
	// update conflict happens below. This makes sure that at least some of
	// the masters will add our endpoint.
	if err := r.masterLeases.UpdateLease(ip.String()); err != nil {
		return err
	}

	return r.doReconcile(serviceName, endpointPorts, reconcilePorts)
}

func (r *leaseEndpointReconciler) doReconcile(serviceName string, endpointPorts []api.EndpointPort, reconcilePorts bool) error {
	ctx := apirequest.NewDefaultContext()

	// Retrieve the current list of endpoints...
	var e *api.Endpoints
	obj, err := r.endpointStorage.Get(ctx, serviceName, &metav1.GetOptions{})
	if err != nil {
		if !errors.IsNotFound(err) {
			return err
		}

		e = &api.Endpoints{
			ObjectMeta: metav1.ObjectMeta{
				Name:      serviceName,
				Namespace: api.NamespaceDefault,
			},
		}
	} else {
		e = obj.(*api.Endpoints)
	}

	// ... and the list of master IP keys from etcd
	masterIPs, err := r.masterLeases.ListLeases()
	if err != nil {
		return err
	}

	// Since we just refreshed our own key, assume that zero endpoints
	// returned from storage indicates an issue or invalid state, and thus do
	// not update the endpoints list based on the result.
	if len(masterIPs) == 0 {
		return fmt.Errorf("no master IPs were listed in storage, refusing to erase all endpoints for the kubernetes service")
	}

	// Next, we compare the current list of endpoints with the list of master IP keys
	formatCorrect, ipCorrect, portsCorrect := checkEndpointSubsetFormatWithLease(e, masterIPs, endpointPorts, reconcilePorts)
	if formatCorrect && ipCorrect && portsCorrect {
		return nil
	}

	if !formatCorrect {
		// Something is egregiously wrong, just re-make the endpoints record.
		e.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{},
			Ports:     endpointPorts,
		}}
	}

	if !formatCorrect || !ipCorrect {
		// repopulate the addresses according to the expected IPs from etcd
		e.Subsets[0].Addresses = make([]api.EndpointAddress, len(masterIPs))
		for ind, ip := range masterIPs {
			e.Subsets[0].Addresses[ind] = api.EndpointAddress{IP: ip}
		}

		// Lexicographic order is retained by this step.
		e.Subsets = endpoints.RepackSubsets(e.Subsets)
	}

	if !portsCorrect {
		// Reset ports.
		e.Subsets[0].Ports = endpointPorts
	}

	glog.Warningf("Resetting endpoints for master service %q to %v", serviceName, masterIPs)
	_, _, err = r.endpointStorage.Update(ctx, e.Name, rest.DefaultUpdatedObjectInfo(e), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	return err
}

// checkEndpointSubsetFormatWithLease determines if the endpoint is in the
// format ReconcileEndpoints expects when the controller is using leases.
//
// Return values:
// * formatCorrect is true if exactly one subset is found.
// * ipsCorrect when the addresses in the endpoints match the expected addresses list
// * portsCorrect is true when endpoint ports exactly match provided ports.
//     portsCorrect is only evaluated when reconcilePorts is set to true.
func checkEndpointSubsetFormatWithLease(e *api.Endpoints, expectedIPs []string, ports []api.EndpointPort, reconcilePorts bool) (formatCorrect bool, ipsCorrect bool, portsCorrect bool) {
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

func (r *leaseEndpointReconciler) StopReconciling(serviceName string, ip net.IP, endpointPorts []api.EndpointPort) error {
	r.reconcilingLock.Lock()
	defer r.reconcilingLock.Unlock()
	r.stopReconcilingCalled = true

	if err := r.masterLeases.RemoveLease(ip.String()); err != nil {
		return err
	}

	return r.doReconcile(serviceName, endpointPorts, true)
}
