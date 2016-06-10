/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package leases

import (
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	kruntime "k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// Leases is an interface which assists in managing the set of active masters
type Leases interface {
	// ListLeases retrieves a list of the current master IPs
	ListLeases() ([]string, error)

	// UpdateLease adds or refreshes a master's lease
	UpdateLease(ip string) error

	// SetLeaseTime configures base lease time
	SetLeaseTime(ttl uint64)
}

type storageLeases struct {
	storage   storage.Interface
	baseKey   string
	leaseTime uint64
}

var _ Leases = &storageLeases{}

// ListLeases retrieves a list of the current master IPs from storage
func (s *storageLeases) ListLeases() ([]string, error) {
	ipInfoList := &api.EndpointsList{}
	if err := s.storage.List(api.NewDefaultContext(), s.baseKey, "0", storage.Everything, ipInfoList); err != nil {
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
	return s.storage.GuaranteedUpdate(api.NewDefaultContext(), s.baseKey+"/"+ip, &api.Endpoints{}, true, nil, func(input kruntime.Object, respMeta storage.ResponseMeta) (kruntime.Object, *uint64, error) {
		// just make sure we've got the right IP set, and then refresh the TTL
		existing := input.(*api.Endpoints)
		existing.Subsets = []api.EndpointSubset{
			{
				Addresses: []api.EndpointAddress{{IP: ip}},
			},
		}

		leaseTime := s.leaseTime

		// NB: GuaranteedUpdate does not perform the store operation unless
		// something changed between load and store (not including resource
		// version), meaning we can't refresh the TTL without actually
		// changing a field.
		existing.Generation += 1

		glog.V(6).Infof("Resetting TTL on master IP %q listed in storage to %v", ip, leaseTime)

		return existing, &leaseTime, nil
	})
}

// SetLeaseTTL configures the base TTL to use when resetting the TTL in UpdateLease
func (s *storageLeases) SetLeaseTime(ttl uint64) {
	s.leaseTime = ttl + 2 // give ourselves some wiggle room
}

// NewLeases creates a new etcd-based Leases implementation. It is expected
// that the lease time will be set later with SetLeaseTime
func NewLeases(storage storage.Interface, baseKey string) Leases {
	return &storageLeases{
		storage:   storage,
		baseKey:   baseKey,
		leaseTime: 0, // expected to be updated later
	}
}
