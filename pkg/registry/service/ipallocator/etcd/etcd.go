/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package etcd

import (
	"fmt"
	"net"
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	etcderr "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service/ipallocator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
)

// Etcd exposes a service.Allocator that is backed by etcd.
// TODO: allow multiple allocations to be tried at once
// TODO: subdivide the keyspace to reduce conflicts
// TODO: investigate issuing a CAS without reading first
type Etcd struct {
	lock sync.Mutex

	alloc  ipallocator.Snapshottable
	helper tools.EtcdHelper
	last   string
}

// Etcd implements ipallocator.Interface and service.IPRegistry
var _ ipallocator.Interface = &Etcd{}
var _ service.IPRegistry = &Etcd{}

const baseKey = "/ranges/serviceips"

// NewEtcd returns a service PortalIP ipallocator that is backed by Etcd and can manage
// persisting the snapshot state of allocation after each allocation is made.
func NewEtcd(alloc ipallocator.Snapshottable, helper tools.EtcdHelper) *Etcd {
	return &Etcd{
		alloc:  alloc,
		helper: helper,
	}
}

// Allocate attempts to allocate the IP locally and then in etcd.
func (e *Etcd) Allocate(ip net.IP) error {
	e.lock.Lock()
	defer e.lock.Unlock()

	if err := e.alloc.Allocate(ip); err != nil {
		return err
	}

	return e.tryUpdate(func() error {
		return e.alloc.Allocate(ip)
	})
}

// AllocateNext attempts to allocate the next IP locally and then in etcd.
func (e *Etcd) AllocateNext() (net.IP, error) {
	e.lock.Lock()
	defer e.lock.Unlock()

	ip, err := e.alloc.AllocateNext()
	if err != nil {
		return nil, err
	}

	err = e.tryUpdate(func() error {
		if err := e.alloc.Allocate(ip); err != nil {
			if err != ipallocator.ErrAllocated {
				return err
			}
			// update the ip here
			ip, err = e.alloc.AllocateNext()
			if err != nil {
				return err
			}
		}
		return nil
	})
	return ip, err
}

// Release attempts to release the provided IP locally and then in etcd.
func (e *Etcd) Release(ip net.IP) error {
	e.lock.Lock()
	defer e.lock.Unlock()

	if err := e.alloc.Release(ip); err != nil {
		return err
	}

	return e.tryUpdate(func() error {
		return e.alloc.Release(ip)
	})
}

// tryUpdate performs a read-update to persist the latest snapshot state of allocation.
func (e *Etcd) tryUpdate(fn func() error) error {
	err := e.helper.GuaranteedUpdate(baseKey, &api.RangeAllocation{}, true,
		func(input runtime.Object) (output runtime.Object, ttl uint64, err error) {
			existing := input.(*api.RangeAllocation)
			if len(existing.ResourceVersion) == 0 {
				return nil, 0, ipallocator.ErrAllocationDisabled
			}
			if existing.ResourceVersion != e.last {
				if err := service.RestoreRange(e.alloc, existing); err != nil {
					return nil, 0, err
				}
				if err := fn(); err != nil {
					return nil, 0, err
				}
			}
			e.last = existing.ResourceVersion
			service.SnapshotRange(existing, e.alloc)
			return existing, 0, nil
		},
	)
	return etcderr.InterpretUpdateError(err, "serviceipallocation", "")
}

// Refresh reloads the ipallocator from etcd.
func (e *Etcd) Refresh() error {
	e.lock.Lock()
	defer e.lock.Unlock()

	existing := &api.RangeAllocation{}
	if err := e.helper.ExtractObj(baseKey, existing, false); err != nil {
		if tools.IsEtcdNotFound(err) {
			return ipallocator.ErrAllocationDisabled
		}
		return etcderr.InterpretGetError(err, "serviceipallocation", "")
	}

	return service.RestoreRange(e.alloc, existing)
}

// Get returns an api.RangeAllocation that represents the current state in
// etcd. If the key does not exist, the object will have an empty ResourceVersion.
func (e *Etcd) Get() (*api.RangeAllocation, error) {
	existing := &api.RangeAllocation{}
	if err := e.helper.ExtractObj(baseKey, existing, true); err != nil {
		return nil, etcderr.InterpretGetError(err, "serviceipallocation", "")
	}
	return existing, nil
}

// CreateOrUpdate attempts to update the current etcd state with the provided
// allocation.
func (e *Etcd) CreateOrUpdate(snapshot *api.RangeAllocation) error {
	e.lock.Lock()
	defer e.lock.Unlock()

	last := ""
	err := e.helper.GuaranteedUpdate(baseKey, &api.RangeAllocation{}, true,
		func(input runtime.Object) (output runtime.Object, ttl uint64, err error) {
			existing := input.(*api.RangeAllocation)
			switch {
			case len(snapshot.ResourceVersion) != 0 && len(existing.ResourceVersion) != 0:
				if snapshot.ResourceVersion != existing.ResourceVersion {
					return nil, 0, errors.NewConflict("serviceipallocation", "", fmt.Errorf("the provided resource version does not match"))
				}
			case len(existing.ResourceVersion) != 0:
				return nil, 0, errors.NewConflict("serviceipallocation", "", fmt.Errorf("another caller has already initialized the resource"))
			}
			last = snapshot.ResourceVersion
			return snapshot, 0, nil
		},
	)
	if err != nil {
		return etcderr.InterpretUpdateError(err, "serviceipallocation", "")
	}
	err = service.RestoreRange(e.alloc, snapshot)
	if err == nil {
		e.last = last
	}
	return err
}
