/*
Copyright 2015 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"sync"

	"k8s.io/kubernetes/pkg/api"
	k8serr "k8s.io/kubernetes/pkg/api/errors"
	storeerr "k8s.io/kubernetes/pkg/api/errors/storage"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/registry/core/rangeallocation"
	"k8s.io/kubernetes/pkg/registry/core/service/allocator"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/storage/storagebackend"

	"golang.org/x/net/context"
)

var (
	errorUnableToAllocate = errors.New("unable to allocate")
)

// Etcd exposes a service.Allocator that is backed by etcd.
// TODO: allow multiple allocations to be tried at once
// TODO: subdivide the keyspace to reduce conflicts
// TODO: investigate issuing a CAS without reading first
type Etcd struct {
	lock sync.Mutex

	alloc   allocator.Snapshottable
	storage storage.Interface
	last    string

	baseKey  string
	resource unversioned.GroupResource
}

// Etcd implements allocator.Interface and rangeallocation.RangeRegistry
var _ allocator.Interface = &Etcd{}
var _ rangeallocation.RangeRegistry = &Etcd{}

// NewEtcd returns an allocator that is backed by Etcd and can manage
// persisting the snapshot state of allocation after each allocation is made.
func NewEtcd(alloc allocator.Snapshottable, baseKey string, resource unversioned.GroupResource, config *storagebackend.Config) *Etcd {
	storage, _ := generic.NewRawStorage(config)
	return &Etcd{
		alloc:    alloc,
		storage:  storage,
		baseKey:  baseKey,
		resource: resource,
	}
}

// Allocate attempts to allocate the item locally and then in etcd.
func (e *Etcd) Allocate(offset int) (bool, error) {
	e.lock.Lock()
	defer e.lock.Unlock()

	ok, err := e.alloc.Allocate(offset)
	if !ok || err != nil {
		return ok, err
	}

	err = e.tryUpdate(func() error {
		ok, err := e.alloc.Allocate(offset)
		if err != nil {
			return err
		}
		if !ok {
			return errorUnableToAllocate
		}
		return nil
	})
	if err != nil {
		if err == errorUnableToAllocate {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

// AllocateNext attempts to allocate the next item locally and then in etcd.
func (e *Etcd) AllocateNext() (int, bool, error) {
	e.lock.Lock()
	defer e.lock.Unlock()

	offset, ok, err := e.alloc.AllocateNext()
	if !ok || err != nil {
		return offset, ok, err
	}

	err = e.tryUpdate(func() error {
		ok, err := e.alloc.Allocate(offset)
		if err != nil {
			return err
		}
		if !ok {
			// update the offset here
			offset, ok, err = e.alloc.AllocateNext()
			if err != nil {
				return err
			}
			if !ok {
				return errorUnableToAllocate
			}
			return nil
		}
		return nil
	})
	return offset, ok, err
}

// Release attempts to release the provided item locally and then in etcd.
func (e *Etcd) Release(item int) error {
	e.lock.Lock()
	defer e.lock.Unlock()

	if err := e.alloc.Release(item); err != nil {
		return err
	}

	return e.tryUpdate(func() error {
		return e.alloc.Release(item)
	})
}

func (e *Etcd) ForEach(fn func(int)) {
	e.lock.Lock()
	defer e.lock.Unlock()
	e.alloc.ForEach(fn)
}

// tryUpdate performs a read-update to persist the latest snapshot state of allocation.
func (e *Etcd) tryUpdate(fn func() error) error {
	err := e.storage.GuaranteedUpdate(context.TODO(), e.baseKey, &api.RangeAllocation{}, true, nil,
		storage.SimpleUpdate(func(input runtime.Object) (output runtime.Object, err error) {
			existing := input.(*api.RangeAllocation)
			if len(existing.ResourceVersion) == 0 {
				return nil, fmt.Errorf("cannot allocate resources of type %s at this time", e.resource.String())
			}
			if existing.ResourceVersion != e.last {
				if err := e.alloc.Restore(existing.Range, existing.Data); err != nil {
					return nil, err
				}
				if err := fn(); err != nil {
					return nil, err
				}
			}
			e.last = existing.ResourceVersion
			rangeSpec, data := e.alloc.Snapshot()
			existing.Range = rangeSpec
			existing.Data = data
			return existing, nil
		}),
	)
	return storeerr.InterpretUpdateError(err, e.resource, "")
}

// Get returns an api.RangeAllocation that represents the current state in
// etcd. If the key does not exist, the object will have an empty ResourceVersion.
func (e *Etcd) Get() (*api.RangeAllocation, error) {
	existing := &api.RangeAllocation{}
	if err := e.storage.Get(context.TODO(), e.baseKey, existing, true); err != nil {
		return nil, storeerr.InterpretGetError(err, e.resource, "")
	}
	return existing, nil
}

// CreateOrUpdate attempts to update the current etcd state with the provided
// allocation.
func (e *Etcd) CreateOrUpdate(snapshot *api.RangeAllocation) error {
	e.lock.Lock()
	defer e.lock.Unlock()

	last := ""
	err := e.storage.GuaranteedUpdate(context.TODO(), e.baseKey, &api.RangeAllocation{}, true, nil,
		storage.SimpleUpdate(func(input runtime.Object) (output runtime.Object, err error) {
			existing := input.(*api.RangeAllocation)
			switch {
			case len(snapshot.ResourceVersion) != 0 && len(existing.ResourceVersion) != 0:
				if snapshot.ResourceVersion != existing.ResourceVersion {
					return nil, k8serr.NewConflict(e.resource, "", fmt.Errorf("the provided resource version does not match"))
				}
			case len(existing.ResourceVersion) != 0:
				return nil, k8serr.NewConflict(e.resource, "", fmt.Errorf("another caller has already initialized the resource"))
			}
			last = snapshot.ResourceVersion
			return snapshot, nil
		}),
	)
	if err != nil {
		return storeerr.InterpretUpdateError(err, e.resource, "")
	}
	err = e.alloc.Restore(snapshot.Range, snapshot.Data)
	if err == nil {
		e.last = last
	}
	return err
}

// Implements allocator.Interface::Has
func (e *Etcd) Has(item int) bool {
	e.lock.Lock()
	defer e.lock.Unlock()

	return e.alloc.Has(item)
}

// Implements allocator.Interface::Free
func (e *Etcd) Free() int {
	e.lock.Lock()
	defer e.lock.Unlock()

	return e.alloc.Free()
}
