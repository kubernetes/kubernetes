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

package storage

import (
	"context"
	"errors"
	"fmt"
	"sync"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/storage"
	storeerr "k8s.io/apiserver/pkg/storage/errors"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	api "k8s.io/kubernetes/pkg/apis/core"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/registry/core/rangeallocation"
	"k8s.io/kubernetes/pkg/registry/core/service/allocator"
)

var (
	errorUnableToAllocate = errors.New("unable to allocate")
)

// Etcd exposes a service.Allocator
// TODO: allow multiple allocations to be tried at once
// TODO: subdivide the keyspace to reduce conflicts
// TODO: investigate issuing a CAS without reading first
type Etcd struct {
	lock sync.Mutex

	alloc   allocator.Snapshottable
	storage storage.Interface
	last    string

	baseKey  string
	resource schema.GroupResource
}

// Etcd implements allocator.Interface and rangeallocation.RangeRegistry
var _ allocator.Interface = &Etcd{}
var _ rangeallocation.RangeRegistry = &Etcd{}

// NewEtcd returns an allocator that is backed by Etcd and can manage
// persisting the snapshot state of allocation after each allocation is made.
func NewEtcd(alloc allocator.Snapshottable, baseKey string, resource schema.GroupResource, config *storagebackend.Config) (*Etcd, error) {
	storage, d, err := generic.NewRawStorage(config, nil)
	if err != nil {
		return nil, err
	}

	// TODO : Remove RegisterStorageCleanup below when PR
	// https://github.com/kubernetes/kubernetes/pull/50690
	// merges as that shuts down storage properly
	registry.RegisterStorageCleanup(d)

	return &Etcd{
		alloc:    alloc,
		storage:  storage,
		baseKey:  baseKey,
		resource: resource,
	}, nil
}

// Allocate attempts to allocate the item.
func (e *Etcd) Allocate(offset int) (bool, error) {
	e.lock.Lock()
	defer e.lock.Unlock()

	err := e.tryUpdate(func() error {
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

// AllocateNext attempts to allocate the next item.
func (e *Etcd) AllocateNext() (int, bool, error) {
	e.lock.Lock()
	defer e.lock.Unlock()
	var offset int
	var ok bool
	var err error

	err = e.tryUpdate(func() error {
		// update the offset here
		offset, ok, err = e.alloc.AllocateNext()
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
			return offset, false, nil
		}
		return offset, false, err
	}
	return offset, true, nil
}

// Release attempts to release the provided item.
func (e *Etcd) Release(item int) error {
	e.lock.Lock()
	defer e.lock.Unlock()

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
			}
			if err := fn(); err != nil {
				return nil, err
			}
			e.last = existing.ResourceVersion
			rangeSpec, data := e.alloc.Snapshot()
			existing.Range = rangeSpec
			existing.Data = data
			return existing, nil
		}),
		nil,
	)
	return storeerr.InterpretUpdateError(err, e.resource, "")
}

// Get returns an api.RangeAllocation that represents the current state in
// etcd. If the key does not exist, the object will have an empty ResourceVersion.
func (e *Etcd) Get() (*api.RangeAllocation, error) {
	existing := &api.RangeAllocation{}
	if err := e.storage.Get(context.TODO(), e.baseKey, storage.GetOptions{IgnoreNotFound: true}, existing); err != nil {
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
					return nil, apierrors.NewConflict(e.resource, "", fmt.Errorf("the provided resource version does not match"))
				}
			case len(existing.ResourceVersion) != 0:
				return nil, apierrors.NewConflict(e.resource, "", fmt.Errorf("another caller has already initialized the resource"))
			}
			last = snapshot.ResourceVersion
			return snapshot, nil
		}),
		nil,
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
