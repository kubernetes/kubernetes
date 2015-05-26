/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	kubeerr "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	etcderr "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

// Etcd implements generic.Registry, backing it with etcd storage.
// It's intended to be embeddable, so that you can implement any
// non-generic functions if needed.
// You must supply a value for every field below before use; these are
// left public as it's meant to be overridable if need be.
// This object is intended to be copyable so that it can be used in
// different ways but share the same underlying behavior.
//
// The intended use of this type is embedding within a Kind specific
// RESTStorage implementation. This type provides CRUD semantics on
// a Kubelike resource, handling details like conflict detection with
// ResourceVersion and semantics. The RESTCreateStrategy and
// RESTUpdateStrategy are generic across all backends, and encapsulate
// logic specific to the API.
//
// TODO: make the default exposed methods exactly match a generic RESTStorage
type Etcd struct {
	// Called to make a new object, should return e.g., &api.Pod{}
	NewFunc func() runtime.Object

	// Called to make a new listing object, should return e.g., &api.PodList{}
	NewListFunc func() runtime.Object

	// Used for error reporting
	EndpointName string

	// Used for listing/watching; should not include trailing "/"
	KeyRootFunc func(ctx api.Context) string

	// Called for Create/Update/Get/Delete. Note that 'namespace' can be
	// gotten from ctx.
	KeyFunc func(ctx api.Context, name string) (string, error)

	// Called to get the name of an object
	ObjectNameFunc func(obj runtime.Object) (string, error)

	// Return the TTL objects should be persisted with. Update is true if this
	// is an operation against an existing object.
	TTLFunc func(obj runtime.Object, update bool) (uint64, error)

	// Returns a matcher corresponding to the provided labels and fields.
	PredicateFunc func(label labels.Selector, field fields.Selector) generic.Matcher

	// Called on all objects returned from the underlying store, after
	// the exit hooks are invoked. Decorators are intended for integrations
	// that are above etcd and should only be used for specific cases where
	// storage of the value in etcd is not appropriate, since they cannot
	// be watched.
	Decorator rest.ObjectFunc
	// Allows extended behavior during creation, required
	CreateStrategy rest.RESTCreateStrategy
	// On create of an object, attempt to run a further operation.
	AfterCreate rest.ObjectFunc
	// Allows extended behavior during updates, required
	UpdateStrategy rest.RESTUpdateStrategy
	// On update of an object, attempt to run a further operation.
	AfterUpdate rest.ObjectFunc
	// Allows extended behavior during updates, optional
	DeleteStrategy rest.RESTDeleteStrategy
	// On deletion of an object, attempt to run a further operation.
	AfterDelete rest.ObjectFunc
	// If true, return the object that was deleted. Otherwise, return a generic
	// success status response.
	ReturnDeletedObject bool

	// Used for all etcd access functions
	Helper tools.EtcdHelper
}

// NamespaceKeyRootFunc is the default function for constructing etcd paths to resource directories enforcing namespace rules.
func NamespaceKeyRootFunc(ctx api.Context, prefix string) string {
	key := prefix
	ns, ok := api.NamespaceFrom(ctx)
	if ok && len(ns) > 0 {
		key = key + "/" + ns
	}
	return key
}

// NamespaceKeyFunc is the default function for constructing etcd paths to a resource relative to prefix enforcing namespace rules.
// If no namespace is on context, it errors.
func NamespaceKeyFunc(ctx api.Context, prefix string, name string) (string, error) {
	key := NamespaceKeyRootFunc(ctx, prefix)
	ns, ok := api.NamespaceFrom(ctx)
	if !ok || len(ns) == 0 {
		return "", kubeerr.NewBadRequest("Namespace parameter required.")
	}
	if len(name) == 0 {
		return "", kubeerr.NewBadRequest("Name parameter required.")
	}
	key = key + "/" + name
	return key, nil
}

// New implements RESTStorage
func (e *Etcd) New() runtime.Object {
	return e.NewFunc()
}

// NewList implements RESTLister
func (e *Etcd) NewList() runtime.Object {
	return e.NewListFunc()
}

// List returns a list of items matching labels and field
func (e *Etcd) List(ctx api.Context, label labels.Selector, field fields.Selector) (runtime.Object, error) {
	return e.ListPredicate(ctx, e.PredicateFunc(label, field))
}

// ListPredicate returns a list of all the items matching m.
func (e *Etcd) ListPredicate(ctx api.Context, m generic.Matcher) (runtime.Object, error) {
	trace := util.NewTrace("List")
	defer trace.LogIfLong(time.Second)
	list := e.NewListFunc()
	if name, ok := m.MatchesSingle(); ok {
		trace.Step("About to read single object")
		key, err := e.KeyFunc(ctx, name)
		if err != nil {
			return nil, err
		}
		err = e.Helper.ExtractObjToList(key, list)
		trace.Step("Object extracted")
		if err != nil {
			return nil, err
		}
	} else {
		trace.Step("About to list directory")
		err := e.Helper.ExtractToList(e.KeyRootFunc(ctx), list)
		trace.Step("List extracted")
		if err != nil {
			return nil, err
		}
	}
	defer trace.Step("List filtered")
	return generic.FilterList(list, m, generic.DecoratorFunc(e.Decorator))
}

// CreateWithName inserts a new item with the provided name
// DEPRECATED: use Create instead
func (e *Etcd) CreateWithName(ctx api.Context, name string, obj runtime.Object) error {
	key, err := e.KeyFunc(ctx, name)
	if err != nil {
		return err
	}
	if e.CreateStrategy != nil {
		if err := rest.BeforeCreate(e.CreateStrategy, ctx, obj); err != nil {
			return err
		}
	}
	ttl := uint64(0)
	if e.TTLFunc != nil {
		ttl, err = e.TTLFunc(obj, false)
		if err != nil {
			return err
		}
	}
	err = e.Helper.CreateObj(key, obj, nil, ttl)
	err = etcderr.InterpretCreateError(err, e.EndpointName, name)
	if err == nil && e.Decorator != nil {
		err = e.Decorator(obj)
	}
	return err
}

// Create inserts a new item according to the unique key from the object.
func (e *Etcd) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	trace := util.NewTrace("Create")
	defer trace.LogIfLong(time.Second)
	if err := rest.BeforeCreate(e.CreateStrategy, ctx, obj); err != nil {
		return nil, err
	}
	name, err := e.ObjectNameFunc(obj)
	if err != nil {
		return nil, err
	}
	key, err := e.KeyFunc(ctx, name)
	if err != nil {
		return nil, err
	}
	ttl := uint64(0)
	if e.TTLFunc != nil {
		ttl, err = e.TTLFunc(obj, false)
		if err != nil {
			return nil, err
		}
	}
	trace.Step("About to create object")
	out := e.NewFunc()
	if err := e.Helper.CreateObj(key, obj, out, ttl); err != nil {
		err = etcderr.InterpretCreateError(err, e.EndpointName, name)
		err = rest.CheckGeneratedNameError(e.CreateStrategy, err, obj)
		return nil, err
	}
	trace.Step("Object created")
	if e.AfterCreate != nil {
		if err := e.AfterCreate(out); err != nil {
			return nil, err
		}
	}
	if e.Decorator != nil {
		if err := e.Decorator(obj); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// UpdateWithName updates the item with the provided name
// DEPRECATED: use Update instead
func (e *Etcd) UpdateWithName(ctx api.Context, name string, obj runtime.Object) error {
	key, err := e.KeyFunc(ctx, name)
	if err != nil {
		return err
	}
	ttl := uint64(0)
	if e.TTLFunc != nil {
		ttl, err = e.TTLFunc(obj, true)
		if err != nil {
			return err
		}
	}
	err = e.Helper.SetObj(key, obj, nil, ttl)
	err = etcderr.InterpretUpdateError(err, e.EndpointName, name)
	if err == nil && e.Decorator != nil {
		err = e.Decorator(obj)
	}
	return err
}

// Update performs an atomic update and set of the object. Returns the result of the update
// or an error. If the registry allows create-on-update, the create flow will be executed.
// A bool is returned along with the object and any errors, to indicate object creation.
func (e *Etcd) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	trace := util.NewTrace("Update")
	defer trace.LogIfLong(time.Second)
	name, err := e.ObjectNameFunc(obj)
	if err != nil {
		return nil, false, err
	}
	key, err := e.KeyFunc(ctx, name)
	if err != nil {
		return nil, false, err
	}
	// TODO: expose TTL
	creating := false
	out := e.NewFunc()
	err = e.Helper.GuaranteedUpdate(key, out, true, func(existing runtime.Object) (runtime.Object, uint64, error) {
		version, err := e.Helper.Versioner.ObjectResourceVersion(existing)
		if err != nil {
			return nil, 0, err
		}
		if version == 0 {
			if !e.UpdateStrategy.AllowCreateOnUpdate() {
				return nil, 0, kubeerr.NewNotFound(e.EndpointName, name)
			}
			creating = true
			if err := rest.BeforeCreate(e.CreateStrategy, ctx, obj); err != nil {
				return nil, 0, err
			}
			ttl := uint64(0)
			if e.TTLFunc != nil {
				ttl, err = e.TTLFunc(obj, false)
				if err != nil {
					return nil, 0, err
				}
			}
			return obj, ttl, nil
		}

		creating = false
		newVersion, err := e.Helper.Versioner.ObjectResourceVersion(obj)
		if err != nil {
			return nil, 0, err
		}
		if newVersion != version {
			// TODO: return the most recent version to a client?
			return nil, 0, kubeerr.NewConflict(e.EndpointName, name, fmt.Errorf("the resource was updated to %d", version))
		}
		if err := rest.BeforeUpdate(e.UpdateStrategy, ctx, obj, existing); err != nil {
			return nil, 0, err
		}
		ttl := uint64(0)
		if e.TTLFunc != nil {
			ttl, err = e.TTLFunc(obj, true)
			if err != nil {
				return nil, 0, err
			}
		}
		return obj, ttl, nil
	})

	if err != nil {
		if creating {
			err = etcderr.InterpretCreateError(err, e.EndpointName, name)
			err = rest.CheckGeneratedNameError(e.CreateStrategy, err, obj)
		} else {
			err = etcderr.InterpretUpdateError(err, e.EndpointName, name)
		}
		return nil, false, err
	}
	if creating {
		if e.AfterCreate != nil {
			if err := e.AfterCreate(out); err != nil {
				return nil, false, err
			}
		}
	} else {
		if e.AfterUpdate != nil {
			if err := e.AfterUpdate(out); err != nil {
				return nil, false, err
			}
		}
	}
	if e.Decorator != nil {
		if err := e.Decorator(obj); err != nil {
			return nil, false, err
		}
	}
	return out, creating, nil
}

// Get retrieves the item from etcd.
func (e *Etcd) Get(ctx api.Context, name string) (runtime.Object, error) {
	trace := util.NewTrace("Get")
	defer trace.LogIfLong(time.Second)
	obj := e.NewFunc()
	key, err := e.KeyFunc(ctx, name)
	if err != nil {
		return nil, err
	}
	trace.Step("About to read object")
	if err := e.Helper.ExtractObj(key, obj, false); err != nil {
		return nil, etcderr.InterpretGetError(err, e.EndpointName, name)
	}
	trace.Step("Object read")
	if e.Decorator != nil {
		if err := e.Decorator(obj); err != nil {
			return nil, err
		}
	}
	return obj, nil
}

// Delete removes the item from etcd.
func (e *Etcd) Delete(ctx api.Context, name string, options *api.DeleteOptions) (runtime.Object, error) {
	trace := util.NewTrace("Delete")
	defer trace.LogIfLong(time.Second)
	key, err := e.KeyFunc(ctx, name)
	if err != nil {
		return nil, err
	}

	obj := e.NewFunc()
	trace.Step("About to read object")
	if err := e.Helper.ExtractObj(key, obj, false); err != nil {
		return nil, etcderr.InterpretDeleteError(err, e.EndpointName, name)
	}

	// support older consumers of delete by treating "nil" as delete immediately
	if options == nil {
		options = api.NewDeleteOptions(0)
	}
	graceful, pendingGraceful, err := rest.BeforeDelete(e.DeleteStrategy, ctx, obj, options)
	if err != nil {
		return nil, err
	}
	if pendingGraceful {
		return e.finalizeDelete(obj, false)
	}
	if graceful && *options.GracePeriodSeconds != 0 {
		trace.Step("Graceful deletion")
		out := e.NewFunc()
		if err := e.Helper.SetObj(key, obj, out, uint64(*options.GracePeriodSeconds)); err != nil {
			return nil, etcderr.InterpretUpdateError(err, e.EndpointName, name)
		}
		return e.finalizeDelete(out, true)
	}

	// delete immediately, or no graceful deletion supported
	out := e.NewFunc()
	trace.Step("About to delete object")
	if err := e.Helper.DeleteObj(key, out); err != nil {
		return nil, etcderr.InterpretDeleteError(err, e.EndpointName, name)
	}
	return e.finalizeDelete(out, true)
}

func (e *Etcd) finalizeDelete(obj runtime.Object, runHooks bool) (runtime.Object, error) {
	if runHooks && e.AfterDelete != nil {
		if err := e.AfterDelete(obj); err != nil {
			return nil, err
		}
	}
	if e.ReturnDeletedObject {
		if e.Decorator != nil {
			if err := e.Decorator(obj); err != nil {
				return nil, err
			}
		}
		return obj, nil
	}
	return &api.Status{Status: api.StatusSuccess}, nil
}

// Watch makes a matcher for the given label and field, and calls
// WatchPredicate. If possible, you should customize PredicateFunc to produre a
// matcher that matches by key. generic.SelectionPredicate does this for you
// automatically.
func (e *Etcd) Watch(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return e.WatchPredicate(ctx, e.PredicateFunc(label, field), resourceVersion)
}

// WatchPredicate starts a watch for the items that m matches.
func (e *Etcd) WatchPredicate(ctx api.Context, m generic.Matcher, resourceVersion string) (watch.Interface, error) {
	version, err := tools.ParseWatchResourceVersion(resourceVersion, e.EndpointName)
	if err != nil {
		return nil, err
	}

	filterFunc := func(obj runtime.Object) bool {
		matches, err := m.Matches(obj)
		if err != nil {
			glog.Errorf("unable to match watch: %v", err)
			return false
		}
		if matches && e.Decorator != nil {
			if err := e.Decorator(obj); err != nil {
				glog.Errorf("unable to decorate watch: %v", err)
				return false
			}
		}
		return matches
	}

	if name, ok := m.MatchesSingle(); ok {
		key, err := e.KeyFunc(ctx, name)
		if err != nil {
			return nil, err
		}
		return e.Helper.Watch(key, version, filterFunc)
	}

	return e.Helper.WatchList(e.KeyRootFunc(ctx), version, filterFunc)
}
