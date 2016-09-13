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
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	storeerr "k8s.io/kubernetes/pkg/api/errors/storage"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extvalidation "k8s.io/kubernetes/pkg/apis/extensions/validation"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/extensions/daemonset"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// rest implements a RESTStorage for DaemonSets against etcd
type REST struct {
	*registry.Store
}

// NewREST returns a RESTStorage object that will work against DaemonSets.
func NewREST(opts generic.RESTOptions) (*REST, *StatusREST, *RollbackREST) {
	prefix := "/" + opts.ResourcePrefix

	newListFunc := func() runtime.Object { return &extensions.DaemonSetList{} }
	storageInterface, dFunc := opts.Decorator(
		opts.StorageConfig,
		cachesize.GetWatchCacheSizeByResource(cachesize.Daemonsets),
		&extensions.DaemonSet{},
		prefix,
		daemonset.Strategy,
		newListFunc,
		storage.NoTriggerPublisher,
	)

	store := &registry.Store{
		NewFunc: func() runtime.Object { return &extensions.DaemonSet{} },

		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: newListFunc,
		// Produces a path that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix
		KeyRootFunc: func(ctx api.Context) string {
			return registry.NamespaceKeyRootFunc(ctx, prefix)
		},
		// Produces a path that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return registry.NamespaceKeyFunc(ctx, prefix, name)
		},
		// Retrieve the name field of a daemon set
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*extensions.DaemonSet).Name, nil
		},
		// Used to match objects based on labels/fields for list and watch
		PredicateFunc:           daemonset.MatchDaemonSet,
		QualifiedResource:       extensions.Resource("daemonsets"),
		EnableGarbageCollection: opts.EnableGarbageCollection,
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		// Used to validate daemon set creation
		CreateStrategy: daemonset.Strategy,

		// Used to validate daemon set updates
		UpdateStrategy: daemonset.Strategy,
		DeleteStrategy: daemonset.Strategy,

		Storage:     storageInterface,
		DestroyFunc: dFunc,
	}
	statusStore := *store
	statusStore.UpdateStrategy = daemonset.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}, &RollbackREST{store: store}
}

// StatusREST implements the REST endpoint for changing the status of a daemonset
type StatusREST struct {
	store *registry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &extensions.DaemonSet{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx api.Context, name string) (runtime.Object, error) {
	return r.store.Get(ctx, name)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}

// RollbackREST implements the REST endpoint for initiating the rollback of a daemon set
type RollbackREST struct {
	store *registry.Store
}

// New creates a rollback
func (r *RollbackREST) New() runtime.Object {
	return &extensions.DaemonSetRollback{}
}

var _ = rest.Creater(&RollbackREST{})

func (r *RollbackREST) Create(ctx api.Context, obj runtime.Object) (out runtime.Object, err error) {
	rollback, ok := obj.(*extensions.DaemonSetRollback)
	if !ok {
		return nil, fmt.Errorf("expected input object type to be DaemonSetRollback, but %T", obj)
	}

	if errs := extvalidation.ValidateDaemonSetRollback(rollback); len(errs) != 0 {
		return nil, errors.NewInvalid(extensions.Kind("DaemonSetRollback"), rollback.Name, errs)
	}

	// Update the DaemonSet with information in DaemonSetRollback to trigger rollback
	err = r.rollbackDaemonSet(ctx, rollback.Name, &rollback.RollbackTo, rollback.UpdatedAnnotations)
	return
}

func (r *RollbackREST) rollbackDaemonSet(ctx api.Context, daemonID string, config *extensions.RollbackConfig, annotations map[string]string) (err error) {
	if _, err = r.setDaemonSetRollback(ctx, daemonID, config, annotations); err != nil {
		err = storeerr.InterpretGetError(err, extensions.Resource("daemonsets"), daemonID)
		err = storeerr.InterpretUpdateError(err, extensions.Resource("daemonsets"), daemonID)
		if _, ok := err.(*errors.StatusError); !ok {
			err = errors.NewConflict(extensions.Resource("daemonsets/rollback"), daemonID, err)
		}
	}
	return
}

func (r *RollbackREST) setDaemonSetRollback(ctx api.Context, daemonID string, config *extensions.RollbackConfig, annotations map[string]string) (finalDaemonSet *extensions.DaemonSet, err error) {
	dKey, err := r.store.KeyFunc(ctx, daemonID)
	if err != nil {
		return nil, err
	}
	err = r.store.Storage.GuaranteedUpdate(ctx, dKey, &extensions.DaemonSet{}, false, nil, storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
		d, ok := obj.(*extensions.DaemonSet)
		if !ok {
			return nil, fmt.Errorf("unexpected object: %#v", obj)
		}
		if d.Annotations == nil {
			d.Annotations = make(map[string]string)
		}
		for k, v := range annotations {
			d.Annotations[k] = v
		}
		d.Spec.RollbackTo = config
		finalDaemonSet = d
		return d, nil
	}))
	return finalDaemonSet, err
}
