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
	"path"

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/registry/namespace"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// rest implements a RESTStorage for namespaces against etcd
type REST struct {
	*etcdgeneric.Etcd
	status *etcdgeneric.Etcd
}

// StatusREST implements the REST endpoint for changing the status of a namespace.
type StatusREST struct {
	store *etcdgeneric.Etcd
}

// FinalizeREST implements the REST endpoint for finalizing a namespace.
type FinalizeREST struct {
	store *etcdgeneric.Etcd
}

// NewREST returns a RESTStorage object that will work against namespaces.
func NewREST(s storage.Interface) (*REST, *StatusREST, *FinalizeREST) {
	prefix := "/namespaces"
	store := &etcdgeneric.Etcd{
		NewFunc:     func() runtime.Object { return &api.Namespace{} },
		NewListFunc: func() runtime.Object { return &api.NamespaceList{} },
		KeyRootFunc: func(ctx api.Context) string {
			return prefix
		},
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return path.Join(prefix, name), nil
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.Namespace).Name, nil
		},
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return namespace.MatchNamespace(label, field)
		},
		EndpointName: "namespaces",

		CreateStrategy:      namespace.Strategy,
		UpdateStrategy:      namespace.Strategy,
		ReturnDeletedObject: true,

		Storage: s,
	}

	statusStore := *store
	statusStore.UpdateStrategy = namespace.StatusStrategy

	finalizeStore := *store
	finalizeStore.UpdateStrategy = namespace.FinalizeStrategy

	return &REST{Etcd: store, status: &statusStore}, &StatusREST{store: &statusStore}, &FinalizeREST{store: &finalizeStore}
}

// Delete enforces life-cycle rules for namespace termination
func (r *REST) Delete(ctx api.Context, name string, options *api.DeleteOptions) (runtime.Object, error) {
	nsObj, err := r.Get(ctx, name)
	if err != nil {
		return nil, err
	}

	namespace := nsObj.(*api.Namespace)

	// upon first request to delete, we switch the phase to start namespace termination
	if namespace.DeletionTimestamp.IsZero() {
		now := unversioned.Now()
		namespace.DeletionTimestamp = &now
		namespace.Status.Phase = api.NamespaceTerminating
		result, _, err := r.status.Update(ctx, namespace)
		return result, err
	}

	// prior to final deletion, we must ensure that finalizers is empty
	if len(namespace.Spec.Finalizers) != 0 {
		err = apierrors.NewConflict("Namespace", namespace.Name, fmt.Errorf("The system is ensuring all content is removed from this namespace.  Upon completion, this namespace will automatically be purged by the system."))
		return nil, err
	}
	return r.Etcd.Delete(ctx, name, nil)
}

func (r *StatusREST) New() runtime.Object {
	return r.store.New()
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	return r.store.Update(ctx, obj)
}

func (r *FinalizeREST) New() runtime.Object {
	return r.store.New()
}

// Update alters the status finalizers subset of an object.
func (r *FinalizeREST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	return r.store.Update(ctx, obj)
}
