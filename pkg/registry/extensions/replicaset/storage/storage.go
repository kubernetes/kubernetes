/*
Copyright 2016 The Kubernetes Authors.

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

// If you make changes to this file, you should also make the corresponding change in ReplicationController.

package storage

import (
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extvalidation "k8s.io/kubernetes/pkg/apis/extensions/validation"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/extensions/replicaset"
)

// ReplicaSetStorage includes dummy storage for ReplicaSets and for Scale subresource.
type ReplicaSetStorage struct {
	ReplicaSet *REST
	Status     *StatusREST
	Scale      *ScaleREST
}

func NewStorage(optsGetter generic.RESTOptionsGetter) ReplicaSetStorage {
	replicaSetRest, replicaSetStatusRest := NewREST(optsGetter)
	replicaSetRegistry := replicaset.NewRegistry(replicaSetRest)

	return ReplicaSetStorage{
		ReplicaSet: replicaSetRest,
		Status:     replicaSetStatusRest,
		Scale:      &ScaleREST{registry: replicaSetRegistry},
	}
}

type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against ReplicaSet.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, *StatusREST) {
	store := &genericregistry.Store{
		Copier:      api.Scheme,
		NewFunc:     func() runtime.Object { return &extensions.ReplicaSet{} },
		NewListFunc: func() runtime.Object { return &extensions.ReplicaSetList{} },
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*extensions.ReplicaSet).Name, nil
		},
		PredicateFunc:     replicaset.MatchReplicaSet,
		QualifiedResource: extensions.Resource("replicasets"),
		WatchCacheSize:    cachesize.GetWatchCacheSizeByResource("replicasets"),

		CreateStrategy: replicaset.Strategy,
		UpdateStrategy: replicaset.Strategy,
		DeleteStrategy: replicaset.Strategy,
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: replicaset.GetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}

	statusStore := *store
	statusStore.UpdateStrategy = replicaset.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}
}

// Implement ShortNamesProvider
var _ rest.ShortNamesProvider = &REST{}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"rs"}
}

// StatusREST implements the REST endpoint for changing the status of a ReplicaSet
type StatusREST struct {
	store *genericregistry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &extensions.ReplicaSet{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}

type ScaleREST struct {
	registry replicaset.Registry
}

// ScaleREST implements Patcher
var _ = rest.Patcher(&ScaleREST{})

// New creates a new Scale object
func (r *ScaleREST) New() runtime.Object {
	return &extensions.Scale{}
}

func (r *ScaleREST) Get(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	rs, err := r.registry.GetReplicaSet(ctx, name, options)
	if err != nil {
		return nil, errors.NewNotFound(extensions.Resource("replicasets/scale"), name)
	}
	scale, err := scaleFromReplicaSet(rs)
	if err != nil {
		return nil, errors.NewBadRequest(fmt.Sprintf("%v", err))
	}
	return scale, err
}

func (r *ScaleREST) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	rs, err := r.registry.GetReplicaSet(ctx, name, &metav1.GetOptions{})
	if err != nil {
		return nil, false, errors.NewNotFound(extensions.Resource("replicasets/scale"), name)
	}

	oldScale, err := scaleFromReplicaSet(rs)
	if err != nil {
		return nil, false, err
	}

	obj, err := objInfo.UpdatedObject(ctx, oldScale)
	if err != nil {
		return nil, false, err
	}
	if obj == nil {
		return nil, false, errors.NewBadRequest(fmt.Sprintf("nil update passed to Scale"))
	}
	scale, ok := obj.(*extensions.Scale)
	if !ok {
		return nil, false, errors.NewBadRequest(fmt.Sprintf("wrong object passed to Scale update: %v", obj))
	}

	if errs := extvalidation.ValidateScale(scale); len(errs) > 0 {
		return nil, false, errors.NewInvalid(extensions.Kind("Scale"), scale.Name, errs)
	}

	rs.Spec.Replicas = scale.Spec.Replicas
	rs.ResourceVersion = scale.ResourceVersion
	rs, err = r.registry.UpdateReplicaSet(ctx, rs)
	if err != nil {
		return nil, false, err
	}
	newScale, err := scaleFromReplicaSet(rs)
	if err != nil {
		return nil, false, errors.NewBadRequest(fmt.Sprintf("%v", err))
	}
	return newScale, false, err
}

// scaleFromReplicaSet returns a scale subresource for a replica set.
func scaleFromReplicaSet(rs *extensions.ReplicaSet) (*extensions.Scale, error) {
	return &extensions.Scale{
		// TODO: Create a variant of ObjectMeta type that only contains the fields below.
		ObjectMeta: metav1.ObjectMeta{
			Name:              rs.Name,
			Namespace:         rs.Namespace,
			UID:               rs.UID,
			ResourceVersion:   rs.ResourceVersion,
			CreationTimestamp: rs.CreationTimestamp,
		},
		Spec: extensions.ScaleSpec{
			Replicas: rs.Spec.Replicas,
		},
		Status: extensions.ScaleStatus{
			Replicas: rs.Status.Replicas,
			Selector: rs.Spec.Selector,
		},
	}, nil
}
