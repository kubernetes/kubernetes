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
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extvalidation "k8s.io/kubernetes/pkg/apis/extensions/validation"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/apps/statefulset"
)

// StatefulSetStorage includes dummy storage for StatefulSets, and their Status and Scale subresource.
type StatefulSetStorage struct {
	StatefulSet *REST
	Status      *StatusREST
	Scale       *ScaleREST
}

func NewStorage(optsGetter generic.RESTOptionsGetter) StatefulSetStorage {
	statefulSetRest, statefulSetStatusRest := NewREST(optsGetter)
	statefulSetRegistry := statefulset.NewRegistry(statefulSetRest)

	return StatefulSetStorage{
		StatefulSet: statefulSetRest,
		Status:      statefulSetStatusRest,
		Scale:       &ScaleREST{registry: statefulSetRegistry},
	}
}

// rest implements a RESTStorage for statefulsets against etcd
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against statefulsets.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, *StatusREST) {
	store := &genericregistry.Store{
		Copier:                   api.Scheme,
		NewFunc:                  func() runtime.Object { return &apps.StatefulSet{} },
		NewListFunc:              func() runtime.Object { return &apps.StatefulSetList{} },
		DefaultQualifiedResource: apps.Resource("statefulsets"),

		CreateStrategy: statefulset.Strategy,
		UpdateStrategy: statefulset.Strategy,
		DeleteStrategy: statefulset.Strategy,

		TableConvertor: printerstorage.TableConvertor{TablePrinter: printers.NewTablePrinter().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}

	statusStore := *store
	statusStore.UpdateStrategy = statefulset.StatusStrategy
	return &REST{store}, &StatusREST{store: &statusStore}
}

// Implement CategoriesProvider
var _ rest.CategoriesProvider = &REST{}

// Categories implements the CategoriesProvider interface. Returns a list of categories a resource is part of.
func (r *REST) Categories() []string {
	return []string{"all"}
}

// StatusREST implements the REST endpoint for changing the status of an statefulSet
type StatusREST struct {
	store *genericregistry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &apps.StatefulSet{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}

// Implement ShortNamesProvider
var _ rest.ShortNamesProvider = &REST{}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"sts"}
}

type ScaleREST struct {
	registry statefulset.Registry
}

// ScaleREST implements Patcher
var _ = rest.Patcher(&ScaleREST{})
var _ = rest.GroupVersionKindProvider(&ScaleREST{})

func (r *ScaleREST) GroupVersionKind() schema.GroupVersionKind {
	return schema.GroupVersionKind{Group: "extensions", Version: "v1beta1", Kind: "Scale"}
}

// New creates a new Scale object
func (r *ScaleREST) New() runtime.Object {
	return &extensions.Scale{}
}

func (r *ScaleREST) Get(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	ss, err := r.registry.GetStatefulSet(ctx, name, options)
	if err != nil {
		return nil, err
	}
	scale, err := scaleFromStatefulSet(ss)
	if err != nil {
		return nil, errors.NewBadRequest(fmt.Sprintf("%v", err))
	}
	return scale, err
}

func (r *ScaleREST) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	ss, err := r.registry.GetStatefulSet(ctx, name, &metav1.GetOptions{})
	if err != nil {
		return nil, false, err
	}

	oldScale, err := scaleFromStatefulSet(ss)
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

	ss.Spec.Replicas = scale.Spec.Replicas
	ss.ResourceVersion = scale.ResourceVersion
	ss, err = r.registry.UpdateStatefulSet(ctx, ss)
	if err != nil {
		return nil, false, err
	}
	newScale, err := scaleFromStatefulSet(ss)
	if err != nil {
		return nil, false, errors.NewBadRequest(fmt.Sprintf("%v", err))
	}
	return newScale, false, err
}

// scaleFromStatefulSet returns a scale subresource for a statefulset.
func scaleFromStatefulSet(ss *apps.StatefulSet) (*extensions.Scale, error) {
	return &extensions.Scale{
		// TODO: Create a variant of ObjectMeta type that only contains the fields below.
		ObjectMeta: metav1.ObjectMeta{
			Name:              ss.Name,
			Namespace:         ss.Namespace,
			UID:               ss.UID,
			ResourceVersion:   ss.ResourceVersion,
			CreationTimestamp: ss.CreationTimestamp,
		},
		Spec: extensions.ScaleSpec{
			Replicas: ss.Spec.Replicas,
		},
		Status: extensions.ScaleStatus{
			Replicas: ss.Status.Replicas,
			Selector: ss.Spec.Selector,
		},
	}, nil
}
