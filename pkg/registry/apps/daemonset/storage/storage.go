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
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/client-go/scale/scheme/autoscalingv1"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/apps/daemonset"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// DaemonSetStorage includes dummy storage for DaemonSets, and their Status and Scale subresource.
type DaemonSetStorage struct {
	DaemonSet *REST
	Status    *StatusREST
	Scale     *ScaleREST
}

// NewStorage returns new instance of DaemonSetStorage.
func NewStorage(optsGetter generic.RESTOptionsGetter) (DaemonSetStorage, error) {
	daemonSetRest, daemonSetStatusRest, err := NewREST(optsGetter)
	if err != nil {
		return DaemonSetStorage{}, err
	}

	return DaemonSetStorage{
		DaemonSet: daemonSetRest,
		Status:    daemonSetStatusRest,
		Scale:     &ScaleREST{store: daemonSetRest.Store},
	}, nil
}

// REST implements a RESTStorage for DaemonSets
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against DaemonSets.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, *StatusREST, error) {
	store := &genericregistry.Store{
		NewFunc:                   func() runtime.Object { return &apps.DaemonSet{} },
		NewListFunc:               func() runtime.Object { return &apps.DaemonSetList{} },
		DefaultQualifiedResource:  apps.Resource("daemonsets"),
		SingularQualifiedResource: apps.Resource("daemonset"),

		CreateStrategy:      daemonset.Strategy,
		UpdateStrategy:      daemonset.Strategy,
		DeleteStrategy:      daemonset.Strategy,
		ResetFieldsStrategy: daemonset.Strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, nil, err
	}

	statusStore := *store
	statusStore.UpdateStrategy = daemonset.StatusStrategy
	statusStore.ResetFieldsStrategy = daemonset.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}, nil
}

// Implement ShortNamesProvider
var _ rest.ShortNamesProvider = &REST{}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"ds"}
}

var _ rest.CategoriesProvider = &REST{}

// Categories implements the CategoriesProvider interface. Returns a list of categories a resource is part of.
func (r *REST) Categories() []string {
	return []string{"all"}
}

// StatusREST implements the REST endpoint for changing the status of a daemonset
type StatusREST struct {
	store *genericregistry.Store
}

// New creates a new DaemonSet object.
func (r *StatusREST) New() runtime.Object {
	return &apps.DaemonSet{}
}

// Destroy cleans up resources on shutdown.
func (r *StatusREST) Destroy() {
	// Given that underlying store is shared with REST,
	// we don't destroy it here explicitly.
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	// We are explicitly setting forceAllowCreate to false in the call to the underlying storage because
	// subresources should never allow create on update.
	return r.store.Update(ctx, name, objInfo, createValidation, updateValidation, false, options)
}

// GetResetFields implements rest.ResetFieldsStrategy
func (r *StatusREST) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return r.store.GetResetFields()
}

func (r *StatusREST) ConvertToTable(ctx context.Context, object runtime.Object, tableOptions runtime.Object) (*metav1.Table, error) {
	return r.store.ConvertToTable(ctx, object, tableOptions)
}

// ScaleREST implements a Scale for DaemonSet.
type ScaleREST struct {
	store *genericregistry.Store
}

// ScaleREST implements only Getter, /scale is read-only by design
var _ = rest.Getter(&ScaleREST{})
var _ = rest.GroupVersionKindProvider(&ScaleREST{})

// GroupVersionKind returns GroupVersionKind for DaemonSet Scale object
func (r *ScaleREST) GroupVersionKind(containingGV schema.GroupVersion) schema.GroupVersionKind {
	return autoscalingv1.SchemeGroupVersion.WithKind("Scale")
}

// New creates a new Scale object
func (r *ScaleREST) New() runtime.Object {
	return &autoscaling.Scale{}
}

// Get retrieves object from Scale storage.
func (r *ScaleREST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	obj, err := r.store.Get(ctx, name, options)
	if err != nil {
		return nil, apierrors.NewNotFound(apps.Resource("daemonsets/scale"), name)
	}
	ds := obj.(*apps.DaemonSet)
	scale, err := scaleFromDaemonSet(ds)
	if err != nil {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("%v", err))
	}
	return scale, err
}

// Destroy cleans up resources on shutdown.
func (r *ScaleREST) Destroy() {
	// Given that underlying store is shared with REST,
	// we don't destroy it here explicitly.
}

// scaleFromDaemonSet returns a scale subresource for a DaemonSet.
func scaleFromDaemonSet(ds *apps.DaemonSet) (*autoscaling.Scale, error) {
	selector, err := metav1.LabelSelectorAsSelector(ds.Spec.Selector)
	if err != nil {
		return nil, err
	}
	return &autoscaling.Scale{
		ObjectMeta: metav1.ObjectMeta{
			Name:              ds.Name,
			Namespace:         ds.Namespace,
			UID:               ds.UID,
			ResourceVersion:   ds.ResourceVersion,
			CreationTimestamp: ds.CreationTimestamp,
		},
		Spec: autoscaling.ScaleSpec{
			Replicas: ds.Status.DesiredNumberScheduled,
		},
		Status: autoscaling.ScaleStatus{
			Replicas: ds.Status.CurrentNumberScheduled,
			Selector: selector.String(),
		},
	}, nil
}
