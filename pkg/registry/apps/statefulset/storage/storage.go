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

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/apis/apps"
	appsv1beta1 "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	appsv1beta2 "k8s.io/kubernetes/pkg/apis/apps/v1beta2"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	autoscalingv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	autoscalingvalidation "k8s.io/kubernetes/pkg/apis/autoscaling/validation"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/apps/statefulset"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// StatefulSetStorage includes dummy storage for StatefulSets, and their Status and Scale subresource.
type StatefulSetStorage struct {
	StatefulSet *REST
	Status      *StatusREST
	Scale       *ScaleREST
}

// ReplicasPathMappings returns the mappings between each group version and a replicas path
func ReplicasPathMappings() managedfields.ResourcePathMappings {
	return replicasPathInStatefulSet
}

// maps a group version to the replicas path in a statefulset object
var replicasPathInStatefulSet = managedfields.ResourcePathMappings{
	schema.GroupVersion{Group: "apps", Version: "v1beta1"}.String(): fieldpath.MakePathOrDie("spec", "replicas"),
	schema.GroupVersion{Group: "apps", Version: "v1beta2"}.String(): fieldpath.MakePathOrDie("spec", "replicas"),
	schema.GroupVersion{Group: "apps", Version: "v1"}.String():      fieldpath.MakePathOrDie("spec", "replicas"),
}

// NewStorage returns new instance of StatefulSetStorage.
func NewStorage(optsGetter generic.RESTOptionsGetter) (StatefulSetStorage, error) {
	statefulSetRest, statefulSetStatusRest, err := NewREST(optsGetter)
	if err != nil {
		return StatefulSetStorage{}, err
	}

	return StatefulSetStorage{
		StatefulSet: statefulSetRest,
		Status:      statefulSetStatusRest,
		Scale:       &ScaleREST{store: statefulSetRest.Store},
	}, nil
}

// REST implements a RESTStorage for statefulsets against etcd
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against statefulsets.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, *StatusREST, error) {
	store := &genericregistry.Store{
		NewFunc:                   func() runtime.Object { return &apps.StatefulSet{} },
		NewListFunc:               func() runtime.Object { return &apps.StatefulSetList{} },
		DefaultQualifiedResource:  apps.Resource("statefulsets"),
		SingularQualifiedResource: apps.Resource("statefulset"),

		CreateStrategy:      statefulset.Strategy,
		UpdateStrategy:      statefulset.Strategy,
		DeleteStrategy:      statefulset.Strategy,
		ResetFieldsStrategy: statefulset.Strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, nil, err
	}

	statusStore := *store
	statusStore.UpdateStrategy = statefulset.StatusStrategy
	statusStore.ResetFieldsStrategy = statefulset.StatusStrategy
	return &REST{store}, &StatusREST{store: &statusStore}, nil
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

// New returns empty StatefulSet object.
func (r *StatusREST) New() runtime.Object {
	return &apps.StatefulSet{}
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

// Implement ShortNamesProvider
var _ rest.ShortNamesProvider = &REST{}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"sts"}
}

// ScaleREST implements a Scale for Deployment.
type ScaleREST struct {
	store *genericregistry.Store
}

// ScaleREST implements Patcher
var _ = rest.Patcher(&ScaleREST{})
var _ = rest.GroupVersionKindProvider(&ScaleREST{})

// GroupVersionKind returns GroupVersionKind for StatefulSet Scale object
func (r *ScaleREST) GroupVersionKind(containingGV schema.GroupVersion) schema.GroupVersionKind {
	switch containingGV {
	case appsv1beta1.SchemeGroupVersion:
		return appsv1beta1.SchemeGroupVersion.WithKind("Scale")
	case appsv1beta2.SchemeGroupVersion:
		return appsv1beta2.SchemeGroupVersion.WithKind("Scale")
	default:
		return autoscalingv1.SchemeGroupVersion.WithKind("Scale")
	}
}

// New creates a new Scale object
func (r *ScaleREST) New() runtime.Object {
	return &autoscaling.Scale{}
}

// Destroy cleans up resources on shutdown.
func (r *ScaleREST) Destroy() {
	// Given that underlying store is shared with REST,
	// we don't destroy it here explicitly.
}

// Get retrieves object from Scale storage.
func (r *ScaleREST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	obj, err := r.store.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}
	ss := obj.(*apps.StatefulSet)
	scale, err := scaleFromStatefulSet(ss)
	if err != nil {
		return nil, errors.NewBadRequest(fmt.Sprintf("%v", err))
	}
	return scale, err
}

// Update alters scale subset of StatefulSet object.
func (r *ScaleREST) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	obj, _, err := r.store.Update(
		ctx,
		name,
		&scaleUpdatedObjectInfo{name, objInfo},
		toScaleCreateValidation(createValidation),
		toScaleUpdateValidation(updateValidation),
		false,
		options,
	)
	if err != nil {
		return nil, false, err
	}
	ss := obj.(*apps.StatefulSet)
	newScale, err := scaleFromStatefulSet(ss)
	if err != nil {
		return nil, false, errors.NewBadRequest(fmt.Sprintf("%v", err))
	}
	return newScale, false, err
}

func (r *ScaleREST) ConvertToTable(ctx context.Context, object runtime.Object, tableOptions runtime.Object) (*metav1.Table, error) {
	return r.store.ConvertToTable(ctx, object, tableOptions)
}

func toScaleCreateValidation(f rest.ValidateObjectFunc) rest.ValidateObjectFunc {
	return func(ctx context.Context, obj runtime.Object) error {
		scale, err := scaleFromStatefulSet(obj.(*apps.StatefulSet))
		if err != nil {
			return err
		}
		return f(ctx, scale)
	}
}

func toScaleUpdateValidation(f rest.ValidateObjectUpdateFunc) rest.ValidateObjectUpdateFunc {
	return func(ctx context.Context, obj, old runtime.Object) error {
		newScale, err := scaleFromStatefulSet(obj.(*apps.StatefulSet))
		if err != nil {
			return err
		}
		oldScale, err := scaleFromStatefulSet(old.(*apps.StatefulSet))
		if err != nil {
			return err
		}
		return f(ctx, newScale, oldScale)
	}
}

// scaleFromStatefulSet returns a scale subresource for a statefulset.
func scaleFromStatefulSet(ss *apps.StatefulSet) (*autoscaling.Scale, error) {
	selector, err := metav1.LabelSelectorAsSelector(ss.Spec.Selector)
	if err != nil {
		return nil, err
	}
	return &autoscaling.Scale{
		// TODO: Create a variant of ObjectMeta type that only contains the fields below.
		ObjectMeta: metav1.ObjectMeta{
			Name:              ss.Name,
			Namespace:         ss.Namespace,
			UID:               ss.UID,
			ResourceVersion:   ss.ResourceVersion,
			CreationTimestamp: ss.CreationTimestamp,
		},
		Spec: autoscaling.ScaleSpec{
			Replicas: ss.Spec.Replicas,
		},
		Status: autoscaling.ScaleStatus{
			Replicas: ss.Status.Replicas,
			Selector: selector.String(),
		},
	}, nil
}

// scaleUpdatedObjectInfo transforms existing statefulset -> existing scale -> new scale -> new statefulset
type scaleUpdatedObjectInfo struct {
	name       string
	reqObjInfo rest.UpdatedObjectInfo
}

func (i *scaleUpdatedObjectInfo) Preconditions() *metav1.Preconditions {
	return i.reqObjInfo.Preconditions()
}

func (i *scaleUpdatedObjectInfo) UpdatedObject(ctx context.Context, oldObj runtime.Object) (runtime.Object, error) {
	statefulset, ok := oldObj.DeepCopyObject().(*apps.StatefulSet)
	if !ok {
		return nil, errors.NewBadRequest(fmt.Sprintf("expected existing object type to be StatefulSet, got %T", statefulset))
	}
	// if zero-value, the existing object does not exist
	if len(statefulset.ResourceVersion) == 0 {
		return nil, errors.NewNotFound(apps.Resource("statefulsets/scale"), i.name)
	}

	groupVersion := schema.GroupVersion{Group: "apps", Version: "v1"}
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		requestGroupVersion := schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
		if _, ok := replicasPathInStatefulSet[requestGroupVersion.String()]; ok {
			groupVersion = requestGroupVersion
		} else {
			klog.Fatalf("Unrecognized group/version in request info %q", requestGroupVersion.String())
		}
	}

	managedFieldsHandler := managedfields.NewScaleHandler(
		statefulset.ManagedFields,
		groupVersion,
		replicasPathInStatefulSet,
	)

	// statefulset -> old scale
	oldScale, err := scaleFromStatefulSet(statefulset)
	if err != nil {
		return nil, err
	}
	scaleManagedFields, err := managedFieldsHandler.ToSubresource()
	if err != nil {
		return nil, err
	}
	oldScale.ManagedFields = scaleManagedFields

	// old scale -> new scale
	newScaleObj, err := i.reqObjInfo.UpdatedObject(ctx, oldScale)
	if err != nil {
		return nil, err
	}
	if newScaleObj == nil {
		return nil, errors.NewBadRequest("nil update passed to Scale")
	}
	scale, ok := newScaleObj.(*autoscaling.Scale)
	if !ok {
		return nil, errors.NewBadRequest(fmt.Sprintf("expected input object type to be Scale, but %T", newScaleObj))
	}

	// validate
	if errs := autoscalingvalidation.ValidateScale(scale); len(errs) > 0 {
		return nil, errors.NewInvalid(autoscaling.Kind("Scale"), statefulset.Name, errs)
	}

	// validate precondition if specified (resourceVersion matching is handled by storage)
	if len(scale.UID) > 0 && scale.UID != statefulset.UID {
		return nil, errors.NewConflict(
			apps.Resource("statefulsets/scale"),
			statefulset.Name,
			fmt.Errorf("Precondition failed: UID in precondition: %v, UID in object meta: %v", scale.UID, statefulset.UID),
		)
	}

	// move replicas/resourceVersion fields to object and return
	statefulset.Spec.Replicas = scale.Spec.Replicas
	statefulset.ResourceVersion = scale.ResourceVersion

	updatedEntries, err := managedFieldsHandler.ToParent(scale.ManagedFields)
	if err != nil {
		return nil, err
	}
	statefulset.ManagedFields = updatedEntries

	return statefulset, nil
}
