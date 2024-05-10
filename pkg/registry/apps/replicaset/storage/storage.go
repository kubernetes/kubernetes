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
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/apis/apps"
	appsv1beta1 "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	appsv1beta2 "k8s.io/kubernetes/pkg/apis/apps/v1beta2"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	autoscalingv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	autoscalingvalidation "k8s.io/kubernetes/pkg/apis/autoscaling/validation"
	extensionsv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/apps/replicaset"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// ReplicaSetStorage includes dummy storage for ReplicaSets and for Scale subresource.
type ReplicaSetStorage struct {
	ReplicaSet *REST
	Status     *StatusREST
	Scale      *ScaleREST
}

// ReplicasPathMappings returns the mappings between each group version and a replicas path
func ReplicasPathMappings() managedfields.ResourcePathMappings {
	return replicasPathInReplicaSet
}

// maps a group version to the replicas path in a replicaset object
var replicasPathInReplicaSet = managedfields.ResourcePathMappings{
	schema.GroupVersion{Group: "apps", Version: "v1beta2"}.String(): fieldpath.MakePathOrDie("spec", "replicas"),
	schema.GroupVersion{Group: "apps", Version: "v1"}.String():      fieldpath.MakePathOrDie("spec", "replicas"),
}

// NewStorage returns new instance of ReplicaSetStorage.
func NewStorage(optsGetter generic.RESTOptionsGetter) (ReplicaSetStorage, error) {
	replicaSetRest, replicaSetStatusRest, err := NewREST(optsGetter)
	if err != nil {
		return ReplicaSetStorage{}, err
	}

	return ReplicaSetStorage{
		ReplicaSet: replicaSetRest,
		Status:     replicaSetStatusRest,
		Scale:      &ScaleREST{store: replicaSetRest.Store},
	}, nil
}

// REST implements a RESTStorage for ReplicaSet.
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against ReplicaSet.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, *StatusREST, error) {
	store := &genericregistry.Store{
		NewFunc:                   func() runtime.Object { return &apps.ReplicaSet{} },
		NewListFunc:               func() runtime.Object { return &apps.ReplicaSetList{} },
		PredicateFunc:             storage.PredicateFuncFromMatcherFunc(apps.ReplicaSetMatcher),
		DefaultQualifiedResource:  apps.Resource("replicasets"),
		SingularQualifiedResource: apps.Resource("replicaset"),

		CreateStrategy:      replicaset.Strategy,
		UpdateStrategy:      replicaset.Strategy,
		DeleteStrategy:      replicaset.Strategy,
		ResetFieldsStrategy: replicaset.Strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: apps.ReplicaSetGetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, nil, err
	}

	statusStore := *store
	statusStore.UpdateStrategy = replicaset.StatusStrategy
	statusStore.ResetFieldsStrategy = replicaset.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}, nil
}

// Implement ShortNamesProvider
var _ rest.ShortNamesProvider = &REST{}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"rs"}
}

// Implement CategoriesProvider
var _ rest.CategoriesProvider = &REST{}

// Categories implements the CategoriesProvider interface. Returns a list of categories a resource is part of.
func (r *REST) Categories() []string {
	return []string{"all"}
}

// StatusREST implements the REST endpoint for changing the status of a ReplicaSet
type StatusREST struct {
	store *genericregistry.Store
}

// New returns empty ReplicaSet object.
func (r *StatusREST) New() runtime.Object {
	return &apps.ReplicaSet{}
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

// ScaleREST implements a Scale for ReplicaSet.
type ScaleREST struct {
	store *genericregistry.Store
}

// ScaleREST implements Patcher
var _ = rest.Patcher(&ScaleREST{})
var _ = rest.GroupVersionKindProvider(&ScaleREST{})

// GroupVersionKind returns GroupVersionKind for ReplicaSet Scale object
func (r *ScaleREST) GroupVersionKind(containingGV schema.GroupVersion) schema.GroupVersionKind {
	switch containingGV {
	case extensionsv1beta1.SchemeGroupVersion:
		return extensionsv1beta1.SchemeGroupVersion.WithKind("Scale")
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
		return nil, errors.NewNotFound(apps.Resource("replicasets/scale"), name)
	}
	rs := obj.(*apps.ReplicaSet)
	scale, err := scaleFromReplicaSet(rs)
	if err != nil {
		return nil, errors.NewBadRequest(fmt.Sprintf("%v", err))
	}
	return scale, err
}

// Update alters scale subset of ReplicaSet object.
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
	rs := obj.(*apps.ReplicaSet)
	newScale, err := scaleFromReplicaSet(rs)
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
		scale, err := scaleFromReplicaSet(obj.(*apps.ReplicaSet))
		if err != nil {
			return err
		}
		return f(ctx, scale)
	}
}

func toScaleUpdateValidation(f rest.ValidateObjectUpdateFunc) rest.ValidateObjectUpdateFunc {
	return func(ctx context.Context, obj, old runtime.Object) error {
		newScale, err := scaleFromReplicaSet(obj.(*apps.ReplicaSet))
		if err != nil {
			return err
		}
		oldScale, err := scaleFromReplicaSet(old.(*apps.ReplicaSet))
		if err != nil {
			return err
		}
		return f(ctx, newScale, oldScale)
	}
}

// scaleFromReplicaSet returns a scale subresource for a replica set.
func scaleFromReplicaSet(rs *apps.ReplicaSet) (*autoscaling.Scale, error) {
	selector, err := metav1.LabelSelectorAsSelector(rs.Spec.Selector)
	if err != nil {
		return nil, err
	}
	return &autoscaling.Scale{
		// TODO: Create a variant of ObjectMeta type that only contains the fields below.
		ObjectMeta: metav1.ObjectMeta{
			Name:              rs.Name,
			Namespace:         rs.Namespace,
			UID:               rs.UID,
			ResourceVersion:   rs.ResourceVersion,
			CreationTimestamp: rs.CreationTimestamp,
		},
		Spec: autoscaling.ScaleSpec{
			Replicas: rs.Spec.Replicas,
		},
		Status: autoscaling.ScaleStatus{
			Replicas: rs.Status.Replicas,
			Selector: selector.String(),
		},
	}, nil
}

// scaleUpdatedObjectInfo transforms existing replicaset -> existing scale -> new scale -> new replicaset
type scaleUpdatedObjectInfo struct {
	name       string
	reqObjInfo rest.UpdatedObjectInfo
}

func (i *scaleUpdatedObjectInfo) Preconditions() *metav1.Preconditions {
	return i.reqObjInfo.Preconditions()
}

func (i *scaleUpdatedObjectInfo) UpdatedObject(ctx context.Context, oldObj runtime.Object) (runtime.Object, error) {
	replicaset, ok := oldObj.DeepCopyObject().(*apps.ReplicaSet)
	if !ok {
		return nil, errors.NewBadRequest(fmt.Sprintf("expected existing object type to be ReplicaSet, got %T", replicaset))
	}
	// if zero-value, the existing object does not exist
	if len(replicaset.ResourceVersion) == 0 {
		return nil, errors.NewNotFound(apps.Resource("replicasets/scale"), i.name)
	}

	groupVersion := schema.GroupVersion{Group: "apps", Version: "v1"}
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		requestGroupVersion := schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
		if _, ok := replicasPathInReplicaSet[requestGroupVersion.String()]; ok {
			groupVersion = requestGroupVersion
		} else {
			klog.Fatalf("Unrecognized group/version in request info %q", requestGroupVersion.String())
		}
	}

	managedFieldsHandler := managedfields.NewScaleHandler(
		replicaset.ManagedFields,
		groupVersion,
		replicasPathInReplicaSet,
	)

	// replicaset -> old scale
	oldScale, err := scaleFromReplicaSet(replicaset)
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
		return nil, errors.NewInvalid(autoscaling.Kind("Scale"), replicaset.Name, errs)
	}

	// validate precondition if specified (resourceVersion matching is handled by storage)
	if len(scale.UID) > 0 && scale.UID != replicaset.UID {
		return nil, errors.NewConflict(
			apps.Resource("replicasets/scale"),
			replicaset.Name,
			fmt.Errorf("Precondition failed: UID in precondition: %v, UID in object meta: %v", scale.UID, replicaset.UID),
		)
	}

	// move replicas/resourceVersion fields to object and return
	replicaset.Spec.Replicas = scale.Spec.Replicas
	replicaset.ResourceVersion = scale.ResourceVersion

	updatedEntries, err := managedFieldsHandler.ToParent(scale.ManagedFields)
	if err != nil {
		return nil, err
	}
	replicaset.ManagedFields = updatedEntries

	return replicaset, nil
}
