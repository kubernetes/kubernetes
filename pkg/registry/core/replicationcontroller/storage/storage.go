/*
Copyright 2014 The Kubernetes Authors.

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

// If you make changes to this file, you should also make the corresponding change in ReplicaSet.

package storage

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	autoscalingv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	"k8s.io/kubernetes/pkg/apis/autoscaling/validation"
	api "k8s.io/kubernetes/pkg/apis/core"
	extensionsv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/core/replicationcontroller"
)

// ControllerStorage includes dummy storage for Replication Controllers and for Scale subresource.
type ControllerStorage struct {
	Controller *REST
	Status     *StatusREST
	Scale      *ScaleREST
}

func NewStorage(optsGetter generic.RESTOptionsGetter) (ControllerStorage, error) {
	controllerREST, statusREST, err := NewREST(optsGetter)
	if err != nil {
		return ControllerStorage{}, err
	}

	return ControllerStorage{
		Controller: controllerREST,
		Status:     statusREST,
		Scale:      &ScaleREST{store: controllerREST.Store},
	}, nil
}

type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against replication controllers.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, *StatusREST, error) {
	store := &genericregistry.Store{
		NewFunc:                  func() runtime.Object { return &api.ReplicationController{} },
		NewListFunc:              func() runtime.Object { return &api.ReplicationControllerList{} },
		PredicateFunc:            replicationcontroller.MatchController,
		DefaultQualifiedResource: api.Resource("replicationcontrollers"),

		CreateStrategy: replicationcontroller.Strategy,
		UpdateStrategy: replicationcontroller.Strategy,
		DeleteStrategy: replicationcontroller.Strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: replicationcontroller.GetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, nil, err
	}

	statusStore := *store
	statusStore.UpdateStrategy = replicationcontroller.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}, nil
}

// Implement ShortNamesProvider
var _ rest.ShortNamesProvider = &REST{}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"rc"}
}

// Implement CategoriesProvider
var _ rest.CategoriesProvider = &REST{}

// Categories implements the CategoriesProvider interface. Returns a list of categories a resource is part of.
func (r *REST) Categories() []string {
	return []string{"all"}
}

// StatusREST implements the REST endpoint for changing the status of a replication controller
type StatusREST struct {
	store *genericregistry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &api.ReplicationController{}
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

type ScaleREST struct {
	store *genericregistry.Store
}

// ScaleREST implements Patcher
var _ = rest.Patcher(&ScaleREST{})
var _ = rest.GroupVersionKindProvider(&ScaleREST{})

func (r *ScaleREST) GroupVersionKind(containingGV schema.GroupVersion) schema.GroupVersionKind {
	switch containingGV {
	case extensionsv1beta1.SchemeGroupVersion:
		return extensionsv1beta1.SchemeGroupVersion.WithKind("Scale")
	default:
		return autoscalingv1.SchemeGroupVersion.WithKind("Scale")
	}
}

// New creates a new Scale object
func (r *ScaleREST) New() runtime.Object {
	return &autoscaling.Scale{}
}

func (r *ScaleREST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	obj, err := r.store.Get(ctx, name, options)
	if err != nil {
		return nil, errors.NewNotFound(autoscaling.Resource("replicationcontrollers/scale"), name)
	}
	rc := obj.(*api.ReplicationController)
	return scaleFromRC(rc), nil
}

func (r *ScaleREST) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	obj, err := r.store.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		return nil, false, errors.NewNotFound(autoscaling.Resource("replicationcontrollers/scale"), name)
	}
	rc := obj.(*api.ReplicationController)

	oldScale := scaleFromRC(rc)
	// TODO: should this pass validation?
	obj, err = objInfo.UpdatedObject(ctx, oldScale)
	if err != nil {
		return nil, false, err
	}

	if obj == nil {
		return nil, false, errors.NewBadRequest("nil update passed to Scale")
	}
	scale, ok := obj.(*autoscaling.Scale)
	if !ok {
		return nil, false, errors.NewBadRequest(fmt.Sprintf("wrong object passed to Scale update: %v", obj))
	}

	if errs := validation.ValidateScale(scale); len(errs) > 0 {
		return nil, false, errors.NewInvalid(autoscaling.Kind("Scale"), scale.Name, errs)
	}

	rc.Spec.Replicas = scale.Spec.Replicas
	rc.ResourceVersion = scale.ResourceVersion
	obj, _, err = r.store.Update(
		ctx,
		rc.Name,
		rest.DefaultUpdatedObjectInfo(rc),
		toScaleCreateValidation(createValidation),
		toScaleUpdateValidation(updateValidation),
		false,
		options,
	)
	if err != nil {
		return nil, false, err
	}
	rc = obj.(*api.ReplicationController)
	return scaleFromRC(rc), false, nil
}

func toScaleCreateValidation(f rest.ValidateObjectFunc) rest.ValidateObjectFunc {
	return func(ctx context.Context, obj runtime.Object) error {
		return f(ctx, scaleFromRC(obj.(*api.ReplicationController)))
	}
}

func toScaleUpdateValidation(f rest.ValidateObjectUpdateFunc) rest.ValidateObjectUpdateFunc {
	return func(ctx context.Context, obj, old runtime.Object) error {
		return f(
			ctx,
			scaleFromRC(obj.(*api.ReplicationController)),
			scaleFromRC(old.(*api.ReplicationController)),
		)
	}
}

// scaleFromRC returns a scale subresource for a replication controller.
func scaleFromRC(rc *api.ReplicationController) *autoscaling.Scale {
	return &autoscaling.Scale{
		ObjectMeta: metav1.ObjectMeta{
			Name:              rc.Name,
			Namespace:         rc.Namespace,
			UID:               rc.UID,
			ResourceVersion:   rc.ResourceVersion,
			CreationTimestamp: rc.CreationTimestamp,
		},
		Spec: autoscaling.ScaleSpec{
			Replicas: rc.Spec.Replicas,
		},
		Status: autoscaling.ScaleStatus{
			Replicas: rc.Status.Replicas,
			Selector: labels.SelectorFromSet(rc.Spec.Selector).String(),
		},
	}
}
