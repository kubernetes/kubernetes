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
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/autoscaling/validation"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/core/replicationcontroller"
)

// ControllerStorage includes dummy storage for Replication Controllers and for Scale subresource.
type ControllerStorage struct {
	Controller *REST
	Status     *StatusREST
	Scale      *ScaleREST
}

func NewStorage(optsGetter generic.RESTOptionsGetter) ControllerStorage {
	controllerREST, statusREST := NewREST(optsGetter)
	controllerRegistry := replicationcontroller.NewRegistry(controllerREST)

	return ControllerStorage{
		Controller: controllerREST,
		Status:     statusREST,
		Scale:      &ScaleREST{registry: controllerRegistry},
	}
}

type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against replication controllers.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, *StatusREST) {
	store := &genericregistry.Store{
		Copier:      api.Scheme,
		NewFunc:     func() runtime.Object { return &api.ReplicationController{} },
		NewListFunc: func() runtime.Object { return &api.ReplicationControllerList{} },
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.ReplicationController).Name, nil
		},
		PredicateFunc:     replicationcontroller.MatchController,
		QualifiedResource: api.Resource("replicationcontrollers"),
		WatchCacheSize:    cachesize.GetWatchCacheSizeByResource("replicationcontrollers"),

		CreateStrategy: replicationcontroller.Strategy,
		UpdateStrategy: replicationcontroller.Strategy,
		DeleteStrategy: replicationcontroller.Strategy,
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: replicationcontroller.GetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}

	statusStore := *store
	statusStore.UpdateStrategy = replicationcontroller.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}
}

// Implement ShortNamesProvider
var _ rest.ShortNamesProvider = &REST{}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"rc"}
}

// StatusREST implements the REST endpoint for changing the status of a replication controller
type StatusREST struct {
	store *genericregistry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &api.ReplicationController{}
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
	registry replicationcontroller.Registry
}

// ScaleREST implements Patcher
var _ = rest.Patcher(&ScaleREST{})

// New creates a new Scale object
func (r *ScaleREST) New() runtime.Object {
	return &autoscaling.Scale{}
}

func (r *ScaleREST) Get(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	rc, err := r.registry.GetController(ctx, name, options)
	if err != nil {
		return nil, errors.NewNotFound(autoscaling.Resource("replicationcontrollers/scale"), name)
	}
	return scaleFromRC(rc), nil
}

func (r *ScaleREST) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	rc, err := r.registry.GetController(ctx, name, &metav1.GetOptions{})
	if err != nil {
		return nil, false, errors.NewNotFound(autoscaling.Resource("replicationcontrollers/scale"), name)
	}

	oldScale := scaleFromRC(rc)
	obj, err := objInfo.UpdatedObject(ctx, oldScale)
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
	rc, err = r.registry.UpdateController(ctx, rc)
	if err != nil {
		return nil, false, err
	}
	return scaleFromRC(rc), false, nil
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
