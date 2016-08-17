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

package etcd

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/autoscaling/validation"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/controller"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// ControllerStorage includes dummy storage for Replication Controllers and for Scale subresource.
type ControllerStorage struct {
	Controller *REST
	Status     *StatusREST
	Scale      *ScaleREST
}

func NewStorage(opts generic.RESTOptions) ControllerStorage {
	controllerREST, statusREST := NewREST(opts)
	controllerRegistry := controller.NewRegistry(controllerREST)

	return ControllerStorage{
		Controller: controllerREST,
		Status:     statusREST,
		Scale:      &ScaleREST{registry: controllerRegistry},
	}
}

type REST struct {
	*registry.Store
}

// NewREST returns a RESTStorage object that will work against replication controllers.
func NewREST(opts generic.RESTOptions) (*REST, *StatusREST) {
	prefix := "/" + opts.ResourcePrefix

	newListFunc := func() runtime.Object { return &api.ReplicationControllerList{} }
	storageInterface := opts.Decorator(
		opts.StorageConfig,
		cachesize.GetWatchCacheSizeByResource(cachesize.Controllers),
		&api.ReplicationController{},
		prefix,
		controller.Strategy,
		newListFunc,
		storage.NoTriggerPublisher,
	)

	store := &registry.Store{
		NewFunc: func() runtime.Object { return &api.ReplicationController{} },

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
		// Retrieve the name field of a replication controller
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.ReplicationController).Name, nil
		},
		// Used to match objects based on labels/fields for list and watch
		PredicateFunc:     controller.MatchController,
		QualifiedResource: api.Resource("replicationcontrollers"),

		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		// Used to validate controller creation
		CreateStrategy: controller.Strategy,

		// Used to validate controller updates
		UpdateStrategy: controller.Strategy,
		DeleteStrategy: controller.Strategy,

		Storage: storageInterface,
	}
	statusStore := *store
	statusStore.UpdateStrategy = controller.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}
}

// StatusREST implements the REST endpoint for changing the status of a replication controller
type StatusREST struct {
	store *registry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &api.ReplicationController{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx api.Context, name string) (runtime.Object, error) {
	return r.store.Get(ctx, name)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}

type ScaleREST struct {
	registry controller.Registry
}

// ScaleREST implements Patcher
var _ = rest.Patcher(&ScaleREST{})

// New creates a new Scale object
func (r *ScaleREST) New() runtime.Object {
	return &autoscaling.Scale{}
}

func (r *ScaleREST) Get(ctx api.Context, name string) (runtime.Object, error) {
	rc, err := r.registry.GetController(ctx, name)
	if err != nil {
		return nil, errors.NewNotFound(autoscaling.Resource("replicationcontrollers/scale"), name)
	}
	return scaleFromRC(rc), nil
}

func (r *ScaleREST) Update(ctx api.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	rc, err := r.registry.GetController(ctx, name)
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
		ObjectMeta: api.ObjectMeta{
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
