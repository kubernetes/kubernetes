/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extvalidation "k8s.io/kubernetes/pkg/apis/extensions/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/controller"
	"k8s.io/kubernetes/pkg/registry/generic"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// ControllerStorage includes dummy storage for Replication Controllers and for Scale subresource.
type ControllerStorage struct {
	Controller *REST
	Status     *StatusREST
	Scale      *ScaleREST
}

func NewStorage(s storage.Interface, storageDecorator generic.StorageDecorator) ControllerStorage {
	controllerREST, statusREST := NewREST(s, storageDecorator)
	controllerRegistry := controller.NewRegistry(controllerREST)

	return ControllerStorage{
		Controller: controllerREST,
		Status:     statusREST,
		Scale:      &ScaleREST{registry: &controllerRegistry},
	}
}

type REST struct {
	*etcdgeneric.Etcd
}

// NewREST returns a RESTStorage object that will work against replication controllers.
func NewREST(s storage.Interface, storageDecorator generic.StorageDecorator) (*REST, *StatusREST) {
	prefix := "/controllers"

	newListFunc := func() runtime.Object { return &api.ReplicationControllerList{} }
	storageInterface := storageDecorator(
		s, 100, &api.ReplicationController{}, prefix, controller.Strategy, newListFunc)

	store := &etcdgeneric.Etcd{
		NewFunc: func() runtime.Object { return &api.ReplicationController{} },

		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: newListFunc,
		// Produces a path that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix
		KeyRootFunc: func(ctx api.Context) string {
			return etcdgeneric.NamespaceKeyRootFunc(ctx, prefix)
		},
		// Produces a path that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return etcdgeneric.NamespaceKeyFunc(ctx, prefix, name)
		},
		// Retrieve the name field of a replication controller
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.ReplicationController).Name, nil
		},
		// Used to match objects based on labels/fields for list and watch
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return controller.MatchController(label, field)
		},
		QualifiedResource: api.Resource("replicationcontrollers"),

		// Used to validate controller creation
		CreateStrategy: controller.Strategy,

		// Used to validate controller updates
		UpdateStrategy: controller.Strategy,

		Storage: storageInterface,
	}
	statusStore := *store
	statusStore.UpdateStrategy = controller.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}
}

// StatusREST implements the REST endpoint for changing the status of a replication controller
type StatusREST struct {
	store *etcdgeneric.Etcd
}

func (r *StatusREST) New() runtime.Object {
	return &api.ReplicationController{}
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	return r.store.Update(ctx, obj)
}

type ScaleREST struct {
	registry *controller.Registry
}

// ScaleREST implements Patcher
var _ = rest.Patcher(&ScaleREST{})

// New creates a new Scale object
func (r *ScaleREST) New() runtime.Object {
	return &extensions.ScaleTwo{}
}

func (r *ScaleREST) Get(ctx api.Context, name string) (runtime.Object, error) {
	rc, err := (*r.registry).GetController(ctx, name)
	if err != nil {
		return nil, errors.NewNotFound(extensions.Resource("replicationcontrollers/scale"), name)
	}
	return &extensions.ScaleTwo{
		ObjectMeta: api.ObjectMeta{
			Name:              name,
			Namespace:         rc.Namespace,
			CreationTimestamp: rc.CreationTimestamp,
		},
		Spec: extensions.ScaleSpec{
			Replicas: rc.Spec.Replicas,
		},
		Status: extensions.ScaleTwoStatus{
			Replicas: rc.Status.Replicas,
			Selector: labels.SelectorFromSet(rc.Spec.Selector).String(),
		},
	}, nil
}

func (r *ScaleREST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	if obj == nil {
		return nil, false, errors.NewBadRequest(fmt.Sprintf("nil update passed to Scale"))
	}
	scale, ok := obj.(*extensions.ScaleTwo)
	if !ok {
		return nil, false, errors.NewBadRequest(fmt.Sprintf("wrong object passed to Scale update: %v", obj))
	}

	if errs := extvalidation.ValidateScaleTwo(scale); len(errs) > 0 {
		return nil, false, errors.NewInvalid(extensions.Kind("ScaleTwo"), scale.Name, errs)
	}

	rc, err := (*r.registry).GetController(ctx, scale.Name)
	if err != nil {
		return nil, false, errors.NewNotFound(extensions.Resource("replicationcontrollers/scale"), scale.Name)
	}
	rc.Spec.Replicas = scale.Spec.Replicas
	rc, err = (*r.registry).UpdateController(ctx, rc)
	if err != nil {
		return nil, false, errors.NewConflict(extensions.Resource("replicationcontrollers/scale"), scale.Name, err)
	}
	return &extensions.ScaleTwo{
		ObjectMeta: api.ObjectMeta{
			Name:              rc.Name,
			Namespace:         rc.Namespace,
			CreationTimestamp: rc.CreationTimestamp,
		},
		Spec: extensions.ScaleSpec{
			Replicas: rc.Spec.Replicas,
		},
		Status: extensions.ScaleTwoStatus{
			Replicas: rc.Status.Replicas,
			Selector: labels.SelectorFromSet(rc.Spec.Selector).String(),
		},
	}, false, nil
}
