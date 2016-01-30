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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/deployment"
	"k8s.io/kubernetes/pkg/registry/generic"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"

	extvalidation "k8s.io/kubernetes/pkg/apis/extensions/validation"
)

// DeploymentStorage includes dummy storage for Deployments and for Scale subresource.
type DeploymentStorage struct {
	Deployment *REST
	Status     *StatusREST
	Scale      *ScaleREST
}

func NewStorage(s storage.Interface, storageDecorator generic.StorageDecorator) DeploymentStorage {
	deploymentRest, deploymentStatusRest := NewREST(s, storageDecorator)
	deploymentRegistry := deployment.NewRegistry(deploymentRest)

	return DeploymentStorage{
		Deployment: deploymentRest,
		Status:     deploymentStatusRest,
		Scale:      &ScaleREST{registry: &deploymentRegistry},
	}
}

type REST struct {
	*etcdgeneric.Etcd
}

// NewREST returns a RESTStorage object that will work against deployments.
func NewREST(s storage.Interface, storageDecorator generic.StorageDecorator) (*REST, *StatusREST) {
	prefix := "/deployments"

	newListFunc := func() runtime.Object { return &extensions.DeploymentList{} }
	storageInterface := storageDecorator(
		s, 100, &extensions.Deployment{}, prefix, deployment.Strategy, newListFunc)

	store := &etcdgeneric.Etcd{
		NewFunc: func() runtime.Object { return &extensions.Deployment{} },
		// NewListFunc returns an object capable of storing results of an etcd list.
		NewListFunc: newListFunc,
		// Produces a path that etcd understands, to the root of the resource
		// by combining the namespace in the context with the given prefix.
		KeyRootFunc: func(ctx api.Context) string {
			return etcdgeneric.NamespaceKeyRootFunc(ctx, prefix)
		},
		// Produces a path that etcd understands, to the resource by combining
		// the namespace in the context with the given prefix.
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return etcdgeneric.NamespaceKeyFunc(ctx, prefix, name)
		},
		// Retrieve the name field of a deployment.
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*extensions.Deployment).Name, nil
		},
		// Used to match objects based on labels/fields for list.
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return deployment.MatchDeployment(label, field)
		},
		QualifiedResource: extensions.Resource("deployments"),

		// Used to validate deployment creation.
		CreateStrategy: deployment.Strategy,

		// Used to validate deployment updates.
		UpdateStrategy: deployment.Strategy,

		Storage: storageInterface,
	}
	statusStore := *store
	statusStore.UpdateStrategy = deployment.StatusStrategy
	return &REST{store}, &StatusREST{store: &statusStore}
}

// StatusREST implements the REST endpoint for changing the status of a deployment
type StatusREST struct {
	store *etcdgeneric.Etcd
}

func (r *StatusREST) New() runtime.Object {
	return &extensions.Deployment{}
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	return r.store.Update(ctx, obj)
}

type ScaleREST struct {
	registry *deployment.Registry
}

// ScaleREST implements Patcher
var _ = rest.Patcher(&ScaleREST{})

// New creates a new Scale object
func (r *ScaleREST) New() runtime.Object {
	return &extensions.Scale{}
}

func (r *ScaleREST) Get(ctx api.Context, name string) (runtime.Object, error) {
	deployment, err := (*r.registry).GetDeployment(ctx, name)
	if err != nil {
		return nil, errors.NewNotFound(extensions.Resource("deployments/scale"), name)
	}
	return extensions.ScaleFromDeployment(deployment), nil
}

func (r *ScaleREST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
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

	deployment, err := (*r.registry).GetDeployment(ctx, scale.Name)
	if err != nil {
		return nil, false, errors.NewNotFound(extensions.Resource("deployments/scale"), scale.Name)
	}
	deployment.Spec.Replicas = scale.Spec.Replicas
	deployment, err = (*r.registry).UpdateDeployment(ctx, deployment)
	if err != nil {
		return nil, false, errors.NewConflict(extensions.Resource("deployments/scale"), scale.Name, err)
	}
	return extensions.ScaleFromDeployment(deployment), false, nil
}
