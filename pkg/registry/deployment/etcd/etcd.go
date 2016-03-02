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
	etcderr "k8s.io/kubernetes/pkg/api/errors/etcd"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extvalidation "k8s.io/kubernetes/pkg/apis/extensions/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/deployment"
	"k8s.io/kubernetes/pkg/registry/generic"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// DeploymentStorage includes dummy storage for Deployments and for Scale subresource.
type DeploymentStorage struct {
	Deployment *REST
	Status     *StatusREST
	Scale      *ScaleREST
	Rollback   *RollbackREST
}

func NewStorage(opts generic.RESTOptions) DeploymentStorage {
	deploymentRest, deploymentStatusRest, deploymentRollbackRest := NewREST(opts)
	deploymentRegistry := deployment.NewRegistry(deploymentRest)

	return DeploymentStorage{
		Deployment: deploymentRest,
		Status:     deploymentStatusRest,
		Scale:      &ScaleREST{registry: &deploymentRegistry},
		Rollback:   deploymentRollbackRest,
	}
}

type REST struct {
	*etcdgeneric.Etcd
}

// NewREST returns a RESTStorage object that will work against deployments.
func NewREST(opts generic.RESTOptions) (*REST, *StatusREST, *RollbackREST) {
	prefix := "/deployments"

	newListFunc := func() runtime.Object { return &extensions.DeploymentList{} }
	storageInterface := opts.Decorator(
		opts.Storage, cachesize.GetWatchCacheSizeByResource(cachesize.Deployments), &extensions.Deployment{}, prefix, deployment.Strategy, newListFunc)

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
		QualifiedResource:       extensions.Resource("deployments"),
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		// Used to validate deployment creation.
		CreateStrategy: deployment.Strategy,

		// Used to validate deployment updates.
		UpdateStrategy: deployment.Strategy,

		Storage: storageInterface,
	}
	statusStore := *store
	statusStore.UpdateStrategy = deployment.StatusStrategy
	return &REST{store}, &StatusREST{store: &statusStore}, &RollbackREST{store: store}
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

// RollbackREST implements the REST endpoint for initiating the rollback of a deployment
type RollbackREST struct {
	store *etcdgeneric.Etcd
}

// New creates a rollback
func (r *RollbackREST) New() runtime.Object {
	return &extensions.DeploymentRollback{}
}

var _ = rest.Creater(&RollbackREST{})

func (r *RollbackREST) Create(ctx api.Context, obj runtime.Object) (out runtime.Object, err error) {
	rollback, ok := obj.(*extensions.DeploymentRollback)
	if !ok {
		return nil, fmt.Errorf("expected input object type to be DeploymentRollback, but %T", obj)
	}

	if errs := extvalidation.ValidateDeploymentRollback(rollback); len(errs) != 0 {
		return nil, errors.NewInvalid(extensions.Kind("DeploymentRollback"), rollback.Name, errs)
	}

	// Update the Deployment with information in DeploymentRollback to trigger rollback
	err = r.rollbackDeployment(ctx, rollback.Name, &rollback.RollbackTo, rollback.UpdatedAnnotations)
	return
}

func (r *RollbackREST) rollbackDeployment(ctx api.Context, deploymentID string, config *extensions.RollbackConfig, annotations map[string]string) (err error) {
	if _, err = r.setDeploymentRollback(ctx, deploymentID, config, annotations); err != nil {
		err = etcderr.InterpretGetError(err, extensions.Resource("deployments"), deploymentID)
		err = etcderr.InterpretUpdateError(err, extensions.Resource("deployments"), deploymentID)
		if _, ok := err.(*errors.StatusError); !ok {
			err = errors.NewConflict(extensions.Resource("deployments/rollback"), deploymentID, err)
		}
	}
	return
}

func (r *RollbackREST) setDeploymentRollback(ctx api.Context, deploymentID string, config *extensions.RollbackConfig, annotations map[string]string) (finalDeployment *extensions.Deployment, err error) {
	dKey, err := r.store.KeyFunc(ctx, deploymentID)
	if err != nil {
		return nil, err
	}
	err = r.store.Storage.GuaranteedUpdate(ctx, dKey, &extensions.Deployment{}, false, storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
		d, ok := obj.(*extensions.Deployment)
		if !ok {
			return nil, fmt.Errorf("unexpected object: %#v", obj)
		}
		if d.Annotations == nil {
			d.Annotations = make(map[string]string)
		}
		for k, v := range annotations {
			d.Annotations[k] = v
		}
		d.Spec.RollbackTo = config
		finalDeployment = d
		return d, nil
	}))
	return finalDeployment, err
}

type ScaleREST struct {
	registry *deployment.Registry
}

// TODO(madhusudancs): Fix this when Scale group issues are resolved (see issue #18528).

//  // ScaleREST implements Patcher
// var _ = rest.Patcher(&ScaleREST{})

// // New creates a new Scale object
// func (r *ScaleREST) New() runtime.Object {
// 	return &extensions.Scale{}
// }

// func (r *ScaleREST) Get(ctx api.Context, name string) (runtime.Object, error) {
// 	deployment, err := (*r.registry).GetDeployment(ctx, name)
// 	if err != nil {
// 		return nil, errors.NewNotFound(extensions.Resource("deployments/scale"), name)
// 	}
// 	scale, err := extensions.ScaleFromDeployment(deployment)
// 	if err != nil {
// 		return nil, errors.NewBadRequest(fmt.Sprintf("%v", err))
// 	}
// 	return scale, nil
// }

// func (r *ScaleREST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
// 	if obj == nil {
// 		return nil, false, errors.NewBadRequest(fmt.Sprintf("nil update passed to Scale"))
// 	}
// 	scale, ok := obj.(*extensions.Scale)
// 	if !ok {
// 		return nil, false, errors.NewBadRequest(fmt.Sprintf("wrong object passed to Scale update: %v", obj))
// 	}

// 	if errs := extvalidation.ValidateScale(scale); len(errs) > 0 {
// 		return nil, false, errors.NewInvalid(extensions.Kind("Scale"), scale.Name, errs)
// 	}

// 	deployment, err := (*r.registry).GetDeployment(ctx, scale.Name)
// 	if err != nil {
// 		return nil, false, errors.NewNotFound(extensions.Resource("deployments/scale"), scale.Name)
// 	}
// 	deployment.Spec.Replicas = scale.Spec.Replicas
// 	deployment, err = (*r.registry).UpdateDeployment(ctx, deployment)
// 	if err != nil {
// 		return nil, false, errors.NewConflict(extensions.Resource("deployments/scale"), scale.Name, err)
// 	}
// 	newScale, err := extensions.ScaleFromDeployment(deployment)
// 	if err != nil {
// 		return nil, false, errors.NewBadRequest(fmt.Sprintf("%v", err))
// 	}
// 	return newScale, false, nil
// }
