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
	"net/http"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	storeerr "k8s.io/apiserver/pkg/storage/errors"
	appsv1beta1 "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	appsv1beta2 "k8s.io/kubernetes/pkg/apis/apps/v1beta2"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	autoscalingv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	autoscalingvalidation "k8s.io/kubernetes/pkg/apis/autoscaling/validation"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extensionsv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	extvalidation "k8s.io/kubernetes/pkg/apis/extensions/validation"
	"k8s.io/kubernetes/pkg/registry/extensions/deployment"
)

// DeploymentStorage includes dummy storage for Deployments and for Scale subresource.
type DeploymentStorage struct {
	Deployment *REST
	Status     *StatusREST
	Scale      *ScaleREST
	Rollback   *RollbackREST
}

func NewStorage(optsGetter generic.RESTOptionsGetter) DeploymentStorage {
	deploymentRest, deploymentStatusRest, deploymentRollbackRest := NewREST(optsGetter)
	deploymentRegistry := deployment.NewRegistry(deploymentRest)

	return DeploymentStorage{
		Deployment: deploymentRest,
		Status:     deploymentStatusRest,
		Scale:      &ScaleREST{registry: deploymentRegistry},
		Rollback:   deploymentRollbackRest,
	}
}

type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against deployments.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, *StatusREST, *RollbackREST) {
	store := &genericregistry.Store{
		NewFunc:                  func() runtime.Object { return &extensions.Deployment{} },
		NewListFunc:              func() runtime.Object { return &extensions.DeploymentList{} },
		DefaultQualifiedResource: extensions.Resource("deployments"),

		CreateStrategy: deployment.Strategy,
		UpdateStrategy: deployment.Strategy,
		DeleteStrategy: deployment.Strategy,
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}

	statusStore := *store
	statusStore.UpdateStrategy = deployment.StatusStrategy
	return &REST{store}, &StatusREST{store: &statusStore}, &RollbackREST{store: store}
}

// Implement ShortNamesProvider
var _ rest.ShortNamesProvider = &REST{}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"deploy"}
}

// Implement CategoriesProvider
var _ rest.CategoriesProvider = &REST{}

// Categories implements the CategoriesProvider interface. Returns a list of categories a resource is part of.
func (r *REST) Categories() []string {
	return []string{"all"}
}

// StatusREST implements the REST endpoint for changing the status of a deployment
type StatusREST struct {
	store *genericregistry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &extensions.Deployment{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo, createValidation, updateValidation)
}

// RollbackREST implements the REST endpoint for initiating the rollback of a deployment
type RollbackREST struct {
	store *genericregistry.Store
}

// New creates a rollback
func (r *RollbackREST) New() runtime.Object {
	return &extensions.DeploymentRollback{}
}

var _ = rest.Creater(&RollbackREST{})

func (r *RollbackREST) Create(ctx genericapirequest.Context, obj runtime.Object, createValidation rest.ValidateObjectFunc, includeUninitialized bool) (runtime.Object, error) {
	rollback, ok := obj.(*extensions.DeploymentRollback)
	if !ok {
		return nil, errors.NewBadRequest(fmt.Sprintf("not a DeploymentRollback: %#v", obj))
	}

	if errs := extvalidation.ValidateDeploymentRollback(rollback); len(errs) != 0 {
		return nil, errors.NewInvalid(extensions.Kind("DeploymentRollback"), rollback.Name, errs)
	}

	// Update the Deployment with information in DeploymentRollback to trigger rollback
	err := r.rollbackDeployment(ctx, rollback.Name, &rollback.RollbackTo, rollback.UpdatedAnnotations)
	if err != nil {
		return nil, err
	}
	return &metav1.Status{
		Status:  metav1.StatusSuccess,
		Message: fmt.Sprintf("rollback request for deployment %q succeeded", rollback.Name),
		Code:    http.StatusOK,
	}, nil
}

func (r *RollbackREST) rollbackDeployment(ctx genericapirequest.Context, deploymentID string, config *extensions.RollbackConfig, annotations map[string]string) error {
	if _, err := r.setDeploymentRollback(ctx, deploymentID, config, annotations); err != nil {
		err = storeerr.InterpretGetError(err, extensions.Resource("deployments"), deploymentID)
		err = storeerr.InterpretUpdateError(err, extensions.Resource("deployments"), deploymentID)
		if _, ok := err.(*errors.StatusError); !ok {
			err = errors.NewInternalError(err)
		}
		return err
	}
	return nil
}

func (r *RollbackREST) setDeploymentRollback(ctx genericapirequest.Context, deploymentID string, config *extensions.RollbackConfig, annotations map[string]string) (*extensions.Deployment, error) {
	dKey, err := r.store.KeyFunc(ctx, deploymentID)
	if err != nil {
		return nil, err
	}
	var finalDeployment *extensions.Deployment
	err = r.store.Storage.GuaranteedUpdate(ctx, dKey, &extensions.Deployment{}, false, nil, storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
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
	registry deployment.Registry
}

// ScaleREST implements Patcher
var _ = rest.Patcher(&ScaleREST{})
var _ = rest.GroupVersionKindProvider(&ScaleREST{})

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

func (r *ScaleREST) Get(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	deployment, err := r.registry.GetDeployment(ctx, name, options)
	if err != nil {
		return nil, errors.NewNotFound(extensions.Resource("deployments/scale"), name)
	}
	scale, err := scaleFromDeployment(deployment)
	if err != nil {
		return nil, errors.NewBadRequest(fmt.Sprintf("%v", err))
	}
	return scale, nil
}

func (r *ScaleREST) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc) (runtime.Object, bool, error) {
	deployment, err := r.registry.GetDeployment(ctx, name, &metav1.GetOptions{})
	if err != nil {
		return nil, false, errors.NewNotFound(extensions.Resource("deployments/scale"), name)
	}

	oldScale, err := scaleFromDeployment(deployment)
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
	scale, ok := obj.(*autoscaling.Scale)
	if !ok {
		return nil, false, errors.NewBadRequest(fmt.Sprintf("expected input object type to be Scale, but %T", obj))
	}

	if errs := autoscalingvalidation.ValidateScale(scale); len(errs) > 0 {
		return nil, false, errors.NewInvalid(extensions.Kind("Scale"), name, errs)
	}

	deployment.Spec.Replicas = scale.Spec.Replicas
	deployment.ResourceVersion = scale.ResourceVersion
	deployment, err = r.registry.UpdateDeployment(ctx, deployment, createValidation, updateValidation)
	if err != nil {
		return nil, false, err
	}
	newScale, err := scaleFromDeployment(deployment)
	if err != nil {
		return nil, false, errors.NewBadRequest(fmt.Sprintf("%v", err))
	}
	return newScale, false, nil
}

// scaleFromDeployment returns a scale subresource for a deployment.
func scaleFromDeployment(deployment *extensions.Deployment) (*autoscaling.Scale, error) {
	selector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		return nil, err
	}
	return &autoscaling.Scale{
		// TODO: Create a variant of ObjectMeta type that only contains the fields below.
		ObjectMeta: metav1.ObjectMeta{
			Name:              deployment.Name,
			Namespace:         deployment.Namespace,
			UID:               deployment.UID,
			ResourceVersion:   deployment.ResourceVersion,
			CreationTimestamp: deployment.CreationTimestamp,
		},
		Spec: autoscaling.ScaleSpec{
			Replicas: deployment.Spec.Replicas,
		},
		Status: autoscaling.ScaleStatus{
			Replicas: deployment.Status.Replicas,
			Selector: selector.String(),
		},
	}, nil
}
