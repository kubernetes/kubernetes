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

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	autoscalingv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	autoscalingvalidation "k8s.io/kubernetes/pkg/apis/autoscaling/validation"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/batch/job"
)

// JobStorage includes dummy storage for Job.
type JobStorage struct {
	Job    *REST
	Status *StatusREST
	Scale  *ScaleREST
}

func NewStorage(optsGetter generic.RESTOptionsGetter) JobStorage {
	jobRest, jobStatusRest := NewREST(optsGetter)
	jobRegistry := job.NewRegistry(jobRest)

	return JobStorage{
		Job:    jobRest,
		Status: jobStatusRest,
		Scale:  &ScaleREST{registry: jobRegistry},
	}
}

// REST implements a RESTStorage for jobs against etcd
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against Jobs.
func NewREST(optsGetter generic.RESTOptionsGetter) (*REST, *StatusREST) {
	store := &genericregistry.Store{
		NewFunc:                  func() runtime.Object { return &batch.Job{} },
		NewListFunc:              func() runtime.Object { return &batch.JobList{} },
		PredicateFunc:            job.MatchJob,
		DefaultQualifiedResource: batch.Resource("jobs"),

		CreateStrategy: job.Strategy,
		UpdateStrategy: job.Strategy,
		DeleteStrategy: job.Strategy,

		TableConvertor: printerstorage.TableConvertor{TablePrinter: printers.NewTablePrinter().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter, AttrFunc: job.GetAttrs}
	if err := store.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}

	statusStore := *store
	statusStore.UpdateStrategy = job.StatusStrategy

	return &REST{store}, &StatusREST{store: &statusStore}
}

// Implement CategoriesProvider
var _ rest.CategoriesProvider = &REST{}

// Categories implements the CategoriesProvider interface. Returns a list of categories a resource is part of.
func (r *REST) Categories() []string {
	return []string{"all"}
}

// StatusREST implements the REST endpoint for changing the status of a resourcequota.
type StatusREST struct {
	store *genericregistry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &batch.Job{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo, createValidation, updateValidation)
}

type ScaleREST struct {
	registry job.Registry
}

// ScaleREST implements Patcher
var _ = rest.Patcher(&ScaleREST{})
var _ = rest.GroupVersionKindProvider(&ScaleREST{})

func (r *ScaleREST) GroupVersionKind(containingGV schema.GroupVersion) schema.GroupVersionKind {
	return autoscalingv1.SchemeGroupVersion.WithKind("Scale")
}

// New creates a new Scale object
func (r *ScaleREST) New() runtime.Object {
	return &autoscaling.Scale{}
}

func (r *ScaleREST) Get(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	job, err := r.registry.GetJob(ctx, name, options)
	if err != nil {
		return nil, errors.NewNotFound(autoscaling.Resource("jobs/scale"), name)
	}
	scale, err := scaleFromJob(job)
	if err != nil {
		return nil, errors.NewInternalError(err)
	}
	return scale, nil
}

func (r *ScaleREST) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc) (runtime.Object, bool, error) {
	job, err := r.registry.GetJob(ctx, name, &metav1.GetOptions{})
	if err != nil {
		return nil, false, errors.NewNotFound(autoscaling.Resource("job/scale"), name)
	}

	oldScale, err := scaleFromJob(job)
	if err != nil {
		return nil, false, errors.NewInternalError(err)
	}

	obj, err := objInfo.UpdatedObject(ctx, oldScale)
	if err != nil {
		return nil, false, errors.NewInternalError(err)
	}
	if obj == nil {
		return nil, false, errors.NewBadRequest(fmt.Sprintf("nil update passed to Scale"))
	}
	scale, ok := obj.(*autoscaling.Scale)
	if !ok {
		return nil, false, errors.NewInternalError(fmt.Errorf("expected input object type to be Scale, but %T", obj))
	}

	if errs := autoscalingvalidation.ValidateScale(scale); len(errs) > 0 {
		return nil, false, errors.NewInvalid(autoscaling.Kind("Scale"), name, errs)
	}

	job.Spec.Parallelism = &scale.Spec.Replicas
	job, err = r.registry.UpdateJob(ctx, job, createValidation, updateValidation)
	if err != nil {
		return nil, false, err
	}
	newScale, err := scaleFromJob(job)
	if err != nil {
		return nil, false, errors.NewInternalError(err)
	}
	return newScale, false, nil
}

// scaleFromJob returns a scale subresource for a job.
func scaleFromJob(job *batch.Job) (*autoscaling.Scale, error) {
	selector, err := metav1.LabelSelectorAsSelector(job.Spec.Selector)
	if err != nil {
		return nil, err
	}
	return &autoscaling.Scale{
		// TODO: Create a variant of ObjectMeta type that only contains the fields below.
		ObjectMeta: metav1.ObjectMeta{
			Name:              job.Name,
			Namespace:         job.Namespace,
			UID:               job.UID,
			ResourceVersion:   job.ResourceVersion,
			CreationTimestamp: job.CreationTimestamp,
		},
		Spec: autoscaling.ScaleSpec{
			Replicas: *job.Spec.Parallelism,
		},
		Status: autoscaling.ScaleStatus{
			Replicas: *job.Spec.Parallelism,
			Selector: selector.String(),
		},
	}, nil
}
