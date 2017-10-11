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

package storage

import (
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/batch/cronjob"
	"k8s.io/kubernetes/pkg/registry/batch/job/storage"
)

type CronJobStorage struct {
	CronJob     *REST
	Status      *StatusREST
	Instantiate *InstantiateREST
}

// REST implements a RESTStorage for scheduled jobs against etcd
type REST struct {
	*genericregistry.Store
}

// NewREST returns a RESTStorage object that will work against CronJobs.
func NewREST(optsGetter generic.RESTOptionsGetter) CronJobStorage {
	cronJobStore := &genericregistry.Store{
		NewFunc:                  func() runtime.Object { return &batch.CronJob{} },
		NewListFunc:              func() runtime.Object { return &batch.CronJobList{} },
		DefaultQualifiedResource: batch.Resource("cronjobs"),

		CreateStrategy: cronjob.Strategy,
		UpdateStrategy: cronjob.Strategy,
		DeleteStrategy: cronjob.Strategy,

		TableConvertor: printerstorage.TableConvertor{TablePrinter: printers.NewTablePrinter().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := cronJobStore.CompleteWithOptions(options); err != nil {
		panic(err) // TODO: Propagate error up
	}

	statusStore := *cronJobStore
	statusStore.UpdateStrategy = cronjob.StatusStrategy

	// Instantiate needs access to job storage to create jobs
	jobStore, _ := storage.NewREST(optsGetter)

	return CronJobStorage{
		CronJob:     &REST{cronJobStore},
		Status:      &StatusREST{store: &statusStore},
		Instantiate: &InstantiateREST{cronJobStore, jobStore.Store},
	}
}

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
	return &batch.CronJob{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo, createValidation, updateValidation)
}

// InstantiateREST implements the REST endpoint for manually triggering a CronJob.
type InstantiateREST struct {
	cronJobStore *genericregistry.Store
	jobStore     *genericregistry.Store
}

var _ = rest.NamedCreater(&InstantiateREST{})

func (r *InstantiateREST) New() runtime.Object {
	return &batch.CronJobManualInstantiation{}
}

func (r *InstantiateREST) Create(ctx genericapirequest.Context, name string, obj runtime.Object, createValidation rest.ValidateObjectFunc, includeUninitialized bool) (out runtime.Object, err error) {
	sourceCronJobObj, err := r.cronJobStore.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	sourceCronJob, ok := sourceCronJobObj.(*batch.CronJob)
	if !ok {
		return nil, fmt.Errorf("got object of type %T, expected *batch.CronJob", sourceCronJobObj)
	}

	// add an extra label onto the created job to signal to the cronjob controller that it should adopt this job
	jobLabels := make(map[string]string)
	for _, key := range sourceCronJob.Spec.JobTemplate.Labels {
		jobLabels[key] = sourceCronJob.Spec.JobTemplate.Labels[key]
	}
	jobLabels["createdByInstantiate"] = "yes"

	job := &batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:        fmt.Sprintf("%s-manual-%d", name, time.Now().Unix()),
			Labels:      jobLabels,
			Annotations: sourceCronJob.Spec.JobTemplate.Annotations,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(sourceCronJob, batch.SchemeGroupVersion.WithKind("CronJob")),
			},
		},
		Spec: sourceCronJob.Spec.JobTemplate.Spec,
	}

	createdJobObj, err := r.jobStore.Create(ctx, job, createValidation, includeUninitialized)
	if err != nil {
		return nil, err
	}
	createdJob, ok := createdJobObj.(*batch.Job)
	if !ok {
		return nil, fmt.Errorf("got object of type %T, expected *batch.Job", createdJobObj)
	}

	return &batch.CronJobManualInstantiation{CreatedJob: *createdJob}, nil
}
