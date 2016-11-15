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

package cronjob

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/batch/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// scheduledJobStrategy implements verification logic for Replication Controllers.
type scheduledJobStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating CronJob objects.
var Strategy = scheduledJobStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped returns true because all scheduled jobs need to be within a namespace.
func (scheduledJobStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the status of a scheduled job before creation.
func (scheduledJobStrategy) PrepareForCreate(ctx api.Context, obj runtime.Object) {
	scheduledJob := obj.(*batch.CronJob)
	scheduledJob.Status = batch.CronJobStatus{}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (scheduledJobStrategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
	newCronJob := obj.(*batch.CronJob)
	oldCronJob := old.(*batch.CronJob)
	newCronJob.Status = oldCronJob.Status
}

// Validate validates a new scheduled job.
func (scheduledJobStrategy) Validate(ctx api.Context, obj runtime.Object) field.ErrorList {
	scheduledJob := obj.(*batch.CronJob)
	return validation.ValidateCronJob(scheduledJob)
}

// Canonicalize normalizes the object after validation.
func (scheduledJobStrategy) Canonicalize(obj runtime.Object) {
}

func (scheduledJobStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// AllowCreateOnUpdate is false for scheduled jobs; this means a POST is needed to create one.
func (scheduledJobStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (scheduledJobStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateCronJob(obj.(*batch.CronJob))
}

type scheduledJobStatusStrategy struct {
	scheduledJobStrategy
}

var StatusStrategy = scheduledJobStatusStrategy{Strategy}

func (scheduledJobStatusStrategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
	newJob := obj.(*batch.CronJob)
	oldJob := old.(*batch.CronJob)
	newJob.Spec = oldJob.Spec
}

func (scheduledJobStatusStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	return field.ErrorList{}
}

// CronJobToSelectableFields returns a field set that represents the object for matching purposes.
func CronJobToSelectableFields(scheduledJob *batch.CronJob) fields.Set {
	return generic.ObjectMetaFieldsSet(&scheduledJob.ObjectMeta, true)
}

// MatchCronJob is the filter used by the generic etcd backend to route
// watch events from etcd to clients of the apiserver only interested in specific
// labels/fields.
func MatchCronJob(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label: label,
		Field: field,
		GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			scheduledJob, ok := obj.(*batch.CronJob)
			if !ok {
				return nil, nil, fmt.Errorf("Given object is not a scheduled job.")
			}
			return labels.Set(scheduledJob.ObjectMeta.Labels), CronJobToSelectableFields(scheduledJob), nil
		},
	}
}
