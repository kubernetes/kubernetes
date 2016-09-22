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

package job

import (
	"fmt"
	"strconv"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/batch/validation"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// jobStrategy implements verification logic for Replication Controllers.
type jobStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Replication Controller objects.
var Strategy = jobStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped returns true because all jobs need to be within a namespace.
func (jobStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears the status of a job before creation.
func (jobStrategy) PrepareForCreate(ctx api.Context, obj runtime.Object) {
	job := obj.(*batch.Job)
	job.Status = batch.JobStatus{}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (jobStrategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
	newJob := obj.(*batch.Job)
	oldJob := old.(*batch.Job)
	newJob.Status = oldJob.Status
}

// Validate validates a new job.
func (jobStrategy) Validate(ctx api.Context, obj runtime.Object) field.ErrorList {
	job := obj.(*batch.Job)
	// TODO: move UID generation earlier and do this in defaulting logic?
	if job.Spec.ManualSelector == nil || *job.Spec.ManualSelector == false {
		generateSelector(job)
	}
	return validation.ValidateJob(job)
}

// generateSelector adds a selector to a job and labels to its template
// which can be used to uniquely identify the pods created by that job,
// if the user has requested this behavior.
func generateSelector(obj *batch.Job) {
	if obj.Spec.Template.Labels == nil {
		obj.Spec.Template.Labels = make(map[string]string)
	}
	// The job-name label is unique except in cases that are expected to be
	// quite uncommon, and is more user friendly than uid.  So, we add it as
	// a label.
	_, found := obj.Spec.Template.Labels["job-name"]
	if found {
		// User asked us to not automatically generate a selector and labels,
		// but set a possibly conflicting value.  If there is a conflict,
		// we will reject in validation.
	} else {
		obj.Spec.Template.Labels["job-name"] = string(obj.ObjectMeta.Name)
	}
	// The controller-uid label makes the pods that belong to this job
	// only match this job.
	_, found = obj.Spec.Template.Labels["controller-uid"]
	if found {
		// User asked us to automatically generate a selector and labels,
		// but set a possibly conflicting value.  If there is a conflict,
		// we will reject in validation.
	} else {
		obj.Spec.Template.Labels["controller-uid"] = string(obj.ObjectMeta.UID)
	}
	// Select the controller-uid label.  This is sufficient for uniqueness.
	if obj.Spec.Selector == nil {
		obj.Spec.Selector = &unversioned.LabelSelector{}
	}
	if obj.Spec.Selector.MatchLabels == nil {
		obj.Spec.Selector.MatchLabels = make(map[string]string)
	}
	if _, found := obj.Spec.Selector.MatchLabels["controller-uid"]; !found {
		obj.Spec.Selector.MatchLabels["controller-uid"] = string(obj.ObjectMeta.UID)
	}
	// If the user specified matchLabel controller-uid=$WRONGUID, then it should fail
	// in validation, either because the selector does not match the pod template
	// (controller-uid=$WRONGUID does not match controller-uid=$UID, which we applied
	// above, or we will reject in validation because the template has the wrong
	// labels.
}

// TODO: generalize generateSelector so it can work for other controller
// objects such as ReplicaSet.  Can use pkg/api/meta to generically get the
// UID, but need some way to generically access the selector and pod labels
// fields.

// Canonicalize normalizes the object after validation.
func (jobStrategy) Canonicalize(obj runtime.Object) {
}

func (jobStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// AllowCreateOnUpdate is false for jobs; this means a POST is needed to create one.
func (jobStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (jobStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	validationErrorList := validation.ValidateJob(obj.(*batch.Job))
	updateErrorList := validation.ValidateJobUpdate(obj.(*batch.Job), old.(*batch.Job))
	return append(validationErrorList, updateErrorList...)
}

type jobStatusStrategy struct {
	jobStrategy
}

var StatusStrategy = jobStatusStrategy{Strategy}

func (jobStatusStrategy) PrepareForUpdate(ctx api.Context, obj, old runtime.Object) {
	newJob := obj.(*batch.Job)
	oldJob := old.(*batch.Job)
	newJob.Spec = oldJob.Spec
}

func (jobStatusStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateJobUpdateStatus(obj.(*batch.Job), old.(*batch.Job))
}

// JobSelectableFields returns a field set that represents the object for matching purposes.
func JobToSelectableFields(job *batch.Job) fields.Set {
	objectMetaFieldsSet := generic.ObjectMetaFieldsSet(&job.ObjectMeta, true)
	specificFieldsSet := fields.Set{
		"status.successful": strconv.Itoa(int(job.Status.Succeeded)),
	}
	return generic.MergeFieldsSets(objectMetaFieldsSet, specificFieldsSet)
}

// MatchJob is the filter used by the generic etcd backend to route
// watch events from etcd to clients of the apiserver only interested in specific
// labels/fields.
func MatchJob(label labels.Selector, field fields.Selector) *generic.SelectionPredicate {
	return &generic.SelectionPredicate{
		Label: label,
		Field: field,
		GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			job, ok := obj.(*batch.Job)
			if !ok {
				return nil, nil, fmt.Errorf("Given object is not a job.")
			}
			return labels.Set(job.ObjectMeta.Labels), JobToSelectableFields(job), nil
		},
	}
}
