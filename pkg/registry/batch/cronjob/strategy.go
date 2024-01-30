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
	"context"
	"fmt"
	"strings"

	batchv1beta1 "k8s.io/api/batch/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/job"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/api/pod"
	"k8s.io/kubernetes/pkg/apis/batch"
	batchvalidation "k8s.io/kubernetes/pkg/apis/batch/validation"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// cronJobStrategy implements verification logic for Replication Controllers.
type cronJobStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating CronJob objects.
var Strategy = cronJobStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// DefaultGarbageCollectionPolicy returns OrphanDependents for batch/v1beta1 for backwards compatibility,
// and DeleteDependents for all other versions.
func (cronJobStrategy) DefaultGarbageCollectionPolicy(ctx context.Context) rest.GarbageCollectionPolicy {
	var groupVersion schema.GroupVersion
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		groupVersion = schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
	}
	switch groupVersion {
	case batchv1beta1.SchemeGroupVersion:
		// for back compatibility
		return rest.OrphanDependents
	default:
		return rest.DeleteDependents
	}
}

// NamespaceScoped returns true because all scheduled jobs need to be within a namespace.
func (cronJobStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (cronJobStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"batch/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
		"batch/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

// PrepareForCreate clears the status of a scheduled job before creation.
func (cronJobStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	cronJob := obj.(*batch.CronJob)
	cronJob.Status = batch.CronJobStatus{}

	cronJob.Generation = 1

	pod.DropDisabledTemplateFields(&cronJob.Spec.JobTemplate.Spec.Template, nil)
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (cronJobStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newCronJob := obj.(*batch.CronJob)
	oldCronJob := old.(*batch.CronJob)
	newCronJob.Status = oldCronJob.Status

	pod.DropDisabledTemplateFields(&newCronJob.Spec.JobTemplate.Spec.Template, &oldCronJob.Spec.JobTemplate.Spec.Template)

	// Any changes to the spec increment the generation number.
	// See metav1.ObjectMeta description for more information on Generation.
	if !apiequality.Semantic.DeepEqual(newCronJob.Spec, oldCronJob.Spec) {
		newCronJob.Generation = oldCronJob.Generation + 1
	}
}

// Validate validates a new scheduled job.
func (cronJobStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	cronJob := obj.(*batch.CronJob)
	opts := pod.GetValidationOptionsFromPodTemplate(&cronJob.Spec.JobTemplate.Spec.Template, nil)
	return batchvalidation.ValidateCronJobCreate(cronJob, opts)
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (cronJobStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	newCronJob := obj.(*batch.CronJob)
	var warnings []string
	if msgs := utilvalidation.IsDNS1123Label(newCronJob.Name); len(msgs) != 0 {
		warnings = append(warnings, fmt.Sprintf("metadata.name: this is used in Pod names and hostnames, which can result in surprising behavior; a DNS label is recommended: %v", msgs))
	}
	warnings = append(warnings, job.WarningsForJobSpec(ctx, field.NewPath("spec", "jobTemplate", "spec"), &newCronJob.Spec.JobTemplate.Spec, nil)...)
	return warnings
}

// Canonicalize normalizes the object after validation.
func (cronJobStrategy) Canonicalize(obj runtime.Object) {
}

func (cronJobStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// AllowCreateOnUpdate is false for scheduled jobs; this means a POST is needed to create one.
func (cronJobStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (cronJobStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	newCronJob := obj.(*batch.CronJob)
	oldCronJob := old.(*batch.CronJob)

	opts := pod.GetValidationOptionsFromPodTemplate(&newCronJob.Spec.JobTemplate.Spec.Template, &oldCronJob.Spec.JobTemplate.Spec.Template)
	return batchvalidation.ValidateCronJobUpdate(newCronJob, oldCronJob, opts)
}

// WarningsOnUpdate returns warnings for the given update.
func (cronJobStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	var warnings []string
	newCronJob := obj.(*batch.CronJob)
	oldCronJob := old.(*batch.CronJob)
	if newCronJob.Generation != oldCronJob.Generation {
		warnings = job.WarningsForJobSpec(ctx, field.NewPath("spec", "jobTemplate", "spec"), &newCronJob.Spec.JobTemplate.Spec, &oldCronJob.Spec.JobTemplate.Spec)
	}
	if strings.Contains(newCronJob.Spec.Schedule, "TZ") {
		warnings = append(warnings, fmt.Sprintf("cannot use TZ or CRON_TZ in %s, use timeZone instead, see https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/ for more details", field.NewPath("spec", "spec", "schedule")))
	}
	return warnings
}

type cronJobStatusStrategy struct {
	cronJobStrategy
}

// StatusStrategy is the default logic invoked when updating object status.
var StatusStrategy = cronJobStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (cronJobStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return map[fieldpath.APIVersion]*fieldpath.Set{
		"batch/v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
		"batch/v1beta1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
	}
}

func (cronJobStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newJob := obj.(*batch.CronJob)
	oldJob := old.(*batch.CronJob)
	newJob.Spec = oldJob.Spec
}

func (cronJobStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return field.ErrorList{}
}

// WarningsOnUpdate returns warnings for the given update.
func (cronJobStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}
