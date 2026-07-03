/*
Copyright 2025 The Kubernetes Authors.

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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	registry "k8s.io/kubernetes/pkg/registry/batch/cronjob"
	"k8s.io/kubernetes/test/declarative_validation/meta"
	"k8s.io/utils/ptr"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testDeclarativeValidate(t, apiVersion)
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "batch",
		APIVersion:        apiVersion,
		Resource:          "cronjobs",
		IsResourceRequest: true,
		Verb:              "create",
	})
	testCases := map[string]struct {
		input        batch.CronJob
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkCronJob(),
		},
		"schedule: empty": {
			input: mkCronJob(tweakSchedule("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "schedule"), "").MarkAlpha(),
			},
		},
		"jobTemplate.spec.maxFailedIndexes set without backoffLimitPerIndex": {
			input: mkCronJob(tweakJobTemplateMaxFailedIndexes(ptr.To[int32](5))),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "jobTemplate", "spec", "backoffLimitPerIndex"), "").WithOrigin("dependentRequired").MarkAlpha(),
			},
		},
		"jobTemplate.spec.maxFailedIndexes and backoffLimitPerIndex both set": {
			input: mkCronJob(
				tweakJobTemplateMaxFailedIndexes(ptr.To[int32](5)),
				tweakJobTemplateBackoffLimitPerIndex(ptr.To[int32](1)),
			),
		},
		"tolerations: valid key": {
			input: mkCronJob(tweakTolerations(api.Toleration{Key: "example.com/valid-key", Operator: api.TolerationOpExists})),
		},
		"tolerations: valid key without prefix": {
			input: mkCronJob(tweakTolerations(api.Toleration{Key: "simple-key", Operator: api.TolerationOpExists})),
		},
		"tolerations: invalid key format": {
			input: mkCronJob(tweakTolerations(api.Toleration{Key: "invalid key", Operator: api.TolerationOpExists})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "jobTemplate", "spec", "template", "spec", "tolerations").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key").MarkAlpha(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, registry.Strategy, tc.expectedErrs)
		})
	}
	obj := mkCronJob()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())

}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testDeclarativeValidateUpdate(t, apiVersion)
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "batch",
		APIVersion:        apiVersion,
		Resource:          "cronjobs",
		Name:              "valid-obj",
		IsResourceRequest: true,
		Verb:              "update",
	})
	testCases := map[string]struct {
		old          batch.CronJob
		update       batch.CronJob
		expectedErrs field.ErrorList
	}{
		"valid (no changes)": {
			old:    mkCronJob(),
			update: mkCronJob(),
		},

		"schedule: updated to empty": {
			old:    mkCronJob(),
			update: mkCronJob(tweakSchedule("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "schedule"), "").MarkAlpha(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, registry.Strategy, tc.expectedErrs)
		})
	}
	updateObj := mkCronJob()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func mkCronJob(tweaks ...func(*batch.CronJob)) batch.CronJob {
	job := batch.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-class",
			Namespace: "test-namespace",
		},
		Spec: validCronjobSpec,
	}

	for _, tweak := range tweaks {
		tweak(&job)
	}

	return job
}

func tweakSchedule(schedule string) func(*batch.CronJob) {
	return func(job *batch.CronJob) {
		job.Spec.Schedule = schedule
	}
}

func tweakJobTemplateMaxFailedIndexes(v *int32) func(*batch.CronJob) {
	return func(job *batch.CronJob) {
		job.Spec.JobTemplate.Spec.MaxFailedIndexes = v
	}
}

func tweakJobTemplateBackoffLimitPerIndex(v *int32) func(*batch.CronJob) {
	return func(job *batch.CronJob) {
		job.Spec.JobTemplate.Spec.BackoffLimitPerIndex = v
	}
}

func tweakTolerations(tolerations ...api.Toleration) func(*batch.CronJob) {
	return func(job *batch.CronJob) {
		job.Spec.JobTemplate.Spec.Template.Spec.Tolerations = tolerations
	}
}

var validCronjobSpec = batch.CronJobSpec{
	Schedule:          "5 5 * * ?",
	ConcurrencyPolicy: batch.AllowConcurrent,
	TimeZone:          ptr.To("Asia/Shanghai"),
	JobTemplate: batch.JobTemplateSpec{
		Spec: batch.JobSpec{
			Template: api.PodTemplateSpec{
				Spec: podtest.MakePodSpec(podtest.SetRestartPolicy(api.RestartPolicyOnFailure)),
			},
			CompletionMode: ptr.To(batch.IndexedCompletion),
			Completions:    ptr.To[int32](10),
			Parallelism:    ptr.To[int32](10),
		},
	},
}
