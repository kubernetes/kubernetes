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
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/batch"
)

var apiVersions = []string{"v1", "v1beta1"}

func TestDeclarativeValidateForDeclarative(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testDeclarativeValidateForDeclarative(t, apiVersion)
	}
}

func testDeclarativeValidateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "batch",
		APIVersion: apiVersion,
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
				field.Required(field.NewPath("spec", "schedule"), ""),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestValidateUpdateForDeclarative(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testValidateUpdateForDeclarative(t, apiVersion)
	}
}

func testValidateUpdateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "batch",
		APIVersion: apiVersion,
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
				field.Required(field.NewPath("spec", "schedule"), ""),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func mkCronJob(mutators ...func(*batch.CronJob)) batch.CronJob {
	job := batch.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-class",
			Namespace: "test-namespace",
		},
		Spec: validCronjobSpec,
	}

	for _, mutate := range mutators {
		mutate(&job)
	}

	return job
}

func tweakSchedule(schedule string) func(*batch.CronJob) {
	return func(job *batch.CronJob) {
		job.Spec.Schedule = schedule
	}
}
