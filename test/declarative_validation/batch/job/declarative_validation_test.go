/*
Copyright The Kubernetes Authors.

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
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	registry "k8s.io/kubernetes/pkg/registry/batch/job"
	"k8s.io/kubernetes/test/declarative_validation/meta"
	"k8s.io/utils/ptr"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:   "batch",
				APIVersion: apiVersion,
				Resource:   "jobs",
			})
			job := mkJob()
			meta.RunObjectMetaTestCases(t, ctx, &job, registry.Strategy, meta.WithStringentFinalizerValidation())
			testDeclarativeValidate(t, ctx, apiVersion)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:   "batch",
				APIVersion: apiVersion,
				Resource:   "jobs",
			})
			job := mkJob()
			meta.RunObjectMetaUpdateTestCases(t, ctx, &job, registry.Strategy, meta.WithStringentFinalizerValidation())
		})
	}
}

func testDeclarativeValidate(t *testing.T, ctx context.Context, apiVersion string) {
	testCases := map[string]struct {
		input        batch.Job
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkJob(),
		},
		"maxFailedIndexes and backoffLimitPerIndex both set": {
			input: mkJob(
				tweakMaxFailedIndexes(ptr.To[int32](5)),
				tweakBackoffLimitPerIndex(ptr.To[int32](1)),
			),
		},
		"maxFailedIndexes set without backoffLimitPerIndex": {
			input: mkJob(tweakMaxFailedIndexes(ptr.To[int32](5))),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "backoffLimitPerIndex"), "").WithOrigin("dependentRequired").MarkAlpha(),
			},
		},
		"valid toleration key": {
			input: mkJob(tweakTolerations(api.Toleration{Key: "example.com/valid-key", Operator: api.TolerationOpExists})),
		},
		"valid toleration key without prefix": {
			input: mkJob(tweakTolerations(api.Toleration{Key: "simple-key", Operator: api.TolerationOpExists})),
		},
		"invalid toleration key format": {
			input: mkJob(tweakTolerations(api.Toleration{Key: "invalid key", Operator: api.TolerationOpExists})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "template", "spec", "tolerations").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key").MarkAlpha(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, registry.Strategy, tc.expectedErrs)
		})
	}
}

func mkJob(mutators ...func(*batch.Job)) batch.Job {
	job := batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "myjob",
			Namespace: metav1.NamespaceDefault,
			UID:       types.UID("1a2b3c"),
		},
		Spec: validJobSpec,
	}
	manualSelector := true
	job.Spec.ManualSelector = &manualSelector
	for _, mutate := range mutators {
		mutate(&job)
	}
	return job
}

func tweakMaxFailedIndexes(v *int32) func(*batch.Job) {
	return func(job *batch.Job) {
		job.Spec.MaxFailedIndexes = v
	}
}

func tweakBackoffLimitPerIndex(v *int32) func(*batch.Job) {
	return func(job *batch.Job) {
		job.Spec.BackoffLimitPerIndex = v
	}
}

func tweakTolerations(tolerations ...api.Toleration) func(*batch.Job) {
	return func(job *batch.Job) {
		job.Spec.Template.Spec.Tolerations = tolerations
	}
}

var validJobLabels = map[string]string{
	batch.ControllerUidLabel:       "1a2b3c",
	batch.LegacyControllerUidLabel: "1a2b3c",
	batch.JobNameLabel:             "myjob",
	batch.LegacyJobNameLabel:       "myjob",
}

var validJobSpec = batch.JobSpec{
	Selector: &metav1.LabelSelector{MatchLabels: validJobLabels},
	Template: api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{Labels: validJobLabels},
		Spec:       podtest.MakePodSpec(podtest.SetRestartPolicy(api.RestartPolicyNever)),
	},
	CompletionMode: ptr.To(batch.IndexedCompletion),
	Completions:    ptr.To[int32](10),
	Parallelism:    ptr.To[int32](10),
}
