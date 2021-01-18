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

package validation

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	corevalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/pointer"
)

var ignoreErrValueDetail = cmpopts.IgnoreFields(field.Error{}, "BadValue", "Detail")

func getValidManualSelector() *metav1.LabelSelector {
	return &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
}

func getValidPodTemplateSpecForManual(selector *metav1.LabelSelector) api.PodTemplateSpec {
	return api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: selector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
	}
}

func getValidGeneratedSelector() *metav1.LabelSelector {
	return &metav1.LabelSelector{
		MatchLabels: map[string]string{"controller-uid": "1a2b3c", "job-name": "myjob"},
	}
}

func getValidPodTemplateSpecForGenerated(selector *metav1.LabelSelector) api.PodTemplateSpec {
	return api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: selector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
	}
}

func TestValidateJob(t *testing.T) {
	validManualSelector := getValidManualSelector()
	validPodTemplateSpecForManual := getValidPodTemplateSpecForManual(validManualSelector)
	validGeneratedSelector := getValidGeneratedSelector()
	validPodTemplateSpecForGenerated := getValidPodTemplateSpecForGenerated(validGeneratedSelector)

	successCases := map[string]struct {
		opts JobValidationOptions
		job  batch.Job
	}{
		"valid manual selector": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "myjob",
					Namespace:   metav1.NamespaceDefault,
					UID:         types.UID("1a2b3c"),
					Annotations: map[string]string{"foo": "bar"},
				},
				Spec: batch.JobSpec{
					Selector:       validManualSelector,
					ManualSelector: pointer.BoolPtr(true),
					Template:       validPodTemplateSpecForManual,
				},
			},
		},
		"valid generated selector": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
		},
		"valid NonIndexed completion mode": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGenerated,
					CompletionMode: completionModePtr(batch.NonIndexedCompletion),
				},
			},
		},
		"valid Indexed completion mode": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGenerated,
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    pointer.Int32Ptr(2),
					Parallelism:    pointer.Int32Ptr(100000),
				},
			},
		},
		"valid job tracking annotation": {
			opts: JobValidationOptions{
				AllowTrackingAnnotation: true,
			},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
					Annotations: map[string]string{
						batch.JobTrackingFinalizer: "",
					},
				},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
		},
	}
	for k, v := range successCases {
		t.Run(k, func(t *testing.T) {
			if errs := ValidateJob(&v.job, v.opts); len(errs) != 0 {
				t.Errorf("Got unexpected validation errors: %v", errs)
			}
		})
	}
	negative := int32(-1)
	negative64 := int64(-1)
	errorCases := map[string]batch.Job{
		"spec.parallelism:must be greater than or equal to 0": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "myjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				Parallelism: &negative,
				Selector:    validGeneratedSelector,
				Template:    validPodTemplateSpecForGenerated,
			},
		},
		"spec.completions:must be greater than or equal to 0": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "myjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				Completions: &negative,
				Selector:    validGeneratedSelector,
				Template:    validPodTemplateSpecForGenerated,
			},
		},
		"spec.activeDeadlineSeconds:must be greater than or equal to 0": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "myjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				ActiveDeadlineSeconds: &negative64,
				Selector:              validGeneratedSelector,
				Template:              validPodTemplateSpecForGenerated,
			},
		},
		"spec.selector:Required value": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "myjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				Template: validPodTemplateSpecForGenerated,
			},
		},
		"spec.template.metadata.labels: Invalid value: map[string]string{\"y\":\"z\"}: `selector` does not match template `labels`": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "myjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				Selector:       validManualSelector,
				ManualSelector: pointer.BoolPtr(true),
				Template: api.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{"y": "z"},
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyOnFailure,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
					},
				},
			},
		},
		"spec.template.metadata.labels: Invalid value: map[string]string{\"controller-uid\":\"4d5e6f\"}: `selector` does not match template `labels`": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "myjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				Selector:       validManualSelector,
				ManualSelector: pointer.BoolPtr(true),
				Template: api.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{"controller-uid": "4d5e6f"},
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyOnFailure,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
					},
				},
			},
		},
		"spec.template.spec.restartPolicy: Required value": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "myjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				Selector:       validManualSelector,
				ManualSelector: pointer.BoolPtr(true),
				Template: api.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: validManualSelector.MatchLabels,
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyAlways,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
					},
				},
			},
		},
		"spec.template.spec.restartPolicy: Unsupported value": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "myjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				Selector:       validManualSelector,
				ManualSelector: pointer.BoolPtr(true),
				Template: api.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: validManualSelector.MatchLabels,
					},
					Spec: api.PodSpec{
						RestartPolicy: "Invalid",
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
					},
				},
			},
		},
		"spec.ttlSecondsAfterFinished: must be greater than or equal to 0": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "myjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				TTLSecondsAfterFinished: &negative,
				Selector:                validGeneratedSelector,
				Template:                validPodTemplateSpecForGenerated,
			},
		},
		"spec.completions: Required value: when completion mode is Indexed": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "myjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				Selector:       validGeneratedSelector,
				Template:       validPodTemplateSpecForGenerated,
				CompletionMode: completionModePtr(batch.IndexedCompletion),
			},
		},
		"spec.parallelism: must be less than or equal to 100000 when completion mode is Indexed": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "myjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.JobSpec{
				Selector:       validGeneratedSelector,
				Template:       validPodTemplateSpecForGenerated,
				CompletionMode: completionModePtr(batch.IndexedCompletion),
				Completions:    pointer.Int32Ptr(2),
				Parallelism:    pointer.Int32Ptr(100001),
			},
		},
		"metadata.annotations[batch.kubernetes.io/job-tracking]: cannot add this annotation": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "myjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
				Annotations: map[string]string{
					batch.JobTrackingFinalizer: "",
				},
			},
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGenerated,
			},
		},
	}

	for k, v := range errorCases {
		t.Run(k, func(t *testing.T) {
			errs := ValidateJob(&v, JobValidationOptions{})
			if len(errs) == 0 {
				t.Errorf("expected failure for %s", k)
			} else {
				s := strings.SplitN(k, ":", 2)
				err := errs[0]
				if err.Field != s[0] || !strings.Contains(err.Error(), s[1]) {
					t.Errorf("unexpected error: %v, expected: %s", err, k)
				}
			}
		})
	}
}

func TestValidateJobUpdate(t *testing.T) {
	validGeneratedSelector := getValidGeneratedSelector()
	validPodTemplateSpecForGenerated := getValidPodTemplateSpecForGenerated(validGeneratedSelector)
	cases := map[string]struct {
		old    batch.Job
		update func(*batch.Job)
		err    *field.Error
	}{
		"mutable fields": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector:                validGeneratedSelector,
					Template:                validPodTemplateSpecForGenerated,
					Parallelism:             pointer.Int32Ptr(5),
					ActiveDeadlineSeconds:   pointer.Int64Ptr(2),
					TTLSecondsAfterFinished: pointer.Int32Ptr(1),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Parallelism = pointer.Int32Ptr(2)
				job.Spec.ActiveDeadlineSeconds = pointer.Int64Ptr(3)
				job.Spec.TTLSecondsAfterFinished = pointer.Int32Ptr(2)
				job.Spec.ManualSelector = pointer.BoolPtr(true)
			},
		},
		"immutable completion": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Completions = pointer.Int32Ptr(1)
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.completions",
			},
		},
		"immutable selector": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Selector.MatchLabels["foo"] = "bar"
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.selector",
			},
		},
		"immutable pod template": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.Containers = nil
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.template",
			},
		},
		"immutable completion mode": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGenerated,
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    pointer.Int32Ptr(2),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.CompletionMode = completionModePtr(batch.NonIndexedCompletion)
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.completionMode",
			},
		},
	}
	ignoreValueAndDetail := cmpopts.IgnoreFields(field.Error{}, "BadValue", "Detail")
	for k, tc := range cases {
		t.Run(k, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			update := tc.old.DeepCopy()
			tc.update(update)
			errs := ValidateJobUpdate(&tc.old, update, corevalidation.PodValidationOptions{})
			var wantErrs field.ErrorList
			if tc.err != nil {
				wantErrs = append(wantErrs, tc.err)
			}
			if diff := cmp.Diff(wantErrs, errs, ignoreValueAndDetail); diff != "" {
				t.Errorf("Unexpected validation errors (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestValidateJobUpdateStatus(t *testing.T) {
	cases := map[string]struct {
		old      batch.Job
		update   batch.Job
		wantErrs field.ErrorList
	}{
		"valid": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "abc",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "1",
				},
				Status: batch.JobStatus{
					Active:    1,
					Succeeded: 2,
					Failed:    3,
				},
			},
			update: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "abc",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "1",
				},
				Status: batch.JobStatus{
					Active:    1,
					Succeeded: 1,
					Failed:    3,
				},
			},
		},
		"negative counts": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "abc",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "10",
				},
				Status: batch.JobStatus{
					Active:    1,
					Succeeded: 2,
					Failed:    3,
				},
			},
			update: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "abc",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "10",
				},
				Status: batch.JobStatus{
					Active:    -1,
					Succeeded: -2,
					Failed:    3,
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.active"},
				{Type: field.ErrorTypeInvalid, Field: "status.succeeded"},
			},
		},
		"empty and duplicated uncounted pods": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "abc",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "5",
				},
			},
			update: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "abc",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "5",
				},
				Status: batch.JobStatus{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Succeeded: []types.UID{"a", "b", "c", "a", ""},
						Failed:    []types.UID{"c", "d", "e", "d", ""},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeDuplicate, Field: "status.uncountedTerminatedPods.succeeded[3]"},
				{Type: field.ErrorTypeInvalid, Field: "status.uncountedTerminatedPods.succeeded[4]"},
				{Type: field.ErrorTypeDuplicate, Field: "status.uncountedTerminatedPods.failed[0]"},
				{Type: field.ErrorTypeDuplicate, Field: "status.uncountedTerminatedPods.failed[3]"},
				{Type: field.ErrorTypeInvalid, Field: "status.uncountedTerminatedPods.failed[4]"},
			},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			errs := ValidateJobUpdateStatus(&tc.update, &tc.old)
			if diff := cmp.Diff(tc.wantErrs, errs, ignoreErrValueDetail); diff != "" {
				t.Errorf("Unexpected errors (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestValidateCronJob(t *testing.T) {
	validManualSelector := getValidManualSelector()
	validPodTemplateSpec := getValidPodTemplateSpecForGenerated(getValidGeneratedSelector())
	validPodTemplateSpec.Labels = map[string]string{}

	successCases := map[string]batch.CronJob{
		"basic scheduled job": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "* * * * ?",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"non-standard scheduled": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "@hourly",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
	}
	for k, v := range successCases {
		if errs := ValidateCronJob(&v, corevalidation.PodValidationOptions{}); len(errs) != 0 {
			t.Errorf("expected success for %s: %v", k, errs)
		}

		// Update validation should pass same success cases
		// copy to avoid polluting the testcase object, set a resourceVersion to allow validating update, and test a no-op update
		v = *v.DeepCopy()
		v.ResourceVersion = "1"
		if errs := ValidateCronJobUpdate(&v, &v, corevalidation.PodValidationOptions{}); len(errs) != 0 {
			t.Errorf("expected success for %s: %v", k, errs)
		}
	}

	negative := int32(-1)
	negative64 := int64(-1)

	errorCases := map[string]batch.CronJob{
		"spec.schedule: Invalid value": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "error",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"spec.schedule: Required value": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"spec.startingDeadlineSeconds:must be greater than or equal to 0": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:                "* * * * ?",
				ConcurrencyPolicy:       batch.AllowConcurrent,
				StartingDeadlineSeconds: &negative64,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"spec.successfulJobsHistoryLimit: must be greater than or equal to 0": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:                   "* * * * ?",
				ConcurrencyPolicy:          batch.AllowConcurrent,
				SuccessfulJobsHistoryLimit: &negative,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"spec.failedJobsHistoryLimit: must be greater than or equal to 0": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:               "* * * * ?",
				ConcurrencyPolicy:      batch.AllowConcurrent,
				FailedJobsHistoryLimit: &negative,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"spec.concurrencyPolicy: Required value": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule: "* * * * ?",
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"spec.jobTemplate.spec.parallelism:must be greater than or equal to 0": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "* * * * ?",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Parallelism: &negative,
						Template:    validPodTemplateSpec,
					},
				},
			},
		},
		"spec.jobTemplate.spec.completions:must be greater than or equal to 0": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "* * * * ?",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{

					Spec: batch.JobSpec{
						Completions: &negative,
						Template:    validPodTemplateSpec,
					},
				},
			},
		},
		"spec.jobTemplate.spec.activeDeadlineSeconds:must be greater than or equal to 0": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "* * * * ?",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						ActiveDeadlineSeconds: &negative64,
						Template:              validPodTemplateSpec,
					},
				},
			},
		},
		"spec.jobTemplate.spec.selector: Invalid value: {\"matchLabels\":{\"a\":\"b\"}}: `selector` will be auto-generated": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "* * * * ?",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Selector: validManualSelector,
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"metadata.name: must be no more than 52 characters": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "10000000002000000000300000000040000000005000000000123",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "* * * * ?",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"spec.jobTemplate.spec.manualSelector: Unsupported value": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "* * * * ?",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						ManualSelector: pointer.BoolPtr(true),
						Template:       validPodTemplateSpec,
					},
				},
			},
		},
		"spec.jobTemplate.spec.template.spec.restartPolicy: Required value": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "* * * * ?",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: api.PodTemplateSpec{
							Spec: api.PodSpec{
								RestartPolicy: api.RestartPolicyAlways,
								DNSPolicy:     api.DNSClusterFirst,
								Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
							},
						},
					},
				},
			},
		},
		"spec.jobTemplate.spec.template.spec.restartPolicy: Unsupported value": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "* * * * ?",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: api.PodTemplateSpec{
							Spec: api.PodSpec{
								RestartPolicy: "Invalid",
								DNSPolicy:     api.DNSClusterFirst,
								Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
							},
						},
					},
				},
			},
		},
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.TTLAfterFinished) {
		errorCases["spec.jobTemplate.spec.ttlSecondsAfterFinished:must be greater than or equal to 0"] = batch.CronJob{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "* * * * ?",
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						TTLSecondsAfterFinished: &negative,
						Template:                validPodTemplateSpec,
					},
				},
			},
		}
	}

	for k, v := range errorCases {
		errs := ValidateCronJob(&v, corevalidation.PodValidationOptions{})
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		} else {
			s := strings.Split(k, ":")
			err := errs[0]
			if err.Field != s[0] || !strings.Contains(err.Error(), s[1]) {
				t.Errorf("unexpected error: %v, expected: %s", err, k)
			}
		}

		// Update validation should fail all failure cases other than the 52 character name limit
		// copy to avoid polluting the testcase object, set a resourceVersion to allow validating update, and test a no-op update
		v = *v.DeepCopy()
		v.ResourceVersion = "1"
		errs = ValidateCronJobUpdate(&v, &v, corevalidation.PodValidationOptions{})
		if len(errs) == 0 {
			if k == "metadata.name: must be no more than 52 characters" {
				continue
			}
			t.Errorf("expected failure for %s", k)
		} else {
			s := strings.Split(k, ":")
			err := errs[0]
			if err.Field != s[0] || !strings.Contains(err.Error(), s[1]) {
				t.Errorf("unexpected error: %v, expected: %s", err, k)
			}
		}
	}
}

func completionModePtr(m batch.CompletionMode) *batch.CompletionMode {
	return &m
}
