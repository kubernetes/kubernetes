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

package v1_test

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	batchv1 "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/utils/pointer"

	. "k8s.io/kubernetes/pkg/apis/batch/v1"
)

func TestSetDefaultJob(t *testing.T) {
	defaultLabels := map[string]string{"default": "default"}
	tests := map[string]struct {
		original     *batchv1.Job
		expected     *batchv1.Job
		expectLabels bool
	}{
		"All unspecified -> sets all to default values": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:    pointer.Int32Ptr(1),
					Parallelism:    pointer.Int32Ptr(1),
					BackoffLimit:   pointer.Int32Ptr(6),
					CompletionMode: completionModePtr(batchv1.NonIndexedCompletion),
					Suspend:        pointer.BoolPtr(false),
				},
			},
			expectLabels: true,
		},
		"All unspecified, suspend job enabled -> sets all to default values": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:    pointer.Int32Ptr(1),
					Parallelism:    pointer.Int32Ptr(1),
					BackoffLimit:   pointer.Int32Ptr(6),
					CompletionMode: completionModePtr(batchv1.NonIndexedCompletion),
					Suspend:        pointer.BoolPtr(false),
				},
			},
			expectLabels: true,
		},
		"suspend set, everything else is defaulted": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Suspend: pointer.BoolPtr(true),
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:    pointer.Int32Ptr(1),
					Parallelism:    pointer.Int32Ptr(1),
					BackoffLimit:   pointer.Int32Ptr(6),
					CompletionMode: completionModePtr(batchv1.NonIndexedCompletion),
					Suspend:        pointer.BoolPtr(true),
				},
			},
			expectLabels: true,
		},
		"All unspecified -> all pointers are defaulted and no default labels": {
			original: &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"mylabel": "myvalue"},
				},
				Spec: batchv1.JobSpec{
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:    pointer.Int32Ptr(1),
					Parallelism:    pointer.Int32Ptr(1),
					BackoffLimit:   pointer.Int32Ptr(6),
					CompletionMode: completionModePtr(batchv1.NonIndexedCompletion),
					Suspend:        pointer.BoolPtr(false),
				},
			},
		},
		"WQ: Parallelism explicitly 0 and completions unset -> BackoffLimit is defaulted": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Parallelism: pointer.Int32Ptr(0),
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Parallelism:    pointer.Int32Ptr(0),
					BackoffLimit:   pointer.Int32Ptr(6),
					CompletionMode: completionModePtr(batchv1.NonIndexedCompletion),
					Suspend:        pointer.BoolPtr(false),
				},
			},
			expectLabels: true,
		},
		"WQ: Parallelism explicitly 2 and completions unset -> BackoffLimit is defaulted": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Parallelism: pointer.Int32Ptr(2),
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Parallelism:    pointer.Int32Ptr(2),
					BackoffLimit:   pointer.Int32Ptr(6),
					CompletionMode: completionModePtr(batchv1.NonIndexedCompletion),
					Suspend:        pointer.BoolPtr(false),
				},
			},
			expectLabels: true,
		},
		"Completions explicitly 2 and others unset -> parallelism and BackoffLimit are defaulted": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions: pointer.Int32Ptr(2),
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:    pointer.Int32Ptr(2),
					Parallelism:    pointer.Int32Ptr(1),
					BackoffLimit:   pointer.Int32Ptr(6),
					CompletionMode: completionModePtr(batchv1.NonIndexedCompletion),
					Suspend:        pointer.BoolPtr(false),
				},
			},
			expectLabels: true,
		},
		"BackoffLimit explicitly 5 and others unset -> parallelism and completions are defaulted": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					BackoffLimit: pointer.Int32Ptr(5),
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:    pointer.Int32Ptr(1),
					Parallelism:    pointer.Int32Ptr(1),
					BackoffLimit:   pointer.Int32Ptr(5),
					CompletionMode: completionModePtr(batchv1.NonIndexedCompletion),
					Suspend:        pointer.BoolPtr(false),
				},
			},
			expectLabels: true,
		},
		"All set -> no change": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:    pointer.Int32Ptr(8),
					Parallelism:    pointer.Int32Ptr(9),
					BackoffLimit:   pointer.Int32Ptr(10),
					CompletionMode: completionModePtr(batchv1.NonIndexedCompletion),
					Suspend:        pointer.BoolPtr(false),
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:    pointer.Int32Ptr(8),
					Parallelism:    pointer.Int32Ptr(9),
					BackoffLimit:   pointer.Int32Ptr(10),
					CompletionMode: completionModePtr(batchv1.NonIndexedCompletion),
					Suspend:        pointer.BoolPtr(false),
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expectLabels: true,
		},
		"All set, flipped -> no change": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:    pointer.Int32Ptr(11),
					Parallelism:    pointer.Int32Ptr(10),
					BackoffLimit:   pointer.Int32Ptr(9),
					CompletionMode: completionModePtr(batchv1.IndexedCompletion),
					Suspend:        pointer.BoolPtr(true),
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:    pointer.Int32Ptr(11),
					Parallelism:    pointer.Int32Ptr(10),
					BackoffLimit:   pointer.Int32Ptr(9),
					CompletionMode: completionModePtr(batchv1.IndexedCompletion),
					Suspend:        pointer.BoolPtr(true),
				},
			},
			expectLabels: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			original := test.original
			expected := test.expected
			obj2 := roundTrip(t, runtime.Object(original))
			actual, ok := obj2.(*batchv1.Job)
			if !ok {
				t.Fatalf("Unexpected object: %v", actual)
			}

			if diff := cmp.Diff(expected.Spec.Suspend, actual.Spec.Suspend); diff != "" {
				t.Errorf(".spec.suspend does not match; -want,+got:\n%s", diff)
			}
			validateDefaultInt32(t, "Completions", actual.Spec.Completions, expected.Spec.Completions)
			validateDefaultInt32(t, "Parallelism", actual.Spec.Parallelism, expected.Spec.Parallelism)
			validateDefaultInt32(t, "BackoffLimit", actual.Spec.BackoffLimit, expected.Spec.BackoffLimit)

			if test.expectLabels != reflect.DeepEqual(actual.Labels, actual.Spec.Template.Labels) {
				if test.expectLabels {
					t.Errorf("Expected labels: %v, got: %v", actual.Spec.Template.Labels, actual.Labels)
				} else {
					t.Errorf("Unexpected equality: %v", actual.Labels)
				}
			}
			if diff := cmp.Diff(expected.Spec.CompletionMode, actual.Spec.CompletionMode); diff != "" {
				t.Errorf("Unexpected CompletionMode (-want,+got):\n%s", diff)
			}
		})
	}
}

func validateDefaultInt32(t *testing.T, field string, actual *int32, expected *int32) {
	if (actual == nil) != (expected == nil) {
		t.Errorf("Got different *%s than expected: %v %v", field, actual, expected)
	}
	if actual != nil && expected != nil {
		if *actual != *expected {
			t.Errorf("Got different %s than expected: %d %d", field, *actual, *expected)
		}
	}
}

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	data, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(SchemeGroupVersion), obj)
	if err != nil {
		t.Errorf("%v\n %#v", err, obj)
		return nil
	}
	obj2, err := runtime.Decode(legacyscheme.Codecs.UniversalDecoder(), data)
	if err != nil {
		t.Errorf("%v\nData: %s\nSource: %#v", err, string(data), obj)
		return nil
	}
	obj3 := reflect.New(reflect.TypeOf(obj).Elem()).Interface().(runtime.Object)
	err = legacyscheme.Scheme.Convert(obj2, obj3, nil)
	if err != nil {
		t.Errorf("%v\nSource: %#v", err, obj2)
		return nil
	}
	return obj3
}

func TestSetDefaultCronJob(t *testing.T) {
	tests := map[string]struct {
		original *batchv1.CronJob
		expected *batchv1.CronJob
	}{
		"empty batchv1.CronJob should default batchv1.ConcurrencyPolicy and Suspend": {
			original: &batchv1.CronJob{},
			expected: &batchv1.CronJob{
				Spec: batchv1.CronJobSpec{
					ConcurrencyPolicy:          batchv1.AllowConcurrent,
					Suspend:                    pointer.BoolPtr(false),
					SuccessfulJobsHistoryLimit: pointer.Int32Ptr(3),
					FailedJobsHistoryLimit:     pointer.Int32Ptr(1),
				},
			},
		},
		"set fields should not be defaulted": {
			original: &batchv1.CronJob{
				Spec: batchv1.CronJobSpec{
					ConcurrencyPolicy:          batchv1.ForbidConcurrent,
					Suspend:                    pointer.BoolPtr(true),
					SuccessfulJobsHistoryLimit: pointer.Int32Ptr(5),
					FailedJobsHistoryLimit:     pointer.Int32Ptr(5),
				},
			},
			expected: &batchv1.CronJob{
				Spec: batchv1.CronJobSpec{
					ConcurrencyPolicy:          batchv1.ForbidConcurrent,
					Suspend:                    pointer.BoolPtr(true),
					SuccessfulJobsHistoryLimit: pointer.Int32Ptr(5),
					FailedJobsHistoryLimit:     pointer.Int32Ptr(5),
				},
			},
		},
	}

	for name, test := range tests {
		original := test.original
		expected := test.expected
		obj2 := roundTrip(t, runtime.Object(original))
		actual, ok := obj2.(*batchv1.CronJob)
		if !ok {
			t.Errorf("%s: unexpected object: %v", name, actual)
			t.FailNow()
		}
		if actual.Spec.ConcurrencyPolicy != expected.Spec.ConcurrencyPolicy {
			t.Errorf("%s: got different concurrencyPolicy than expected: %v %v", name, actual.Spec.ConcurrencyPolicy, expected.Spec.ConcurrencyPolicy)
		}
		if *actual.Spec.Suspend != *expected.Spec.Suspend {
			t.Errorf("%s: got different suspend than expected: %v %v", name, *actual.Spec.Suspend, *expected.Spec.Suspend)
		}
		if *actual.Spec.SuccessfulJobsHistoryLimit != *expected.Spec.SuccessfulJobsHistoryLimit {
			t.Errorf("%s: got different successfulJobsHistoryLimit than expected: %v %v", name, *actual.Spec.SuccessfulJobsHistoryLimit, *expected.Spec.SuccessfulJobsHistoryLimit)
		}
		if *actual.Spec.FailedJobsHistoryLimit != *expected.Spec.FailedJobsHistoryLimit {
			t.Errorf("%s: got different failedJobsHistoryLimit than expected: %v %v", name, *actual.Spec.FailedJobsHistoryLimit, *expected.Spec.FailedJobsHistoryLimit)
		}
	}
}

func completionModePtr(m batchv1.CompletionMode) *batchv1.CompletionMode {
	return &m
}
