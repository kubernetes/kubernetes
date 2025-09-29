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
	"math"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"

	. "k8s.io/kubernetes/pkg/apis/batch/v1"
)

func TestSetDefaultJob(t *testing.T) {
	defaultLabels := map[string]string{"default": "default"}
	validPodTemplateSpec := v1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
	}
	tests := map[string]struct {
		original                   *batchv1.Job
		expected                   *batchv1.Job
		expectLabels               bool
		enablePodReplacementPolicy bool
	}{
		"Pod failure policy with some field values unspecified -> set default values": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
					PodFailurePolicy: &batchv1.PodFailurePolicy{
						Rules: []batchv1.PodFailurePolicyRule{
							{
								Action: batchv1.PodFailurePolicyActionFailJob,
								OnPodConditions: []batchv1.PodFailurePolicyOnPodConditionsPattern{
									{
										Type:   v1.DisruptionTarget,
										Status: v1.ConditionTrue,
									},
									{
										Type:   v1.PodConditionType("MemoryLimitExceeded"),
										Status: v1.ConditionFalse,
									},
									{
										Type: v1.PodConditionType("DiskLimitExceeded"),
									},
								},
							},
							{
								Action: batchv1.PodFailurePolicyActionFailJob,
								OnExitCodes: &batchv1.PodFailurePolicyOnExitCodesRequirement{
									Operator: batchv1.PodFailurePolicyOnExitCodesOpIn,
									Values:   []int32{1},
								},
							},
							{
								Action: batchv1.PodFailurePolicyActionFailJob,
								OnPodConditions: []batchv1.PodFailurePolicyOnPodConditionsPattern{
									{
										Type: v1.DisruptionTarget,
									},
								},
							},
						},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:    ptr.To[int32](1),
					Parallelism:    ptr.To[int32](1),
					BackoffLimit:   ptr.To[int32](6),
					CompletionMode: ptr.To(batchv1.NonIndexedCompletion),
					Suspend:        ptr.To(false),
					ManualSelector: ptr.To(false),
					PodFailurePolicy: &batchv1.PodFailurePolicy{
						Rules: []batchv1.PodFailurePolicyRule{
							{
								Action: batchv1.PodFailurePolicyActionFailJob,
								OnPodConditions: []batchv1.PodFailurePolicyOnPodConditionsPattern{
									{
										Type:   v1.DisruptionTarget,
										Status: v1.ConditionTrue,
									},
									{
										Type:   v1.PodConditionType("MemoryLimitExceeded"),
										Status: v1.ConditionFalse,
									},
									{
										Type:   v1.PodConditionType("DiskLimitExceeded"),
										Status: v1.ConditionTrue,
									},
								},
							},
							{
								Action: batchv1.PodFailurePolicyActionFailJob,
								OnExitCodes: &batchv1.PodFailurePolicyOnExitCodesRequirement{
									Operator: batchv1.PodFailurePolicyOnExitCodesOpIn,
									Values:   []int32{1},
								},
							},
							{
								Action: batchv1.PodFailurePolicyActionFailJob,
								OnPodConditions: []batchv1.PodFailurePolicyOnPodConditionsPattern{
									{
										Type:   v1.DisruptionTarget,
										Status: v1.ConditionTrue,
									},
								},
							},
						},
					},
				},
			},
			expectLabels: true,
		},
		"Pod failure policy and defaulting for pod replacement policy": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
					PodFailurePolicy: &batchv1.PodFailurePolicy{
						Rules: []batchv1.PodFailurePolicyRule{
							{
								Action: batchv1.PodFailurePolicyActionFailJob,
								OnExitCodes: &batchv1.PodFailurePolicyOnExitCodesRequirement{
									Operator: batchv1.PodFailurePolicyOnExitCodesOpIn,
									Values:   []int32{1},
								},
							},
						},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:          ptr.To[int32](1),
					Parallelism:          ptr.To[int32](1),
					BackoffLimit:         ptr.To[int32](6),
					CompletionMode:       ptr.To(batchv1.NonIndexedCompletion),
					Suspend:              ptr.To(false),
					PodReplacementPolicy: ptr.To(batchv1.Failed),
					ManualSelector:       ptr.To(false),
					PodFailurePolicy: &batchv1.PodFailurePolicy{
						Rules: []batchv1.PodFailurePolicyRule{
							{
								Action: batchv1.PodFailurePolicyActionFailJob,
								OnExitCodes: &batchv1.PodFailurePolicyOnExitCodesRequirement{
									Operator: batchv1.PodFailurePolicyOnExitCodesOpIn,
									Values:   []int32{1},
								},
							},
						},
					},
				},
			},
			expectLabels:               true,
			enablePodReplacementPolicy: true,
		},
		"All unspecified and podReplacementPolicyEnabled -> sets all to default values": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:          ptr.To[int32](1),
					Parallelism:          ptr.To[int32](1),
					BackoffLimit:         ptr.To[int32](6),
					CompletionMode:       ptr.To(batchv1.NonIndexedCompletion),
					Suspend:              ptr.To(false),
					PodReplacementPolicy: ptr.To(batchv1.TerminatingOrFailed),
					ManualSelector:       ptr.To(false),
				},
			},
			expectLabels:               true,
			enablePodReplacementPolicy: true,
		},
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
					Completions:    ptr.To[int32](1),
					Parallelism:    ptr.To[int32](1),
					BackoffLimit:   ptr.To[int32](6),
					CompletionMode: ptr.To(batchv1.NonIndexedCompletion),
					Suspend:        ptr.To(false),
					ManualSelector: ptr.To(false),
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
					Completions:    ptr.To[int32](1),
					Parallelism:    ptr.To[int32](1),
					BackoffLimit:   ptr.To[int32](6),
					CompletionMode: ptr.To(batchv1.NonIndexedCompletion),
					Suspend:        ptr.To(false),
					ManualSelector: ptr.To(false),
				},
			},
			expectLabels: true,
		},
		"suspend set, everything else is defaulted": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Suspend: ptr.To(true),
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:    ptr.To[int32](1),
					Parallelism:    ptr.To[int32](1),
					BackoffLimit:   ptr.To[int32](6),
					CompletionMode: ptr.To(batchv1.NonIndexedCompletion),
					Suspend:        ptr.To(true),
					ManualSelector: ptr.To(false),
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
					Completions:    ptr.To[int32](1),
					Parallelism:    ptr.To[int32](1),
					BackoffLimit:   ptr.To[int32](6),
					CompletionMode: ptr.To(batchv1.NonIndexedCompletion),
					Suspend:        ptr.To(false),
					ManualSelector: ptr.To(false),
				},
			},
		},
		"WQ: Parallelism explicitly 0 and completions unset -> BackoffLimit is defaulted": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Parallelism: ptr.To[int32](0),
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Parallelism:    ptr.To[int32](0),
					BackoffLimit:   ptr.To[int32](6),
					CompletionMode: ptr.To(batchv1.NonIndexedCompletion),
					Suspend:        ptr.To(false),
					ManualSelector: ptr.To(false),
				},
			},
			expectLabels: true,
		},
		"WQ: Parallelism explicitly 2 and completions unset -> BackoffLimit is defaulted": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Parallelism: ptr.To[int32](2),
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Parallelism:    ptr.To[int32](2),
					BackoffLimit:   ptr.To[int32](6),
					CompletionMode: ptr.To(batchv1.NonIndexedCompletion),
					Suspend:        ptr.To(false),
					ManualSelector: ptr.To(false),
				},
			},
			expectLabels: true,
		},
		"Completions explicitly 2 and others unset -> parallelism and BackoffLimit are defaulted": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions: ptr.To[int32](2),
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:    ptr.To[int32](2),
					Parallelism:    ptr.To[int32](1),
					BackoffLimit:   ptr.To[int32](6),
					CompletionMode: ptr.To(batchv1.NonIndexedCompletion),
					Suspend:        ptr.To(false),
					ManualSelector: ptr.To(false),
				},
			},
			expectLabels: true,
		},
		"BackoffLimit explicitly 5 and others unset -> parallelism and completions are defaulted": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					BackoffLimit: ptr.To[int32](5),
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:    ptr.To[int32](1),
					Parallelism:    ptr.To[int32](1),
					BackoffLimit:   ptr.To[int32](5),
					CompletionMode: ptr.To(batchv1.NonIndexedCompletion),
					Suspend:        ptr.To(false),
					ManualSelector: ptr.To(false),
				},
			},
			expectLabels: true,
		},
		"All set -> no change": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:          ptr.To[int32](8),
					Parallelism:          ptr.To[int32](9),
					BackoffLimit:         ptr.To[int32](10),
					CompletionMode:       ptr.To(batchv1.NonIndexedCompletion),
					Suspend:              ptr.To(false),
					PodReplacementPolicy: ptr.To(batchv1.TerminatingOrFailed),
					ManualSelector:       ptr.To(false),
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:          ptr.To[int32](8),
					Parallelism:          ptr.To[int32](9),
					BackoffLimit:         ptr.To[int32](10),
					CompletionMode:       ptr.To(batchv1.NonIndexedCompletion),
					Suspend:              ptr.To(false),
					PodReplacementPolicy: ptr.To(batchv1.TerminatingOrFailed),
					ManualSelector:       ptr.To(false),
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
					Completions:          ptr.To[int32](11),
					Parallelism:          ptr.To[int32](10),
					BackoffLimit:         ptr.To[int32](9),
					CompletionMode:       ptr.To(batchv1.IndexedCompletion),
					Suspend:              ptr.To(true),
					PodReplacementPolicy: ptr.To(batchv1.Failed),
					ManualSelector:       ptr.To(true),
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: defaultLabels},
					},
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:          ptr.To[int32](11),
					Parallelism:          ptr.To[int32](10),
					BackoffLimit:         ptr.To[int32](9),
					CompletionMode:       ptr.To(batchv1.IndexedCompletion),
					Suspend:              ptr.To(true),
					PodReplacementPolicy: ptr.To(batchv1.Failed),
					ManualSelector:       ptr.To(true),
				},
			},
			expectLabels: true,
		},
		"BackoffLimitPerIndex specified, but no BackoffLimit -> default BackoffLimit to max int32": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:          ptr.To[int32](11),
					Parallelism:          ptr.To[int32](10),
					BackoffLimitPerIndex: ptr.To[int32](1),
					CompletionMode:       ptr.To(batchv1.IndexedCompletion),
					Template:             validPodTemplateSpec,
					Suspend:              ptr.To(true),
					ManualSelector:       ptr.To(false),
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:          ptr.To[int32](11),
					Parallelism:          ptr.To[int32](10),
					BackoffLimit:         ptr.To[int32](math.MaxInt32),
					BackoffLimitPerIndex: ptr.To[int32](1),
					CompletionMode:       ptr.To(batchv1.IndexedCompletion),
					Template:             validPodTemplateSpec,
					Suspend:              ptr.To(true),
					ManualSelector:       ptr.To(false),
				},
			},
			expectLabels: true,
		},
		"BackoffLimitPerIndex and BackoffLimit specified -> no change": {
			original: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:          ptr.To[int32](11),
					Parallelism:          ptr.To[int32](10),
					BackoffLimit:         ptr.To[int32](3),
					BackoffLimitPerIndex: ptr.To[int32](1),
					CompletionMode:       ptr.To(batchv1.IndexedCompletion),
					Template:             validPodTemplateSpec,
					Suspend:              ptr.To(true),
					ManualSelector:       ptr.To(true),
				},
			},
			expected: &batchv1.Job{
				Spec: batchv1.JobSpec{
					Completions:          ptr.To[int32](11),
					Parallelism:          ptr.To[int32](10),
					BackoffLimit:         ptr.To[int32](3),
					BackoffLimitPerIndex: ptr.To[int32](1),
					CompletionMode:       ptr.To(batchv1.IndexedCompletion),
					Template:             validPodTemplateSpec,
					Suspend:              ptr.To(true),
					ManualSelector:       ptr.To(true),
				},
			},
			expectLabels: true,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if !test.enablePodReplacementPolicy {
				// TODO: this will be removed in 1.37.
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, utilversion.MustParse("1.33"))
			}
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobPodReplacementPolicy, test.enablePodReplacementPolicy)
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

			if diff := cmp.Diff(expected.Spec.PodFailurePolicy, actual.Spec.PodFailurePolicy); diff != "" {
				t.Errorf("unexpected diff in errors (-want, +got):\n%s", diff)
			}
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
			if diff := cmp.Diff(expected.Spec.PodReplacementPolicy, actual.Spec.PodReplacementPolicy); diff != "" {
				t.Errorf("Unexpected PodReplacementPolicy (-want,+got):\n%s", diff)
			}
			if diff := cmp.Diff(expected.Spec.ManualSelector, actual.Spec.ManualSelector); diff != "" {
				t.Errorf("Unexpected ManualSelector (-want,+got):\n%s", diff)
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
					Suspend:                    ptr.To(false),
					SuccessfulJobsHistoryLimit: ptr.To[int32](3),
					FailedJobsHistoryLimit:     ptr.To[int32](1),
				},
			},
		},
		"set fields should not be defaulted": {
			original: &batchv1.CronJob{
				Spec: batchv1.CronJobSpec{
					ConcurrencyPolicy:          batchv1.ForbidConcurrent,
					Suspend:                    ptr.To(true),
					SuccessfulJobsHistoryLimit: ptr.To[int32](5),
					FailedJobsHistoryLimit:     ptr.To[int32](5),
				},
			},
			expected: &batchv1.CronJob{
				Spec: batchv1.CronJobSpec{
					ConcurrencyPolicy:          batchv1.ForbidConcurrent,
					Suspend:                    ptr.To(true),
					SuccessfulJobsHistoryLimit: ptr.To[int32](5),
					FailedJobsHistoryLimit:     ptr.To[int32](5),
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
