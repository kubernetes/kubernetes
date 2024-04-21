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
	"errors"
	_ "time/tzdata"

	"fmt"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	"k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	corevalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/utils/pointer"
	"k8s.io/utils/ptr"
)

var (
	timeZoneEmpty      = ""
	timeZoneLocal      = "LOCAL"
	timeZoneUTC        = "UTC"
	timeZoneCorrect    = "Europe/Rome"
	timeZoneBadPrefix  = " Europe/Rome"
	timeZoneBadSuffix  = "Europe/Rome "
	timeZoneBadName    = "Europe/InvalidRome"
	timeZoneEmptySpace = " "
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
		Spec: podtest.MakePod("",
			podtest.SetRestartPolicy(api.RestartPolicyOnFailure),
			podtest.SetContainers(podtest.MakeContainer("abc")),
		).Spec,
	}
}

func getValidGeneratedSelector() *metav1.LabelSelector {
	return &metav1.LabelSelector{
		MatchLabels: map[string]string{batch.ControllerUidLabel: "1a2b3c", batch.LegacyControllerUidLabel: "1a2b3c", batch.JobNameLabel: "myjob", batch.LegacyJobNameLabel: "myjob"},
	}
}

func getValidPodTemplateSpecForGenerated(selector *metav1.LabelSelector) api.PodTemplateSpec {
	return api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: selector.MatchLabels,
		},
		Spec: podtest.MakePod("",
			podtest.SetRestartPolicy(api.RestartPolicyOnFailure),
			podtest.SetContainers(podtest.MakeContainer("abc")),
			podtest.SetInitContainers(podtest.MakeContainer("def")),
		).Spec,
	}
}

func TestValidateJob(t *testing.T) {
	validJobObjectMeta := metav1.ObjectMeta{
		Name:      "myjob",
		Namespace: metav1.NamespaceDefault,
		UID:       types.UID("1a2b3c"),
	}
	validManualSelector := getValidManualSelector()
	failedPodReplacement := batch.Failed
	terminatingOrFailedPodReplacement := batch.TerminatingOrFailed
	validPodTemplateSpecForManual := getValidPodTemplateSpecForManual(validManualSelector)
	validGeneratedSelector := getValidGeneratedSelector()
	validPodTemplateSpecForGenerated := getValidPodTemplateSpecForGenerated(validGeneratedSelector)
	validPodTemplateSpecForGeneratedRestartPolicyNever := getValidPodTemplateSpecForGenerated(validGeneratedSelector)
	validPodTemplateSpecForGeneratedRestartPolicyNever.Spec.RestartPolicy = api.RestartPolicyNever
	validHostNetPodTemplateSpec := func() api.PodTemplateSpec {
		spec := getValidPodTemplateSpecForGenerated(validGeneratedSelector)
		spec.Spec.SecurityContext = &api.PodSecurityContext{
			HostNetwork: true,
		}
		spec.Spec.Containers[0].Ports = []api.ContainerPort{{
			ContainerPort: 12345,
			Protocol:      api.ProtocolTCP,
		}}
		return spec
	}()

	successCases := map[string]struct {
		opts JobValidationOptions
		job  batch.Job
	}{
		"valid success policy": {
			opts: JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: []batch.SuccessPolicyRule{
							{
								SucceededCount:   ptr.To[int32](1),
								SucceededIndexes: ptr.To("0,2,4"),
							},
							{
								SucceededIndexes: ptr.To("1,3,5-9"),
							},
						},
					},
				},
			},
		},
		"valid pod failure policy": {
			opts: JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
								Type:   api.DisruptionTarget,
								Status: api.ConditionTrue,
							}},
						}, {
							Action: batch.PodFailurePolicyActionFailJob,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
								Type:   api.PodConditionType("CustomConditionType"),
								Status: api.ConditionFalse,
							}},
						}, {
							Action: batch.PodFailurePolicyActionCount,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								ContainerName: pointer.String("abc"),
								Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
								Values:        []int32{1, 2, 3},
							},
						}, {
							Action: batch.PodFailurePolicyActionIgnore,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								ContainerName: pointer.String("def"),
								Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
								Values:        []int32{4},
							},
						}, {
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpNotIn,
								Values:   []int32{5, 6, 7},
							},
						}},
					},
				},
			},
		},
		"valid pod failure policy with FailIndex": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					Completions:          pointer.Int32(2),
					BackoffLimitPerIndex: pointer.Int32(1),
					Selector:             validGeneratedSelector,
					ManualSelector:       pointer.Bool(true),
					Template:             validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailIndex,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{10},
							},
						}},
					},
				},
			},
		},
		"valid manual selector": {
			opts: JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "myjob",
					Namespace:   metav1.NamespaceDefault,
					UID:         types.UID("1a2b3c"),
					Annotations: map[string]string{"foo": "bar"},
				},
				Spec: batch.JobSpec{
					Selector:       validManualSelector,
					ManualSelector: pointer.Bool(true),
					Template:       validPodTemplateSpecForManual,
				},
			},
		},
		"valid generated selector": {
			opts: JobValidationOptions{RequirePrefixedLabels: true},
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
		"valid pod replacement": {
			opts: JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
					PodReplacementPolicy: &terminatingOrFailedPodReplacement,
				},
			},
		},
		"valid pod replacement with failed": {
			opts: JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
					PodReplacementPolicy: &failedPodReplacement,
				},
			},
		},
		"valid hostnet": {
			opts: JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validHostNetPodTemplateSpec,
				},
			},
		},
		"valid NonIndexed completion mode": {
			opts: JobValidationOptions{RequirePrefixedLabels: true},
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
			opts: JobValidationOptions{RequirePrefixedLabels: true},
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
					Completions:    pointer.Int32(2),
					Parallelism:    pointer.Int32(100000),
				},
			},
		},
		"valid parallelism and maxFailedIndexes for high completions when backoffLimitPerIndex is used": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Completions:          pointer.Int32(100_000),
					Parallelism:          pointer.Int32(100_000),
					MaxFailedIndexes:     pointer.Int32(100_000),
					BackoffLimitPerIndex: pointer.Int32(1),
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"valid parallelism and maxFailedIndexes for unlimited completions when backoffLimitPerIndex is used": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Completions:          pointer.Int32(1_000_000_000),
					Parallelism:          pointer.Int32(10_000),
					MaxFailedIndexes:     pointer.Int32(10_000),
					BackoffLimitPerIndex: pointer.Int32(1),
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"valid job tracking annotation": {
			opts: JobValidationOptions{
				RequirePrefixedLabels: true,
			},
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
		"valid batch labels": {
			opts: JobValidationOptions{
				RequirePrefixedLabels: true,
			},
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
		"do not allow new batch labels": {
			opts: JobValidationOptions{
				RequirePrefixedLabels: false,
			},
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{batch.LegacyControllerUidLabel: "1a2b3c"},
					},
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{batch.LegacyControllerUidLabel: "1a2b3c", batch.LegacyJobNameLabel: "myjob"},
						},
						Spec: podtest.MakePod("",
							podtest.SetRestartPolicy(api.RestartPolicyOnFailure),
							podtest.SetContainers(podtest.MakeContainer("abc")),
							podtest.SetInitContainers(podtest.MakeContainer("def")),
						).Spec,
					},
				},
			},
		},
		"valid managedBy field": {
			opts: JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:  validGeneratedSelector,
					Template:  validPodTemplateSpecForGenerated,
					ManagedBy: ptr.To("example.com/foo"),
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
	errorCases := map[string]struct {
		opts JobValidationOptions
		job  batch.Job
	}{
		`spec.managedBy: Too long: may not be longer than 63`: {
			opts: JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:  validGeneratedSelector,
					Template:  validPodTemplateSpecForGenerated,
					ManagedBy: ptr.To("example.com/" + strings.Repeat("x", 60)),
				},
			},
		},
		`spec.managedBy: Invalid value: "invalid custom controller name": must be a domain-prefixed path (such as "acme.io/foo")`: {
			opts: JobValidationOptions{RequirePrefixedLabels: true},
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:  validGeneratedSelector,
					Template:  validPodTemplateSpecForGenerated,
					ManagedBy: ptr.To("invalid custom controller name"),
				},
			},
		},
		`spec.successPolicy: Invalid value: batch.SuccessPolicy{Rules:[]batch.SuccessPolicyRule{}}: requires indexed completion mode`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: []batch.SuccessPolicyRule{},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.successPolicy.rules: Required value: at least one rules must be specified when the successPolicy is specified`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy:  &batch.SuccessPolicy{},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.successPolicy.rules[0]: Required value: at least one of succeededCount or succeededIndexes must be specified`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: []batch.SuccessPolicyRule{{
							SucceededCount:   nil,
							SucceededIndexes: nil,
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.successPolicy.rules[0].succeededIndexes: Invalid value: "invalid-format": error parsing succeededIndexes: cannot convert string to integer for index: "invalid"`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: []batch.SuccessPolicyRule{{
							SucceededIndexes: ptr.To("invalid-format"),
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.successPolicy.rules[0].succeededIndexes: Too long: must have at most 65536 bytes`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: []batch.SuccessPolicyRule{{
							SucceededIndexes: ptr.To(strings.Repeat("1", maxJobSuccessPolicySucceededIndexesLimit+1)),
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.successPolicy.rules[0].succeededCount: must be greater than or equal to 0`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: []batch.SuccessPolicyRule{{
							SucceededCount: ptr.To[int32](-1),
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.successPolicy.rules[0].succeededCount: Invalid value: 6: must be less than or equal to 5 (the number of specified completions)`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: []batch.SuccessPolicyRule{{
							SucceededCount: ptr.To[int32](6),
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.successPolicy.rules[0].succeededCount: Invalid value: 4: must be less than or equal to 3 (the number of indexes in the specified succeededIndexes field)`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: []batch.SuccessPolicyRule{{
							SucceededCount:   ptr.To[int32](4),
							SucceededIndexes: ptr.To("0-2"),
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.successPolicy.rules: Too many: 21: must have at most 20 items`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: func() []batch.SuccessPolicyRule {
							var rules []batch.SuccessPolicyRule
							for i := 0; i < 21; i++ {
								rules = append(rules, batch.SuccessPolicyRule{
									SucceededCount: ptr.To[int32](5),
								})
							}
							return rules
						}(),
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules[0]: Invalid value: specifying one of OnExitCodes and OnPodConditions is required`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailJob,
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.values[1]: Duplicate value: 11`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{11, 11},
							},
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.values: Too many: 256: must have at most 255 items`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values: func() (values []int32) {
									tooManyValues := make([]int32, maxPodFailurePolicyOnExitCodesValues+1)
									for i := range tooManyValues {
										tooManyValues[i] = int32(i)
									}
									return tooManyValues
								}(),
							},
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules: Too many: 21: must have at most 20 items`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: func() []batch.PodFailurePolicyRule {
							tooManyRules := make([]batch.PodFailurePolicyRule, maxPodFailurePolicyRules+1)
							for i := range tooManyRules {
								tooManyRules[i] = batch.PodFailurePolicyRule{
									Action: batch.PodFailurePolicyActionFailJob,
									OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
										Operator: batch.PodFailurePolicyOnExitCodesOpIn,
										Values:   []int32{int32(i + 1)},
									},
								}
							}
							return tooManyRules
						}(),
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules[0].onPodConditions: Too many: 21: must have at most 20 items`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailJob,
							OnPodConditions: func() []batch.PodFailurePolicyOnPodConditionsPattern {
								tooManyPatterns := make([]batch.PodFailurePolicyOnPodConditionsPattern, maxPodFailurePolicyOnPodConditionsPatterns+1)
								for i := range tooManyPatterns {
									tooManyPatterns[i] = batch.PodFailurePolicyOnPodConditionsPattern{
										Type:   api.PodConditionType(fmt.Sprintf("CustomType_%d", i)),
										Status: api.ConditionTrue,
									}
								}
								return tooManyPatterns
							}(),
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.values[2]: Duplicate value: 13`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{12, 13, 13, 13},
							},
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.values: Invalid value: []int32{19, 11}: must be ordered`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{19, 11},
							},
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.values: Invalid value: []int32{}: at least one value is required`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{},
							},
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules[0].action: Required value: valid values: ["Count" "FailIndex" "FailJob" "Ignore"]`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: "",
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{1, 2, 3},
							},
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.operator: Required value: valid values: ["In" "NotIn"]`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: "",
								Values:   []int32{1, 2, 3},
							},
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules[0]: Invalid value: specifying both OnExitCodes and OnPodConditions is not supported`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								ContainerName: pointer.String("abc"),
								Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
								Values:        []int32{1, 2, 3},
							},
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
								Type:   api.DisruptionTarget,
								Status: api.ConditionTrue,
							}},
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.values[1]: Invalid value: 0: must not be 0 for the In operator`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{1, 0, 2},
							},
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules[1].onExitCodes.containerName: Invalid value: "xyz": must be one of the container or initContainer names in the pod template`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								ContainerName: pointer.String("abc"),
								Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
								Values:        []int32{1, 2, 3},
							},
						}, {
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								ContainerName: pointer.String("xyz"),
								Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
								Values:        []int32{5, 6, 7},
							},
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules[0].action: Unsupported value: "UnknownAction": supported values: "Count", "FailIndex", "FailJob", "Ignore"`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: "UnknownAction",
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								ContainerName: pointer.String("abc"),
								Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
								Values:        []int32{1, 2, 3},
							},
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.operator: Unsupported value: "UnknownOperator": supported values: "In", "NotIn"`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: "UnknownOperator",
								Values:   []int32{1, 2, 3},
							},
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules[0].onPodConditions[0].status: Required value: valid values: ["False" "True" "Unknown"]`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
								Type: api.DisruptionTarget,
							}},
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules[0].onPodConditions[0].status: Unsupported value: "UnknownStatus": supported values: "False", "True", "Unknown"`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
								Type:   api.DisruptionTarget,
								Status: "UnknownStatus",
							}},
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules[0].onPodConditions[0].type: Invalid value: "": name part must be non-empty`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
								Status: api.ConditionTrue,
							}},
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podFailurePolicy.rules[0].onPodConditions[0].type: Invalid value: "Invalid Condition Type": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
								Type:   api.PodConditionType("Invalid Condition Type"),
								Status: api.ConditionTrue,
							}},
						}},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podReplacementPolicy: Unsupported value: "TerminatingOrFailed": supported values: "Failed"`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector:             validGeneratedSelector,
					PodReplacementPolicy: &terminatingOrFailedPodReplacement,
					Template:             validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
								Type:   api.DisruptionTarget,
								Status: api.ConditionTrue,
							}},
						},
						},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.podReplacementPolicy: Unsupported value: "": supported values: "Failed", "TerminatingOrFailed"`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					PodReplacementPolicy: (*batch.PodReplacementPolicy)(pointer.String("")),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGeneratedRestartPolicyNever,
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		`spec.template.spec.restartPolicy: Invalid value: "OnFailure": only "Never" is supported when podFailurePolicy is specified`: {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validGeneratedSelector.MatchLabels,
						},
						Spec: podtest.MakePod("",
							podtest.SetRestartPolicy(api.RestartPolicyOnFailure),
							podtest.SetContainers(podtest.MakeContainer("abc")),
						).Spec,
					},
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{},
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.parallelism:must be greater than or equal to 0": {
			job: batch.Job{
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
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.backoffLimit:must be greater than or equal to 0": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					BackoffLimit: pointer.Int32(-1),
					Selector:     validGeneratedSelector,
					Template:     validPodTemplateSpecForGenerated,
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.backoffLimitPerIndex: Invalid value: 1: requires indexed completion mode": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					BackoffLimitPerIndex: pointer.Int32(1),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.backoffLimitPerIndex:must be greater than or equal to 0": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					BackoffLimitPerIndex: pointer.Int32(-1),
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.maxFailedIndexes: Invalid value: 11: must be less than or equal to completions": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Completions:          pointer.Int32(10),
					MaxFailedIndexes:     pointer.Int32(11),
					BackoffLimitPerIndex: pointer.Int32(1),
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.maxFailedIndexes: Required value: must be specified when completions is above 100000": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Completions:          pointer.Int32(100_001),
					BackoffLimitPerIndex: pointer.Int32(1),
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.parallelism: Invalid value: 50000: must be less than or equal to 10000 when completions are above 100000 and used with backoff limit per index": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Completions:          pointer.Int32(100_001),
					Parallelism:          pointer.Int32(50_000),
					BackoffLimitPerIndex: pointer.Int32(1),
					MaxFailedIndexes:     pointer.Int32(1),
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.maxFailedIndexes: Invalid value: 100001: must be less than or equal to 100000": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Completions:          pointer.Int32(100_001),
					BackoffLimitPerIndex: pointer.Int32(1),
					MaxFailedIndexes:     pointer.Int32(100_001),
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.maxFailedIndexes: Invalid value: 50000: must be less than or equal to 10000 when completions are above 100000 and used with backoff limit per index": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Completions:          pointer.Int32(100_001),
					BackoffLimitPerIndex: pointer.Int32(1),
					MaxFailedIndexes:     pointer.Int32(50_000),
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.maxFailedIndexes:must be greater than or equal to 0": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					BackoffLimitPerIndex: pointer.Int32(1),
					MaxFailedIndexes:     pointer.Int32(-1),
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGenerated,
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.backoffLimitPerIndex: Required value: when maxFailedIndexes is specified": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					MaxFailedIndexes: pointer.Int32(1),
					CompletionMode:   completionModePtr(batch.IndexedCompletion),
					Selector:         validGeneratedSelector,
					Template:         validPodTemplateSpecForGenerated,
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.completions:must be greater than or equal to 0": {
			job: batch.Job{
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
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.activeDeadlineSeconds:must be greater than or equal to 0": {
			job: batch.Job{
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
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.selector:Required value": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Template: validPodTemplateSpecForGenerated,
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.template.metadata.labels: Invalid value: map[string]string{\"y\":\"z\"}: `selector` does not match template `labels`": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector:       validManualSelector,
					ManualSelector: pointer.Bool(true),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"y": "z"},
						},
						Spec: podtest.MakePod("",
							podtest.SetRestartPolicy(api.RestartPolicyOnFailure),
							podtest.SetContainers(podtest.MakeContainer("abc")),
						).Spec,
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.template.metadata.labels: Invalid value: map[string]string{\"controller-uid\":\"4d5e6f\"}: `selector` does not match template `labels`": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector:       validManualSelector,
					ManualSelector: pointer.Bool(true),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"controller-uid": "4d5e6f"},
						},
						Spec: podtest.MakePod("",
							podtest.SetRestartPolicy(api.RestartPolicyOnFailure),
							podtest.SetContainers(podtest.MakeContainer("abc")),
						).Spec,
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.template.spec.restartPolicy: Required value": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector:       validManualSelector,
					ManualSelector: pointer.Bool(true),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validManualSelector.MatchLabels,
						},
						Spec: podtest.MakePod("",
							podtest.SetRestartPolicy(api.RestartPolicyAlways),
							podtest.SetContainers(podtest.MakeContainer("abc")),
						).Spec,
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.template.spec.restartPolicy: Unsupported value": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector:       validManualSelector,
					ManualSelector: pointer.Bool(true),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validManualSelector.MatchLabels,
						},
						Spec: podtest.MakePod("",
							podtest.SetRestartPolicy("Invalid"),
							podtest.SetContainers(podtest.MakeContainer("abc")),
						).Spec,
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.ttlSecondsAfterFinished: must be greater than or equal to 0": {
			job: batch.Job{
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
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.completions: Required value: when completion mode is Indexed": {
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
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.parallelism: must be less than or equal to 100000 when completion mode is Indexed": {
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
					Completions:    pointer.Int32(2),
					Parallelism:    pointer.Int32(100001),
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.template.metadata.labels[controller-uid]: Required value: must be '1a2b3c'": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{batch.LegacyControllerUidLabel: "1a2b3c"},
					},
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{batch.LegacyJobNameLabel: "myjob"},
						},
						Spec: podtest.MakePod("",
							podtest.SetRestartPolicy(api.RestartPolicyOnFailure),
							podtest.SetContainers(podtest.MakeContainer("abc")),
							podtest.SetInitContainers(podtest.MakeContainer("def")),
						).Spec,
					},
				},
			},
			opts: JobValidationOptions{},
		},
		"metadata.uid: Required value": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
				},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{batch.LegacyControllerUidLabel: "test"},
					},
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{batch.LegacyJobNameLabel: "myjob"},
						},
						Spec: podtest.MakePod("",
							podtest.SetRestartPolicy(api.RestartPolicyOnFailure),
							podtest.SetContainers(podtest.MakeContainer("abc")),
							podtest.SetInitContainers(podtest.MakeContainer("def")),
						).Spec,
					},
				},
			},
			opts: JobValidationOptions{},
		},
		"spec.selector: Invalid value: v1.LabelSelector{MatchLabels:map[string]string{\"a\":\"b\"}, MatchExpressions:[]v1.LabelSelectorRequirement(nil)}: `selector` not auto-generated": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					Template: validPodTemplateSpecForGenerated,
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
		"spec.template.metadata.labels[batch.kubernetes.io/controller-uid]: Required value: must be '1a2b3c'": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: metav1.NamespaceDefault,
					UID:       types.UID("1a2b3c"),
				},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{batch.ControllerUidLabel: "1a2b3c"},
					},
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{batch.JobNameLabel: "myjob", batch.LegacyControllerUidLabel: "1a2b3c", batch.LegacyJobNameLabel: "myjob"},
						},
						Spec: podtest.MakePod("",
							podtest.SetRestartPolicy(api.RestartPolicyOnFailure),
							podtest.SetContainers(podtest.MakeContainer("abc")),
							podtest.SetInitContainers(podtest.MakeContainer("def")),
						).Spec,
					},
				},
			},
			opts: JobValidationOptions{RequirePrefixedLabels: true},
		},
	}

	for k, v := range errorCases {
		t.Run(k, func(t *testing.T) {
			errs := ValidateJob(&v.job, v.opts)
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
	validPodTemplateSpecForGeneratedRestartPolicyNever := getValidPodTemplateSpecForGenerated(validGeneratedSelector)
	validPodTemplateSpecForGeneratedRestartPolicyNever.Spec.RestartPolicy = api.RestartPolicyNever

	validNodeAffinity := &api.Affinity{
		NodeAffinity: &api.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
				NodeSelectorTerms: []api.NodeSelectorTerm{{
					MatchExpressions: []api.NodeSelectorRequirement{{
						Key:      "foo",
						Operator: api.NodeSelectorOpIn,
						Values:   []string{"bar", "value2"},
					}},
				}},
			},
		},
	}
	validPodTemplateWithAffinity := getValidPodTemplateSpecForGenerated(validGeneratedSelector)
	validPodTemplateWithAffinity.Spec.Affinity = &api.Affinity{
		NodeAffinity: &api.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
				NodeSelectorTerms: []api.NodeSelectorTerm{{
					MatchExpressions: []api.NodeSelectorRequirement{{
						Key:      "foo",
						Operator: api.NodeSelectorOpIn,
						Values:   []string{"bar", "value"},
					}},
				}},
			},
		},
	}
	// This is to test immutability of the selector, both the new and old
	// selector should match the labels in the template, which is immutable
	// on its own; therfore, the only way to test selector immutability is
	// when the new selector is changed but still matches the existing labels.
	newSelector := getValidGeneratedSelector()
	newSelector.MatchLabels["foo"] = "bar"
	validTolerations := []api.Toleration{{
		Key:      "foo",
		Operator: api.TolerationOpEqual,
		Value:    "bar",
		Effect:   api.TaintEffectPreferNoSchedule,
	}}
	cases := map[string]struct {
		old    batch.Job
		update func(*batch.Job)
		opts   JobValidationOptions
		err    *field.Error
	}{
		"mutable fields": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector:                validGeneratedSelector,
					Template:                validPodTemplateSpecForGenerated,
					Parallelism:             pointer.Int32(5),
					ActiveDeadlineSeconds:   pointer.Int64(2),
					TTLSecondsAfterFinished: pointer.Int32(1),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Parallelism = pointer.Int32(2)
				job.Spec.ActiveDeadlineSeconds = pointer.Int64(3)
				job.Spec.TTLSecondsAfterFinished = pointer.Int32(2)
				job.Spec.ManualSelector = pointer.Bool(true)
			},
		},
		"invalid attempt to set managedBy field": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.ManagedBy = ptr.To("example.com/custom-controller")
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.managedBy",
			},
		},
		"invalid update of the managedBy field": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector:  validGeneratedSelector,
					Template:  validPodTemplateSpecForGenerated,
					ManagedBy: ptr.To("example.com/custom-controller1"),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.ManagedBy = ptr.To("example.com/custom-controller2")
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.managedBy",
			},
		},
		"immutable completions for non-indexed jobs": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Completions = pointer.Int32(1)
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.completions",
			},
		},
		"immutable completions for indexed job when AllowElasticIndexedJobs is false": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Completions = pointer.Int32(1)
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
					Template: getValidPodTemplateSpecForGenerated(newSelector),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Selector = newSelector
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.selector",
			},
		},
		"add success policy": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.SuccessPolicy = &batch.SuccessPolicy{
					Rules: []batch.SuccessPolicyRule{{
						SucceededCount: ptr.To[int32](2),
					}},
				}
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.successPolicy",
			},
		},
		"update success policy": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: []batch.SuccessPolicyRule{{
							SucceededIndexes: ptr.To("1-3"),
						}},
					},
				},
			},
			update: func(job *batch.Job) {
				job.Spec.SuccessPolicy.Rules = append(job.Spec.SuccessPolicy.Rules, batch.SuccessPolicyRule{
					SucceededCount: ptr.To[int32](3),
				})
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.successPolicy",
			},
		},
		"remove success policy": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](5),
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					SuccessPolicy: &batch.SuccessPolicy{
						Rules: []batch.SuccessPolicyRule{{
							SucceededIndexes: ptr.To("1-3"),
						}},
					},
				},
			},
			update: func(job *batch.Job) {
				job.Spec.SuccessPolicy = nil
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.successPolicy",
			},
		},
		"add pod failure policy": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.PodFailurePolicy = &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{{
						Action: batch.PodFailurePolicyActionIgnore,
						OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
							Type:   api.DisruptionTarget,
							Status: api.ConditionTrue,
						}},
					}},
				}
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.podFailurePolicy",
			},
		},
		"update pod failure policy": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
								Type:   api.DisruptionTarget,
								Status: api.ConditionTrue,
							}},
						}},
					},
				},
			},
			update: func(job *batch.Job) {
				job.Spec.PodFailurePolicy.Rules = append(job.Spec.PodFailurePolicy.Rules, batch.PodFailurePolicyRule{
					Action: batch.PodFailurePolicyActionCount,
					OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
						Type:   api.DisruptionTarget,
						Status: api.ConditionTrue,
					}},
				})
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.podFailurePolicy",
			},
		},
		"remove pod failure policy": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{{
								Type:   api.DisruptionTarget,
								Status: api.ConditionTrue,
							}},
						}},
					},
				},
			},
			update: func(job *batch.Job) {
				job.Spec.PodFailurePolicy = nil
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.podFailurePolicy",
			},
		},
		"set backoff limit per index": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGeneratedRestartPolicyNever,
					Completions:    pointer.Int32(3),
					CompletionMode: completionModePtr(batch.IndexedCompletion),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.BackoffLimitPerIndex = pointer.Int32(1)
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.backoffLimitPerIndex",
			},
		},
		"unset backoff limit per index": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGeneratedRestartPolicyNever,
					Completions:          pointer.Int32(3),
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					BackoffLimitPerIndex: pointer.Int32(1),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.BackoffLimitPerIndex = nil
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.backoffLimitPerIndex",
			},
		},
		"update backoff limit per index": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGeneratedRestartPolicyNever,
					Completions:          pointer.Int32(3),
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					BackoffLimitPerIndex: pointer.Int32(1),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.BackoffLimitPerIndex = pointer.Int32(2)
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.backoffLimitPerIndex",
			},
		},
		"set max failed indexes": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGeneratedRestartPolicyNever,
					Completions:          pointer.Int32(3),
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					BackoffLimitPerIndex: pointer.Int32(1),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.MaxFailedIndexes = pointer.Int32(1)
			},
		},
		"unset max failed indexes": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGeneratedRestartPolicyNever,
					Completions:          pointer.Int32(3),
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					BackoffLimitPerIndex: pointer.Int32(1),
					MaxFailedIndexes:     pointer.Int32(1),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.MaxFailedIndexes = nil
			},
		},
		"update max failed indexes": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector:             validGeneratedSelector,
					Template:             validPodTemplateSpecForGeneratedRestartPolicyNever,
					Completions:          pointer.Int32(3),
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					BackoffLimitPerIndex: pointer.Int32(1),
					MaxFailedIndexes:     pointer.Int32(1),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.MaxFailedIndexes = pointer.Int32(2)
			},
		},
		"immutable pod template": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGenerated,
					Completions:    pointer.Int32(3),
					CompletionMode: completionModePtr(batch.IndexedCompletion),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.DNSPolicy = api.DNSClusterFirstWithHostNet
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
					Completions:    pointer.Int32(2),
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
		"immutable completions for non-indexed job when AllowElasticIndexedJobs is true": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGenerated,
					CompletionMode: completionModePtr(batch.NonIndexedCompletion),
					Completions:    pointer.Int32(2),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Completions = pointer.Int32(4)
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.completions",
			},
			opts: JobValidationOptions{AllowElasticIndexedJobs: true},
		},

		"immutable node affinity": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.Affinity = validNodeAffinity
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.template",
			},
		},
		"add node affinity": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.Affinity = validNodeAffinity
			},
			opts: JobValidationOptions{
				AllowMutableSchedulingDirectives: true,
			},
		},
		"update node affinity": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateWithAffinity,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.Affinity = validNodeAffinity
			},
			opts: JobValidationOptions{
				AllowMutableSchedulingDirectives: true,
			},
		},
		"remove node affinity": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateWithAffinity,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.Affinity.NodeAffinity = nil
			},
			opts: JobValidationOptions{
				AllowMutableSchedulingDirectives: true,
			},
		},
		"remove affinity": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateWithAffinity,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.Affinity = nil
			},
			opts: JobValidationOptions{
				AllowMutableSchedulingDirectives: true,
			},
		},
		"immutable tolerations": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.Tolerations = validTolerations
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.template",
			},
		},
		"mutable tolerations": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.Tolerations = validTolerations
			},
			opts: JobValidationOptions{
				AllowMutableSchedulingDirectives: true,
			},
		},
		"immutable node selector": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.NodeSelector = map[string]string{"foo": "bar"}
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.template",
			},
		},
		"mutable node selector": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.NodeSelector = map[string]string{"foo": "bar"}
			},
			opts: JobValidationOptions{
				AllowMutableSchedulingDirectives: true,
			},
		},
		"immutable annotations": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Annotations = map[string]string{"foo": "baz"}
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.template",
			},
		},
		"mutable annotations": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Annotations = map[string]string{"foo": "baz"}
			},
			opts: JobValidationOptions{
				AllowMutableSchedulingDirectives: true,
			},
		},
		"immutable labels": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
			update: func(job *batch.Job) {
				newLabels := getValidGeneratedSelector().MatchLabels
				newLabels["bar"] = "baz"
				job.Spec.Template.Labels = newLabels
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.template",
			},
		},
		"mutable labels": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
			update: func(job *batch.Job) {
				newLabels := getValidGeneratedSelector().MatchLabels
				newLabels["bar"] = "baz"
				job.Spec.Template.Labels = newLabels
			},
			opts: JobValidationOptions{
				AllowMutableSchedulingDirectives: true,
			},
		},
		"immutable schedulingGates": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.SchedulingGates = append(job.Spec.Template.Spec.SchedulingGates, api.PodSchedulingGate{Name: "gate"})
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.template",
			},
		},
		"mutable schedulingGates": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.SchedulingGates = append(job.Spec.Template.Spec.SchedulingGates, api.PodSchedulingGate{Name: "gate"})
			},
			opts: JobValidationOptions{
				AllowMutableSchedulingDirectives: true,
			},
		},
		"update completions and parallelism to same value is valid": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGenerated,
					Completions:    pointer.Int32(1),
					Parallelism:    pointer.Int32(1),
					CompletionMode: completionModePtr(batch.IndexedCompletion),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Completions = pointer.Int32(2)
				job.Spec.Parallelism = pointer.Int32(2)
			},
			opts: JobValidationOptions{
				AllowElasticIndexedJobs: true,
			},
		},
		"previous parallelism != previous completions, new parallelism == new completions": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGenerated,
					Completions:    pointer.Int32(1),
					Parallelism:    pointer.Int32(2),
					CompletionMode: completionModePtr(batch.IndexedCompletion),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Completions = pointer.Int32(3)
				job.Spec.Parallelism = pointer.Int32(3)
			},
			opts: JobValidationOptions{
				AllowElasticIndexedJobs: true,
			},
		},
		"indexed job updating completions and parallelism to different values is invalid": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGenerated,
					Completions:    pointer.Int32(1),
					Parallelism:    pointer.Int32(1),
					CompletionMode: completionModePtr(batch.IndexedCompletion),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Completions = pointer.Int32(2)
				job.Spec.Parallelism = pointer.Int32(3)
			},
			opts: JobValidationOptions{
				AllowElasticIndexedJobs: true,
			},
			err: &field.Error{
				Type:  field.ErrorTypeInvalid,
				Field: "spec.completions",
			},
		},
		"indexed job with completions set updated to nil does not panic": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGenerated,
					Completions:    pointer.Int32(1),
					Parallelism:    pointer.Int32(1),
					CompletionMode: completionModePtr(batch.IndexedCompletion),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Completions = nil
				job.Spec.Parallelism = pointer.Int32(3)
			},
			opts: JobValidationOptions{
				AllowElasticIndexedJobs: true,
			},
			err: &field.Error{
				Type:  field.ErrorTypeRequired,
				Field: "spec.completions",
			},
		},
		"indexed job with completions unchanged, parallelism reduced to less than completions": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGenerated,
					Completions:    pointer.Int32(2),
					Parallelism:    pointer.Int32(2),
					CompletionMode: completionModePtr(batch.IndexedCompletion),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Completions = pointer.Int32(2)
				job.Spec.Parallelism = pointer.Int32(1)
			},
			opts: JobValidationOptions{
				AllowElasticIndexedJobs: true,
			},
		},
		"indexed job with completions unchanged, parallelism increased higher than completions": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector:       validGeneratedSelector,
					Template:       validPodTemplateSpecForGenerated,
					Completions:    pointer.Int32(2),
					Parallelism:    pointer.Int32(2),
					CompletionMode: completionModePtr(batch.IndexedCompletion),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Completions = pointer.Int32(2)
				job.Spec.Parallelism = pointer.Int32(3)
			},
			opts: JobValidationOptions{
				AllowElasticIndexedJobs: true,
			},
		},
	}
	ignoreValueAndDetail := cmpopts.IgnoreFields(field.Error{}, "BadValue", "Detail")
	for k, tc := range cases {
		t.Run(k, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			update := tc.old.DeepCopy()
			tc.update(update)
			errs := ValidateJobUpdate(update, &tc.old, tc.opts)
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
		opts JobStatusValidationOptions

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
					Active:      1,
					Succeeded:   2,
					Failed:      3,
					Terminating: pointer.Int32(4),
				},
			},
			update: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "abc",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "1",
				},
				Status: batch.JobStatus{
					Active:      2,
					Succeeded:   3,
					Failed:      4,
					Ready:       pointer.Int32(1),
					Terminating: pointer.Int32(4),
				},
			},
		},
		"nil ready and terminating": {
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
					Active:    2,
					Succeeded: 3,
					Failed:    4,
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
					Active:      1,
					Succeeded:   2,
					Failed:      3,
					Terminating: pointer.Int32(4),
				},
			},
			update: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "abc",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "10",
				},
				Status: batch.JobStatus{
					Active:      -1,
					Succeeded:   -2,
					Failed:      -3,
					Ready:       pointer.Int32(-1),
					Terminating: pointer.Int32(-2),
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.active"},
				{Type: field.ErrorTypeInvalid, Field: "status.succeeded"},
				{Type: field.ErrorTypeInvalid, Field: "status.failed"},
				{Type: field.ErrorTypeInvalid, Field: "status.ready"},
				{Type: field.ErrorTypeInvalid, Field: "status.terminating"},
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
			errs := ValidateJobUpdateStatus(&tc.update, &tc.old, tc.opts)
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
	validHostNetPodTemplateSpec := func() api.PodTemplateSpec {
		spec := getValidPodTemplateSpecForGenerated(getValidGeneratedSelector())
		spec.Spec.SecurityContext = &api.PodSecurityContext{
			HostNetwork: true,
		}
		spec.Spec.Containers[0].Ports = []api.ContainerPort{{
			ContainerPort: 12345,
			Protocol:      api.ProtocolTCP,
		}}
		return spec
	}()

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
		"hostnet job": {
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
						Template: validHostNetPodTemplateSpec,
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
		"correct timeZone value": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          &timeZoneCorrect,
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
		t.Run(k, func(t *testing.T) {
			if errs := ValidateCronJobCreate(&v, corevalidation.PodValidationOptions{}); len(errs) != 0 {
				t.Errorf("expected success for %s: %v", k, errs)
			}

			// Update validation should pass same success cases
			// copy to avoid polluting the testcase object, set a resourceVersion to allow validating update, and test a no-op update
			v = *v.DeepCopy()
			v.ResourceVersion = "1"
			if errs := ValidateCronJobUpdate(&v, &v, corevalidation.PodValidationOptions{}); len(errs) != 0 {
				t.Errorf("expected success for %s: %v", k, errs)
			}
		})
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
		"spec.timeZone: timeZone must be nil or non-empty string": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          &timeZoneEmpty,
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"spec.timeZone: timeZone must be an explicit time zone as defined in https://www.iana.org/time-zones": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          &timeZoneLocal,
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"spec.timeZone: Invalid value: \" Continent/Zone\": unknown time zone  Continent/Zone": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          &timeZoneBadPrefix,
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"spec.timeZone: Invalid value: \"Continent/InvalidZone\": unknown time zone  Continent/InvalidZone": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          &timeZoneBadName,
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"spec.timeZone: Invalid value: \" \": unknown time zone  ": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          &timeZoneEmptySpace,
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"spec.timeZone: Invalid value: \"Continent/Zone \": unknown time zone Continent/Zone ": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          &timeZoneBadSuffix,
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
						ManualSelector: pointer.Bool(true),
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
							Spec: podtest.MakePod("",
								podtest.SetRestartPolicy(api.RestartPolicyAlways),
								podtest.SetContainers(podtest.MakeContainer("abc")),
							).Spec,
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
							Spec: podtest.MakePod("",
								podtest.SetRestartPolicy("Invalid"),
								podtest.SetContainers(podtest.MakeContainer("abc")),
							).Spec,
						},
					},
				},
			},
		},
		"spec.jobTemplate.spec.ttlSecondsAfterFinished:must be greater than or equal to 0": {
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
		},
	}

	for k, v := range errorCases {
		t.Run(k, func(t *testing.T) {
			errs := ValidateCronJobCreate(&v, corevalidation.PodValidationOptions{})
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
			oldSpec := *v.DeepCopy()
			oldSpec.ResourceVersion = "1"
			oldSpec.Spec.TimeZone = nil

			newSpec := *v.DeepCopy()
			newSpec.ResourceVersion = "2"

			errs = ValidateCronJobUpdate(&newSpec, &oldSpec, corevalidation.PodValidationOptions{})
			if len(errs) == 0 {
				if k == "metadata.name: must be no more than 52 characters" {
					return
				}
				t.Errorf("expected failure for %s", k)
			} else {
				s := strings.Split(k, ":")
				err := errs[0]
				if err.Field != s[0] || !strings.Contains(err.Error(), s[1]) {
					t.Errorf("unexpected error: %v, expected: %s", err, k)
				}
			}
		})
	}
}

func TestValidateCronJobScheduleTZ(t *testing.T) {
	validPodTemplateSpec := getValidPodTemplateSpecForGenerated(getValidGeneratedSelector())
	validPodTemplateSpec.Labels = map[string]string{}
	validSchedule := "0 * * * *"
	invalidSchedule := "TZ=UTC 0 * * * *"
	invalidCronJob := &batch.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mycronjob",
			Namespace: metav1.NamespaceDefault,
			UID:       types.UID("1a2b3c"),
		},
		Spec: batch.CronJobSpec{
			Schedule:          invalidSchedule,
			ConcurrencyPolicy: batch.AllowConcurrent,
			JobTemplate: batch.JobTemplateSpec{
				Spec: batch.JobSpec{
					Template: validPodTemplateSpec,
				},
			},
		},
	}
	validCronJob := &batch.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mycronjob",
			Namespace: metav1.NamespaceDefault,
			UID:       types.UID("1a2b3c"),
		},
		Spec: batch.CronJobSpec{
			Schedule:          validSchedule,
			ConcurrencyPolicy: batch.AllowConcurrent,
			JobTemplate: batch.JobTemplateSpec{
				Spec: batch.JobSpec{
					Template: validPodTemplateSpec,
				},
			},
		},
	}

	testCases := map[string]struct {
		cronJob   *batch.CronJob
		createErr string
		update    func(*batch.CronJob)
		updateErr string
	}{
		"update removing TZ should work": {
			cronJob:   invalidCronJob,
			createErr: "cannot use TZ or CRON_TZ in schedule",
			update: func(cj *batch.CronJob) {
				cj.Spec.Schedule = validSchedule
			},
		},
		"update not modifying TZ should work": {
			cronJob:   invalidCronJob,
			createErr: "cannot use TZ or CRON_TZ in schedule, use timeZone field instead",
			update: func(cj *batch.CronJob) {
				cj.Spec.Schedule = invalidSchedule
			},
		},
		"update not modifying TZ but adding .spec.timeZone should fail": {
			cronJob:   invalidCronJob,
			createErr: "cannot use TZ or CRON_TZ in schedule, use timeZone field instead",
			update: func(cj *batch.CronJob) {
				cj.Spec.TimeZone = &timeZoneUTC
			},
			updateErr: "cannot use both timeZone field and TZ or CRON_TZ in schedule",
		},
		"update adding TZ should fail": {
			cronJob: validCronJob,
			update: func(cj *batch.CronJob) {
				cj.Spec.Schedule = invalidSchedule
			},
			updateErr: "cannot use TZ or CRON_TZ in schedule",
		},
	}

	for k, v := range testCases {
		t.Run(k, func(t *testing.T) {
			errs := ValidateCronJobCreate(v.cronJob, corevalidation.PodValidationOptions{})
			if len(errs) > 0 {
				err := errs[0]
				if len(v.createErr) == 0 {
					t.Errorf("unexpected error: %#v, none expected", err)
					return
				}
				if !strings.Contains(err.Error(), v.createErr) {
					t.Errorf("unexpected error: %v, expected: %s", err, v.createErr)
				}
			} else if len(v.createErr) != 0 {
				t.Errorf("no error, expected %v", v.createErr)
				return
			}

			oldSpec := v.cronJob.DeepCopy()
			oldSpec.ResourceVersion = "1"

			newSpec := v.cronJob.DeepCopy()
			newSpec.ResourceVersion = "2"
			if v.update != nil {
				v.update(newSpec)
			}

			errs = ValidateCronJobUpdate(newSpec, oldSpec, corevalidation.PodValidationOptions{})
			if len(errs) > 0 {
				err := errs[0]
				if len(v.updateErr) == 0 {
					t.Errorf("unexpected error: %#v, none expected", err)
					return
				}
				if !strings.Contains(err.Error(), v.updateErr) {
					t.Errorf("unexpected error: %v, expected: %s", err, v.updateErr)
				}
			} else if len(v.updateErr) != 0 {
				t.Errorf("no error, expected %v", v.updateErr)
				return
			}
		})
	}
}

func TestValidateCronJobSpec(t *testing.T) {
	validPodTemplateSpec := getValidPodTemplateSpecForGenerated(getValidGeneratedSelector())
	validPodTemplateSpec.Labels = map[string]string{}

	type testCase struct {
		old       *batch.CronJobSpec
		new       *batch.CronJobSpec
		expectErr bool
	}

	cases := map[string]testCase{
		"no validation because timeZone is nil for old and new": {
			old: &batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          nil,
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
			new: &batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          nil,
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"check validation because timeZone is different for new": {
			old: &batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          nil,
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
			new: &batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          pointer.String("America/New_York"),
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"check validation because timeZone is different for new and invalid": {
			old: &batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          nil,
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
			new: &batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          pointer.String("broken"),
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
			expectErr: true,
		},
		"old timeZone and new timeZone are valid": {
			old: &batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          pointer.String("America/New_York"),
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
			new: &batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          pointer.String("America/Chicago"),
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"old timeZone is valid, but new timeZone is invalid": {
			old: &batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          pointer.String("America/New_York"),
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
			new: &batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          pointer.String("broken"),
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
			expectErr: true,
		},
		"old timeZone and new timeZone are invalid, but unchanged": {
			old: &batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          pointer.String("broken"),
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
			new: &batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          pointer.String("broken"),
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
		"old timeZone and new timeZone are invalid, but different": {
			old: &batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          pointer.String("broken"),
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
			new: &batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          pointer.String("still broken"),
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
			expectErr: true,
		},
		"old timeZone is invalid, but new timeZone is valid": {
			old: &batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          pointer.String("broken"),
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
			new: &batch.CronJobSpec{
				Schedule:          "0 * * * *",
				TimeZone:          pointer.String("America/New_York"),
				ConcurrencyPolicy: batch.AllowConcurrent,
				JobTemplate: batch.JobTemplateSpec{
					Spec: batch.JobSpec{
						Template: validPodTemplateSpec,
					},
				},
			},
		},
	}

	for k, v := range cases {
		errs := validateCronJobSpec(v.new, v.old, field.NewPath("spec"), corevalidation.PodValidationOptions{})
		if len(errs) > 0 && !v.expectErr {
			t.Errorf("unexpected error for %s: %v", k, errs)
		} else if len(errs) == 0 && v.expectErr {
			t.Errorf("expected error for %s but got nil", k)
		}
	}
}

func completionModePtr(m batch.CompletionMode) *batch.CompletionMode {
	return &m
}

func TestTimeZones(t *testing.T) {
	// all valid time zones as of go1.19 release on 2022-08-02
	data := []string{
		`Africa/Abidjan`,
		`Africa/Accra`,
		`Africa/Addis_Ababa`,
		`Africa/Algiers`,
		`Africa/Asmara`,
		`Africa/Asmera`,
		`Africa/Bamako`,
		`Africa/Bangui`,
		`Africa/Banjul`,
		`Africa/Bissau`,
		`Africa/Blantyre`,
		`Africa/Brazzaville`,
		`Africa/Bujumbura`,
		`Africa/Cairo`,
		`Africa/Casablanca`,
		`Africa/Ceuta`,
		`Africa/Conakry`,
		`Africa/Dakar`,
		`Africa/Dar_es_Salaam`,
		`Africa/Djibouti`,
		`Africa/Douala`,
		`Africa/El_Aaiun`,
		`Africa/Freetown`,
		`Africa/Gaborone`,
		`Africa/Harare`,
		`Africa/Johannesburg`,
		`Africa/Juba`,
		`Africa/Kampala`,
		`Africa/Khartoum`,
		`Africa/Kigali`,
		`Africa/Kinshasa`,
		`Africa/Lagos`,
		`Africa/Libreville`,
		`Africa/Lome`,
		`Africa/Luanda`,
		`Africa/Lubumbashi`,
		`Africa/Lusaka`,
		`Africa/Malabo`,
		`Africa/Maputo`,
		`Africa/Maseru`,
		`Africa/Mbabane`,
		`Africa/Mogadishu`,
		`Africa/Monrovia`,
		`Africa/Nairobi`,
		`Africa/Ndjamena`,
		`Africa/Niamey`,
		`Africa/Nouakchott`,
		`Africa/Ouagadougou`,
		`Africa/Porto-Novo`,
		`Africa/Sao_Tome`,
		`Africa/Timbuktu`,
		`Africa/Tripoli`,
		`Africa/Tunis`,
		`Africa/Windhoek`,
		`America/Adak`,
		`America/Anchorage`,
		`America/Anguilla`,
		`America/Antigua`,
		`America/Araguaina`,
		`America/Argentina/Buenos_Aires`,
		`America/Argentina/Catamarca`,
		`America/Argentina/ComodRivadavia`,
		`America/Argentina/Cordoba`,
		`America/Argentina/Jujuy`,
		`America/Argentina/La_Rioja`,
		`America/Argentina/Mendoza`,
		`America/Argentina/Rio_Gallegos`,
		`America/Argentina/Salta`,
		`America/Argentina/San_Juan`,
		`America/Argentina/San_Luis`,
		`America/Argentina/Tucuman`,
		`America/Argentina/Ushuaia`,
		`America/Aruba`,
		`America/Asuncion`,
		`America/Atikokan`,
		`America/Atka`,
		`America/Bahia`,
		`America/Bahia_Banderas`,
		`America/Barbados`,
		`America/Belem`,
		`America/Belize`,
		`America/Blanc-Sablon`,
		`America/Boa_Vista`,
		`America/Bogota`,
		`America/Boise`,
		`America/Buenos_Aires`,
		`America/Cambridge_Bay`,
		`America/Campo_Grande`,
		`America/Cancun`,
		`America/Caracas`,
		`America/Catamarca`,
		`America/Cayenne`,
		`America/Cayman`,
		`America/Chicago`,
		`America/Chihuahua`,
		`America/Coral_Harbour`,
		`America/Cordoba`,
		`America/Costa_Rica`,
		`America/Creston`,
		`America/Cuiaba`,
		`America/Curacao`,
		`America/Danmarkshavn`,
		`America/Dawson`,
		`America/Dawson_Creek`,
		`America/Denver`,
		`America/Detroit`,
		`America/Dominica`,
		`America/Edmonton`,
		`America/Eirunepe`,
		`America/El_Salvador`,
		`America/Ensenada`,
		`America/Fort_Nelson`,
		`America/Fort_Wayne`,
		`America/Fortaleza`,
		`America/Glace_Bay`,
		`America/Godthab`,
		`America/Goose_Bay`,
		`America/Grand_Turk`,
		`America/Grenada`,
		`America/Guadeloupe`,
		`America/Guatemala`,
		`America/Guayaquil`,
		`America/Guyana`,
		`America/Halifax`,
		`America/Havana`,
		`America/Hermosillo`,
		`America/Indiana/Indianapolis`,
		`America/Indiana/Knox`,
		`America/Indiana/Marengo`,
		`America/Indiana/Petersburg`,
		`America/Indiana/Tell_City`,
		`America/Indiana/Vevay`,
		`America/Indiana/Vincennes`,
		`America/Indiana/Winamac`,
		`America/Indianapolis`,
		`America/Inuvik`,
		`America/Iqaluit`,
		`America/Jamaica`,
		`America/Jujuy`,
		`America/Juneau`,
		`America/Kentucky/Louisville`,
		`America/Kentucky/Monticello`,
		`America/Knox_IN`,
		`America/Kralendijk`,
		`America/La_Paz`,
		`America/Lima`,
		`America/Los_Angeles`,
		`America/Louisville`,
		`America/Lower_Princes`,
		`America/Maceio`,
		`America/Managua`,
		`America/Manaus`,
		`America/Marigot`,
		`America/Martinique`,
		`America/Matamoros`,
		`America/Mazatlan`,
		`America/Mendoza`,
		`America/Menominee`,
		`America/Merida`,
		`America/Metlakatla`,
		`America/Mexico_City`,
		`America/Miquelon`,
		`America/Moncton`,
		`America/Monterrey`,
		`America/Montevideo`,
		`America/Montreal`,
		`America/Montserrat`,
		`America/Nassau`,
		`America/New_York`,
		`America/Nipigon`,
		`America/Nome`,
		`America/Noronha`,
		`America/North_Dakota/Beulah`,
		`America/North_Dakota/Center`,
		`America/North_Dakota/New_Salem`,
		`America/Nuuk`,
		`America/Ojinaga`,
		`America/Panama`,
		`America/Pangnirtung`,
		`America/Paramaribo`,
		`America/Phoenix`,
		`America/Port-au-Prince`,
		`America/Port_of_Spain`,
		`America/Porto_Acre`,
		`America/Porto_Velho`,
		`America/Puerto_Rico`,
		`America/Punta_Arenas`,
		`America/Rainy_River`,
		`America/Rankin_Inlet`,
		`America/Recife`,
		`America/Regina`,
		`America/Resolute`,
		`America/Rio_Branco`,
		`America/Rosario`,
		`America/Santa_Isabel`,
		`America/Santarem`,
		`America/Santiago`,
		`America/Santo_Domingo`,
		`America/Sao_Paulo`,
		`America/Scoresbysund`,
		`America/Shiprock`,
		`America/Sitka`,
		`America/St_Barthelemy`,
		`America/St_Johns`,
		`America/St_Kitts`,
		`America/St_Lucia`,
		`America/St_Thomas`,
		`America/St_Vincent`,
		`America/Swift_Current`,
		`America/Tegucigalpa`,
		`America/Thule`,
		`America/Thunder_Bay`,
		`America/Tijuana`,
		`America/Toronto`,
		`America/Tortola`,
		`America/Vancouver`,
		`America/Virgin`,
		`America/Whitehorse`,
		`America/Winnipeg`,
		`America/Yakutat`,
		`America/Yellowknife`,
		`Antarctica/Casey`,
		`Antarctica/Davis`,
		`Antarctica/DumontDUrville`,
		`Antarctica/Macquarie`,
		`Antarctica/Mawson`,
		`Antarctica/McMurdo`,
		`Antarctica/Palmer`,
		`Antarctica/Rothera`,
		`Antarctica/South_Pole`,
		`Antarctica/Syowa`,
		`Antarctica/Troll`,
		`Antarctica/Vostok`,
		`Arctic/Longyearbyen`,
		`Asia/Aden`,
		`Asia/Almaty`,
		`Asia/Amman`,
		`Asia/Anadyr`,
		`Asia/Aqtau`,
		`Asia/Aqtobe`,
		`Asia/Ashgabat`,
		`Asia/Ashkhabad`,
		`Asia/Atyrau`,
		`Asia/Baghdad`,
		`Asia/Bahrain`,
		`Asia/Baku`,
		`Asia/Bangkok`,
		`Asia/Barnaul`,
		`Asia/Beirut`,
		`Asia/Bishkek`,
		`Asia/Brunei`,
		`Asia/Calcutta`,
		`Asia/Chita`,
		`Asia/Choibalsan`,
		`Asia/Chongqing`,
		`Asia/Chungking`,
		`Asia/Colombo`,
		`Asia/Dacca`,
		`Asia/Damascus`,
		`Asia/Dhaka`,
		`Asia/Dili`,
		`Asia/Dubai`,
		`Asia/Dushanbe`,
		`Asia/Famagusta`,
		`Asia/Gaza`,
		`Asia/Harbin`,
		`Asia/Hebron`,
		`Asia/Ho_Chi_Minh`,
		`Asia/Hong_Kong`,
		`Asia/Hovd`,
		`Asia/Irkutsk`,
		`Asia/Istanbul`,
		`Asia/Jakarta`,
		`Asia/Jayapura`,
		`Asia/Jerusalem`,
		`Asia/Kabul`,
		`Asia/Kamchatka`,
		`Asia/Karachi`,
		`Asia/Kashgar`,
		`Asia/Kathmandu`,
		`Asia/Katmandu`,
		`Asia/Khandyga`,
		`Asia/Kolkata`,
		`Asia/Krasnoyarsk`,
		`Asia/Kuala_Lumpur`,
		`Asia/Kuching`,
		`Asia/Kuwait`,
		`Asia/Macao`,
		`Asia/Macau`,
		`Asia/Magadan`,
		`Asia/Makassar`,
		`Asia/Manila`,
		`Asia/Muscat`,
		`Asia/Nicosia`,
		`Asia/Novokuznetsk`,
		`Asia/Novosibirsk`,
		`Asia/Omsk`,
		`Asia/Oral`,
		`Asia/Phnom_Penh`,
		`Asia/Pontianak`,
		`Asia/Pyongyang`,
		`Asia/Qatar`,
		`Asia/Qostanay`,
		`Asia/Qyzylorda`,
		`Asia/Rangoon`,
		`Asia/Riyadh`,
		`Asia/Saigon`,
		`Asia/Sakhalin`,
		`Asia/Samarkand`,
		`Asia/Seoul`,
		`Asia/Shanghai`,
		`Asia/Singapore`,
		`Asia/Srednekolymsk`,
		`Asia/Taipei`,
		`Asia/Tashkent`,
		`Asia/Tbilisi`,
		`Asia/Tehran`,
		`Asia/Tel_Aviv`,
		`Asia/Thimbu`,
		`Asia/Thimphu`,
		`Asia/Tokyo`,
		`Asia/Tomsk`,
		`Asia/Ujung_Pandang`,
		`Asia/Ulaanbaatar`,
		`Asia/Ulan_Bator`,
		`Asia/Urumqi`,
		`Asia/Ust-Nera`,
		`Asia/Vientiane`,
		`Asia/Vladivostok`,
		`Asia/Yakutsk`,
		`Asia/Yangon`,
		`Asia/Yekaterinburg`,
		`Asia/Yerevan`,
		`Atlantic/Azores`,
		`Atlantic/Bermuda`,
		`Atlantic/Canary`,
		`Atlantic/Cape_Verde`,
		`Atlantic/Faeroe`,
		`Atlantic/Faroe`,
		`Atlantic/Jan_Mayen`,
		`Atlantic/Madeira`,
		`Atlantic/Reykjavik`,
		`Atlantic/South_Georgia`,
		`Atlantic/St_Helena`,
		`Atlantic/Stanley`,
		`Australia/ACT`,
		`Australia/Adelaide`,
		`Australia/Brisbane`,
		`Australia/Broken_Hill`,
		`Australia/Canberra`,
		`Australia/Currie`,
		`Australia/Darwin`,
		`Australia/Eucla`,
		`Australia/Hobart`,
		`Australia/LHI`,
		`Australia/Lindeman`,
		`Australia/Lord_Howe`,
		`Australia/Melbourne`,
		`Australia/North`,
		`Australia/NSW`,
		`Australia/Perth`,
		`Australia/Queensland`,
		`Australia/South`,
		`Australia/Sydney`,
		`Australia/Tasmania`,
		`Australia/Victoria`,
		`Australia/West`,
		`Australia/Yancowinna`,
		`Brazil/Acre`,
		`Brazil/DeNoronha`,
		`Brazil/East`,
		`Brazil/West`,
		`Canada/Atlantic`,
		`Canada/Central`,
		`Canada/Eastern`,
		`Canada/Mountain`,
		`Canada/Newfoundland`,
		`Canada/Pacific`,
		`Canada/Saskatchewan`,
		`Canada/Yukon`,
		`CET`,
		`Chile/Continental`,
		`Chile/EasterIsland`,
		`CST6CDT`,
		`Cuba`,
		`EET`,
		`Egypt`,
		`Eire`,
		`EST`,
		`EST5EDT`,
		`Etc/GMT`,
		`Etc/GMT+0`,
		`Etc/GMT+1`,
		`Etc/GMT+10`,
		`Etc/GMT+11`,
		`Etc/GMT+12`,
		`Etc/GMT+2`,
		`Etc/GMT+3`,
		`Etc/GMT+4`,
		`Etc/GMT+5`,
		`Etc/GMT+6`,
		`Etc/GMT+7`,
		`Etc/GMT+8`,
		`Etc/GMT+9`,
		`Etc/GMT-0`,
		`Etc/GMT-1`,
		`Etc/GMT-10`,
		`Etc/GMT-11`,
		`Etc/GMT-12`,
		`Etc/GMT-13`,
		`Etc/GMT-14`,
		`Etc/GMT-2`,
		`Etc/GMT-3`,
		`Etc/GMT-4`,
		`Etc/GMT-5`,
		`Etc/GMT-6`,
		`Etc/GMT-7`,
		`Etc/GMT-8`,
		`Etc/GMT-9`,
		`Etc/GMT0`,
		`Etc/Greenwich`,
		`Etc/UCT`,
		`Etc/Universal`,
		`Etc/UTC`,
		`Etc/Zulu`,
		`Europe/Amsterdam`,
		`Europe/Andorra`,
		`Europe/Astrakhan`,
		`Europe/Athens`,
		`Europe/Belfast`,
		`Europe/Belgrade`,
		`Europe/Berlin`,
		`Europe/Bratislava`,
		`Europe/Brussels`,
		`Europe/Bucharest`,
		`Europe/Budapest`,
		`Europe/Busingen`,
		`Europe/Chisinau`,
		`Europe/Copenhagen`,
		`Europe/Dublin`,
		`Europe/Gibraltar`,
		`Europe/Guernsey`,
		`Europe/Helsinki`,
		`Europe/Isle_of_Man`,
		`Europe/Istanbul`,
		`Europe/Jersey`,
		`Europe/Kaliningrad`,
		`Europe/Kiev`,
		`Europe/Kirov`,
		`Europe/Lisbon`,
		`Europe/Ljubljana`,
		`Europe/London`,
		`Europe/Luxembourg`,
		`Europe/Madrid`,
		`Europe/Malta`,
		`Europe/Mariehamn`,
		`Europe/Minsk`,
		`Europe/Monaco`,
		`Europe/Moscow`,
		`Europe/Nicosia`,
		`Europe/Oslo`,
		`Europe/Paris`,
		`Europe/Podgorica`,
		`Europe/Prague`,
		`Europe/Riga`,
		`Europe/Rome`,
		`Europe/Samara`,
		`Europe/San_Marino`,
		`Europe/Sarajevo`,
		`Europe/Saratov`,
		`Europe/Simferopol`,
		`Europe/Skopje`,
		`Europe/Sofia`,
		`Europe/Stockholm`,
		`Europe/Tallinn`,
		`Europe/Tirane`,
		`Europe/Tiraspol`,
		`Europe/Ulyanovsk`,
		`Europe/Uzhgorod`,
		`Europe/Vaduz`,
		`Europe/Vatican`,
		`Europe/Vienna`,
		`Europe/Vilnius`,
		`Europe/Volgograd`,
		`Europe/Warsaw`,
		`Europe/Zagreb`,
		`Europe/Zaporozhye`,
		`Europe/Zurich`,
		`Factory`,
		`GB`,
		`GB-Eire`,
		`GMT`,
		`GMT+0`,
		`GMT-0`,
		`GMT0`,
		`Greenwich`,
		`Hongkong`,
		`HST`,
		`Iceland`,
		`Indian/Antananarivo`,
		`Indian/Chagos`,
		`Indian/Christmas`,
		`Indian/Cocos`,
		`Indian/Comoro`,
		`Indian/Kerguelen`,
		`Indian/Mahe`,
		`Indian/Maldives`,
		`Indian/Mauritius`,
		`Indian/Mayotte`,
		`Indian/Reunion`,
		`Iran`,
		`Israel`,
		`Jamaica`,
		`Japan`,
		`Kwajalein`,
		`Libya`,
		`MET`,
		`Mexico/BajaNorte`,
		`Mexico/BajaSur`,
		`Mexico/General`,
		`MST`,
		`MST7MDT`,
		`Navajo`,
		`NZ`,
		`NZ-CHAT`,
		`Pacific/Apia`,
		`Pacific/Auckland`,
		`Pacific/Bougainville`,
		`Pacific/Chatham`,
		`Pacific/Chuuk`,
		`Pacific/Easter`,
		`Pacific/Efate`,
		`Pacific/Enderbury`,
		`Pacific/Fakaofo`,
		`Pacific/Fiji`,
		`Pacific/Funafuti`,
		`Pacific/Galapagos`,
		`Pacific/Gambier`,
		`Pacific/Guadalcanal`,
		`Pacific/Guam`,
		`Pacific/Honolulu`,
		`Pacific/Johnston`,
		`Pacific/Kanton`,
		`Pacific/Kiritimati`,
		`Pacific/Kosrae`,
		`Pacific/Kwajalein`,
		`Pacific/Majuro`,
		`Pacific/Marquesas`,
		`Pacific/Midway`,
		`Pacific/Nauru`,
		`Pacific/Niue`,
		`Pacific/Norfolk`,
		`Pacific/Noumea`,
		`Pacific/Pago_Pago`,
		`Pacific/Palau`,
		`Pacific/Pitcairn`,
		`Pacific/Pohnpei`,
		`Pacific/Ponape`,
		`Pacific/Port_Moresby`,
		`Pacific/Rarotonga`,
		`Pacific/Saipan`,
		`Pacific/Samoa`,
		`Pacific/Tahiti`,
		`Pacific/Tarawa`,
		`Pacific/Tongatapu`,
		`Pacific/Truk`,
		`Pacific/Wake`,
		`Pacific/Wallis`,
		`Pacific/Yap`,
		`Poland`,
		`Portugal`,
		`PRC`,
		`PST8PDT`,
		`ROC`,
		`ROK`,
		`Singapore`,
		`Turkey`,
		`UCT`,
		`Universal`,
		`US/Alaska`,
		`US/Aleutian`,
		`US/Arizona`,
		`US/Central`,
		`US/East-Indiana`,
		`US/Eastern`,
		`US/Hawaii`,
		`US/Indiana-Starke`,
		`US/Michigan`,
		`US/Mountain`,
		`US/Pacific`,
		`US/Samoa`,
		`UTC`,
		`W-SU`,
		`WET`,
		`Zulu`,
	}
	for _, tz := range data {
		errs := validateTimeZone(&tz, nil)
		if len(errs) > 0 {
			t.Errorf("%s failed: %v", tz, errs)
		}
	}
}

func TestValidateIndexesString(t *testing.T) {
	testCases := map[string]struct {
		indexesString string
		completions   int32
		wantTotal     int32
		wantError     error
	}{
		"empty is valid": {
			indexesString: "",
			completions:   6,
			wantTotal:     0,
		},
		"single number is valid": {
			indexesString: "1",
			completions:   6,
			wantTotal:     1,
		},
		"single interval is valid": {
			indexesString: "1-3",
			completions:   6,
			wantTotal:     3,
		},
		"mixed intervals valid": {
			indexesString: "0,1-3,5,7-10",
			completions:   12,
			wantTotal:     9,
		},
		"invalid due to extra space": {
			indexesString: "0,1-3, 5",
			completions:   6,
			wantTotal:     0,
			wantError:     errors.New(`cannot convert string to integer for index: " 5"`),
		},
		"invalid due to too large index": {
			indexesString: "0,1-3,5",
			completions:   5,
			wantTotal:     0,
			wantError:     errors.New(`too large index: "5"`),
		},
		"invalid due to non-increasing order of intervals": {
			indexesString: "1-3,0,5",
			completions:   6,
			wantTotal:     0,
			wantError:     errors.New(`non-increasing order, previous: 3, current: 0`),
		},
		"invalid due to non-increasing order between intervals": {
			indexesString: "0,0,5",
			completions:   6,
			wantTotal:     0,
			wantError:     errors.New(`non-increasing order, previous: 0, current: 0`),
		},
		"invalid due to non-increasing order within interval": {
			indexesString: "0,1-1,5",
			completions:   6,
			wantTotal:     0,
			wantError:     errors.New(`non-increasing order, previous: 1, current: 1`),
		},
		"invalid due to starting with '-'": {
			indexesString: "-1,0",
			completions:   6,
			wantTotal:     0,
			wantError:     errors.New(`cannot convert string to integer for index: ""`),
		},
		"invalid due to ending with '-'": {
			indexesString: "0,1-",
			completions:   6,
			wantTotal:     0,
			wantError:     errors.New(`cannot convert string to integer for index: ""`),
		},
		"invalid due to repeated '-'": {
			indexesString: "0,1--3",
			completions:   6,
			wantTotal:     0,
			wantError:     errors.New(`the fragment "1--3" violates the requirement that an index interval can have at most two parts separated by '-'`),
		},
		"invalid due to repeated ','": {
			indexesString: "0,,1,3",
			completions:   6,
			wantTotal:     0,
			wantError:     errors.New(`cannot convert string to integer for index: ""`),
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			gotTotal, gotErr := validateIndexesFormat(tc.indexesString, tc.completions)
			if tc.wantError == nil && gotErr != nil {
				t.Errorf("unexpected error: %s", gotErr)
			} else if tc.wantError != nil && gotErr == nil {
				t.Errorf("missing error: %s", tc.wantError)
			} else if tc.wantError != nil && gotErr != nil {
				if diff := cmp.Diff(tc.wantError.Error(), gotErr.Error()); diff != "" {
					t.Errorf("unexpected error, diff: %s", diff)
				}
			}
			if tc.wantTotal != gotTotal {
				t.Errorf("unexpected total want:%d, got:%d", tc.wantTotal, gotTotal)
			}
		})
	}
}

func TestValidateFailedIndexesNotOverlapCompleted(t *testing.T) {
	testCases := map[string]struct {
		completedIndexesStr string
		failedIndexesStr    string
		completions         int32
		wantError           error
	}{
		"empty intervals": {
			completedIndexesStr: "",
			failedIndexesStr:    "",
			completions:         6,
		},
		"empty completed intervals": {
			completedIndexesStr: "",
			failedIndexesStr:    "1-3",
			completions:         6,
		},
		"empty failed intervals": {
			completedIndexesStr: "1-2",
			failedIndexesStr:    "",
			completions:         6,
		},
		"non-overlapping intervals": {
			completedIndexesStr: "0,2-4,6-8,12-19",
			failedIndexesStr:    "1,9-10",
			completions:         20,
		},
		"overlapping intervals": {
			completedIndexesStr: "0,2-4,6-8,12-19",
			failedIndexesStr:    "1,8,9-10",
			completions:         20,
			wantError:           errors.New("failedIndexes and completedIndexes overlap at index: 8"),
		},
		"overlapping intervals, corrupted completed interval skipped": {
			completedIndexesStr: "0,2-4,x,6-8,12-19",
			failedIndexesStr:    "1,8,9-10",
			completions:         20,
			wantError:           errors.New("failedIndexes and completedIndexes overlap at index: 8"),
		},
		"overlapping intervals, corrupted failed interval skipped": {
			completedIndexesStr: "0,2-4,6-8,12-19",
			failedIndexesStr:    "1,y,8,9-10",
			completions:         20,
			wantError:           errors.New("failedIndexes and completedIndexes overlap at index: 8"),
		},
		"overlapping intervals, first corrupted intervals skipped": {
			completedIndexesStr: "x,0,2-4,6-8,12-19",
			failedIndexesStr:    "y,1,8,9-10",
			completions:         20,
			wantError:           errors.New("failedIndexes and completedIndexes overlap at index: 8"),
		},
		"non-overlapping intervals, last intervals corrupted": {
			completedIndexesStr: "0,2-4,6-8,12-19,x",
			failedIndexesStr:    "1,9-10,y",
			completions:         20,
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			gotErr := validateFailedIndexesNotOverlapCompleted(tc.completedIndexesStr, tc.failedIndexesStr, tc.completions)
			if tc.wantError == nil && gotErr != nil {
				t.Errorf("unexpected error: %s", gotErr)
			} else if tc.wantError != nil && gotErr == nil {
				t.Errorf("missing error: %s", tc.wantError)
			} else if tc.wantError != nil && gotErr != nil {
				if diff := cmp.Diff(tc.wantError.Error(), gotErr.Error()); diff != "" {
					t.Errorf("unexpected error, diff: %s", diff)
				}
			}
		})
	}
}
