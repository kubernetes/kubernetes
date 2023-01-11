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
	"archive/zip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	corevalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/utils/pointer"
)

var (
	timeZoneEmpty      = ""
	timeZoneLocal      = "LOCAL"
	timeZoneUTC        = "UTC"
	timeZoneCorrect    = "Continent/Zone"
	timeZoneBadPrefix  = " Continent/Zone"
	timeZoneBadSuffix  = "Continent/Zone "
	timeZoneBadName    = "Continent/InvalidZone"
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
			RestartPolicy:  api.RestartPolicyOnFailure,
			DNSPolicy:      api.DNSClusterFirst,
			Containers:     []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
			InitContainers: []api.Container{{Name: "def", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
	}
}

func TestValidateJob(t *testing.T) {
	validJobObjectMeta := metav1.ObjectMeta{
		Name:      "myjob",
		Namespace: metav1.NamespaceDefault,
		UID:       types.UID("1a2b3c"),
	}
	validManualSelector := getValidManualSelector()
	validPodTemplateSpecForManual := getValidPodTemplateSpecForManual(validManualSelector)
	validGeneratedSelector := getValidGeneratedSelector()
	validPodTemplateSpecForGenerated := getValidPodTemplateSpecForGenerated(validGeneratedSelector)
	validPodTemplateSpecForGeneratedRestartPolicyNever := getValidPodTemplateSpecForGenerated(validGeneratedSelector)
	validPodTemplateSpecForGeneratedRestartPolicyNever.Spec.RestartPolicy = api.RestartPolicyNever

	successCases := map[string]struct {
		opts JobValidationOptions
		job  batch.Job
	}{
		"valid pod failure policy": {
			job: batch.Job{
				ObjectMeta: validJobObjectMeta,
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{
							{
								Action: batch.PodFailurePolicyActionIgnore,
								OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{
									{
										Type:   api.DisruptionTarget,
										Status: api.ConditionTrue,
									},
								},
							},
							{
								Action: batch.PodFailurePolicyActionFailJob,
								OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{
									{
										Type:   api.PodConditionType("CustomConditionType"),
										Status: api.ConditionFalse,
									},
								},
							},
							{
								Action: batch.PodFailurePolicyActionCount,
								OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
									ContainerName: pointer.String("abc"),
									Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
									Values:        []int32{1, 2, 3},
								},
							},
							{
								Action: batch.PodFailurePolicyActionIgnore,
								OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
									ContainerName: pointer.String("def"),
									Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
									Values:        []int32{4},
								},
							},
							{
								Action: batch.PodFailurePolicyActionFailJob,
								OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
									Operator: batch.PodFailurePolicyOnExitCodesOpNotIn,
									Values:   []int32{5, 6, 7},
								},
							},
						},
					},
				},
			},
		},
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
		`spec.podFailurePolicy.rules[0]: Invalid value: specifying one of OnExitCodes and OnPodConditions is required`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{
						{
							Action: batch.PodFailurePolicyActionFailJob,
						},
					},
				},
			},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.values[1]: Duplicate value: 11`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{
						{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{11, 11},
							},
						},
					},
				},
			},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.values: Too many: 256: must have at most 255 items`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{
						{
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
						},
					},
				},
			},
		},
		`spec.podFailurePolicy.rules: Too many: 21: must have at most 20 items`: {
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
		`spec.podFailurePolicy.rules[0].onPodConditions: Too many: 21: must have at most 20 items`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{
						{
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
						},
					},
				},
			},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.values[2]: Duplicate value: 13`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{
						{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{12, 13, 13, 13},
							},
						},
					},
				},
			},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.values: Invalid value: []int32{19, 11}: must be ordered`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{
						{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{19, 11},
							},
						},
					},
				},
			},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.values: Invalid value: []int32{}: at least one value is required`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{
						{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{},
							},
						},
					},
				},
			},
		},
		`spec.podFailurePolicy.rules[0].action: Required value: valid values: ["Count" "FailJob" "Ignore"]`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{
						{
							Action: "",
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{1, 2, 3},
							},
						},
					},
				},
			},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.operator: Required value: valid values: ["In" "NotIn"]`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{
						{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: "",
								Values:   []int32{1, 2, 3},
							},
						},
					},
				},
			},
		},
		`spec.podFailurePolicy.rules[0]: Invalid value: specifying both OnExitCodes and OnPodConditions is not supported`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{
						{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								ContainerName: pointer.String("abc"),
								Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
								Values:        []int32{1, 2, 3},
							},
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{
								{
									Type:   api.DisruptionTarget,
									Status: api.ConditionTrue,
								},
							},
						},
					},
				},
			},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.values[1]: Invalid value: 0: must not be 0 for the In operator`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{
						{
							Action: batch.PodFailurePolicyActionIgnore,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: batch.PodFailurePolicyOnExitCodesOpIn,
								Values:   []int32{1, 0, 2},
							},
						},
					},
				},
			},
		},
		`spec.podFailurePolicy.rules[1].onExitCodes.containerName: Invalid value: "xyz": must be one of the container or initContainer names in the pod template`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{
						{
							Action: batch.PodFailurePolicyActionIgnore,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								ContainerName: pointer.String("abc"),
								Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
								Values:        []int32{1, 2, 3},
							},
						},
						{
							Action: batch.PodFailurePolicyActionFailJob,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								ContainerName: pointer.String("xyz"),
								Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
								Values:        []int32{5, 6, 7},
							},
						},
					},
				},
			},
		},
		`spec.podFailurePolicy.rules[0].action: Unsupported value: "UnknownAction": supported values: "Count", "FailJob", "Ignore"`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{
						{
							Action: "UnknownAction",
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								ContainerName: pointer.String("abc"),
								Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
								Values:        []int32{1, 2, 3},
							},
						},
					},
				},
			},
		},
		`spec.podFailurePolicy.rules[0].onExitCodes.operator: Unsupported value: "UnknownOperator": supported values: "In", "NotIn"`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{
						{
							Action: batch.PodFailurePolicyActionIgnore,
							OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
								Operator: "UnknownOperator",
								Values:   []int32{1, 2, 3},
							},
						},
					},
				},
			},
		},
		`spec.podFailurePolicy.rules[0].onPodConditions[0].status: Required value: valid values: ["False" "True" "Unknown"]`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{
						{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{
								{
									Type: api.DisruptionTarget,
								},
							},
						},
					},
				},
			},
		},
		`spec.podFailurePolicy.rules[0].onPodConditions[0].status: Unsupported value: "UnknownStatus": supported values: "False", "True", "Unknown"`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{
						{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{
								{
									Type:   api.DisruptionTarget,
									Status: "UnknownStatus",
								},
							},
						},
					},
				},
			},
		},
		`spec.podFailurePolicy.rules[0].onPodConditions[0].type: Invalid value: "": name part must be non-empty`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{
						{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{
								{
									Status: api.ConditionTrue,
								},
							},
						},
					},
				},
			},
		},
		`spec.podFailurePolicy.rules[0].onPodConditions[0].type: Invalid value: "Invalid Condition Type": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: validPodTemplateSpecForGeneratedRestartPolicyNever,
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{
						{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{
								{
									Type:   api.PodConditionType("Invalid Condition Type"),
									Status: api.ConditionTrue,
								},
							},
						},
					},
				},
			},
		},
		`spec.template.spec.restartPolicy: Invalid value: "OnFailure": only "Never" is supported when podFailurePolicy is specified`: {
			ObjectMeta: validJobObjectMeta,
			Spec: batch.JobSpec{
				Selector: validGeneratedSelector,
				Template: api.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: validGeneratedSelector.MatchLabels,
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyOnFailure,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
					},
				},
				PodFailurePolicy: &batch.PodFailurePolicy{
					Rules: []batch.PodFailurePolicyRule{},
				},
			},
		},
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
	validPodTemplateSpecForGeneratedRestartPolicyNever := getValidPodTemplateSpecForGenerated(validGeneratedSelector)
	validPodTemplateSpecForGeneratedRestartPolicyNever.Spec.RestartPolicy = api.RestartPolicyNever

	validNodeAffinity := &api.Affinity{
		NodeAffinity: &api.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
				NodeSelectorTerms: []api.NodeSelectorTerm{
					{
						MatchExpressions: []api.NodeSelectorRequirement{
							{
								Key:      "foo",
								Operator: api.NodeSelectorOpIn,
								Values:   []string{"bar", "value2"},
							},
						},
					},
				},
			},
		},
	}
	validPodTemplateWithAffinity := getValidPodTemplateSpecForGenerated(validGeneratedSelector)
	validPodTemplateWithAffinity.Spec.Affinity = &api.Affinity{
		NodeAffinity: &api.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
				NodeSelectorTerms: []api.NodeSelectorTerm{
					{
						MatchExpressions: []api.NodeSelectorRequirement{
							{
								Key:      "foo",
								Operator: api.NodeSelectorOpIn,
								Values:   []string{"bar", "value"},
							},
						},
					},
				},
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
					Rules: []batch.PodFailurePolicyRule{
						{
							Action: batch.PodFailurePolicyActionIgnore,
							OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{
								{
									Type:   api.DisruptionTarget,
									Status: api.ConditionTrue,
								},
							},
						},
					},
				}
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
						Rules: []batch.PodFailurePolicyRule{
							{
								Action: batch.PodFailurePolicyActionIgnore,
								OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{
									{
										Type:   api.DisruptionTarget,
										Status: api.ConditionTrue,
									},
								},
							},
						},
					},
				},
			},
			update: func(job *batch.Job) {
				job.Spec.PodFailurePolicy.Rules = append(job.Spec.PodFailurePolicy.Rules, batch.PodFailurePolicyRule{
					Action: batch.PodFailurePolicyActionCount,
					OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{
						{
							Type:   api.DisruptionTarget,
							Status: api.ConditionTrue,
						},
					},
				})
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
						Rules: []batch.PodFailurePolicyRule{
							{
								Action: batch.PodFailurePolicyActionIgnore,
								OnPodConditions: []batch.PodFailurePolicyOnPodConditionsPattern{
									{
										Type:   api.DisruptionTarget,
										Status: api.ConditionTrue,
									},
								},
							},
						},
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
		"immutable pod template": {
			old: batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
				Spec: batch.JobSpec{
					Selector: validGeneratedSelector,
					Template: validPodTemplateSpecForGenerated,
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
					Active:    2,
					Succeeded: 3,
					Failed:    4,
					Ready:     pointer.Int32(1),
				},
			},
		},
		"nil ready": {
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
					Failed:    -3,
					Ready:     pointer.Int32(-1),
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.active"},
				{Type: field.ErrorTypeInvalid, Field: "status.succeeded"},
				{Type: field.ErrorTypeInvalid, Field: "status.failed"},
				{Type: field.ErrorTypeInvalid, Field: "status.ready"},
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

	zoneDir := t.TempDir()
	if err := setupFakeTimeZoneDatabase(zoneDir); err != nil {
		t.Fatalf("Unexpected error setting up fake timezone database: %v", err)
	}
	t.Setenv("ZONEINFO", zoneDir)

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
		"spec.schedule: cannot use both timeZone field and TZ or CRON_TZ in schedule": {
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mycronjob",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("1a2b3c"),
			},
			Spec: batch.CronJobSpec{
				Schedule:          "TZ=UTC 0 * * * *",
				TimeZone:          &timeZoneUTC,
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

	// NOTE: The following test doesn't fail on Windows.
	// When opening a file on Windows, syscall.Open is called, which will trim the whitespace suffix.
	if runtime.GOOS != "windows" {
		errorCases["spec.timeZone: Invalid value: \"Continent/Zone \": unknown time zone Continent/Zone "] = batch.CronJob{
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
		}
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

// Sets up fake timezone database in a zoneDir directory with a single valid
// time zone called "Continent/Zone" by copying UTC metadata from golang's
// built-in databse. Returns an error in case of problems.
func setupFakeTimeZoneDatabase(zoneDir string) error {
	reader, err := zip.OpenReader(runtime.GOROOT() + "/lib/time/zoneinfo.zip")
	if err != nil {
		return err
	}
	defer reader.Close()

	if err := os.Mkdir(filepath.Join(zoneDir, "Continent"), os.ModePerm); err != nil {
		return err
	}
	zoneFile, err := os.OpenFile(filepath.Join(zoneDir, "Continent", "Zone"), os.O_RDWR|os.O_TRUNC|os.O_CREATE, 0666)
	if err != nil {
		return err
	}
	defer zoneFile.Close()

	for _, file := range reader.File {
		if file.Name != "UTC" {
			continue
		}
		rc, err := file.Open()
		if err != nil {
			return err
		}
		if _, err := io.Copy(zoneFile, rc); err != nil {
			return err
		}
		rc.Close()
		break
	}
	return nil
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
