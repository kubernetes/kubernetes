/*
Copyright 2019 The Kubernetes Authors.

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
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
	"math"
	"testing"
)

func TestFlowSchemaValidation(t *testing.T) {
	testCases := []struct {
		name           string
		flowSchema     *flowcontrol.FlowSchema
		expectedErrors field.ErrorList
	}{
		{
			name: "empty spec should fail",
			flowSchema: &flowcontrol.FlowSchema{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-foo",
				},
				Spec: flowcontrol.FlowSchemaSpec{},
			},
			expectedErrors: field.ErrorList{
				field.Required(field.NewPath("spec").Child("priorityLevelConfiguration").Child("name"), "must reference to a priority level"),
				field.Required(field.NewPath("spec").Child("rules"), "rules must contain at least one value"),
			},
		},
		{
			name: "missing policy-rule should fail",
			flowSchema: &flowcontrol.FlowSchema{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-foo",
				},
				Spec: flowcontrol.FlowSchemaSpec{
					PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
						Name: "system-bar",
					},
					Rules: []flowcontrol.PolicyRuleWithSubjects{
						{
							Subjects: []flowcontrol.Subject{
								{
									Kind:     flowcontrol.UserKind,
									APIGroup: flowcontrol.GroupName,
									Name:     "noxu",
								},
							},
						},
					},
				},
			},
			expectedErrors: field.ErrorList{
				field.Required(field.NewPath("spec").Child("rules").Index(0).Child("rule").Child("verbs"), "verbs must contain at least one value"),
				field.Required(field.NewPath("spec").Child("rules").Index(0).Child("rule").Child("apiGroups"), "resource rules must supply at least one api group"),
				field.Required(field.NewPath("spec").Child("rules").Index(0).Child("rule").Child("resources"), "resource rules must supply at least one resource"),
			},
		},
		{
			name: "normal flow-schema w/ * verbs/apiGroups/resources should work",
			flowSchema: &flowcontrol.FlowSchema{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-foo",
				},
				Spec: flowcontrol.FlowSchemaSpec{
					PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
						Name: "system-bar",
					},
					Rules: []flowcontrol.PolicyRuleWithSubjects{
						{
							Subjects: []flowcontrol.Subject{
								{
									Kind:     flowcontrol.UserKind,
									APIGroup: flowcontrol.GroupName,
									Name:     "noxu",
								},
							},
							Rule: flowcontrol.PolicyRule{
								Verbs:     []string{flowcontrol.VerbAll},
								APIGroups: []string{flowcontrol.APIGroupAll},
								Resources: []string{flowcontrol.ResourceAll},
							},
						},
					},
				},
			},
			expectedErrors: field.ErrorList{},
		},
		{
			name: "flow-schema mixes * verbs/apiGroups/resources should fail",
			flowSchema: &flowcontrol.FlowSchema{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-foo",
				},
				Spec: flowcontrol.FlowSchemaSpec{
					PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
						Name: "system-bar",
					},
					Rules: []flowcontrol.PolicyRuleWithSubjects{
						{
							Subjects: []flowcontrol.Subject{
								{
									Kind:     flowcontrol.UserKind,
									APIGroup: flowcontrol.GroupName,
									Name:     "noxu",
								},
							},
							Rule: flowcontrol.PolicyRule{
								Verbs:     []string{flowcontrol.VerbAll, "create"},
								APIGroups: []string{flowcontrol.APIGroupAll, "tak"},
								Resources: []string{flowcontrol.ResourceAll, "tok"},
							},
						},
					},
				},
			},
			expectedErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("rules").Index(0).Child("rule").Child("verbs"), []string{"*", "create"}, "if '*' is present, must not specify other verbs"),
				field.Invalid(field.NewPath("spec").Child("rules").Index(0).Child("rule").Child("apiGroups"), []string{"*", "tak"}, "if '*' is present, must not specify other api groups"),
				field.Invalid(field.NewPath("spec").Child("rules").Index(0).Child("rule").Child("resources"), []string{"*", "tok"}, "if '*' is present, must not specify other resources"),
			},
		},
		{
			name: "flow-schema mixes non-resource URLs w/ regular resources should fail",
			flowSchema: &flowcontrol.FlowSchema{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-foo",
				},
				Spec: flowcontrol.FlowSchemaSpec{
					PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
						Name: "system-bar",
					},
					Rules: []flowcontrol.PolicyRuleWithSubjects{
						{
							Subjects: []flowcontrol.Subject{
								{
									Kind:     flowcontrol.UserKind,
									APIGroup: flowcontrol.GroupName,
									Name:     "noxu",
								},
							},
							Rule: flowcontrol.PolicyRule{
								Verbs:           []string{flowcontrol.VerbAll, "create"},
								APIGroups:       []string{flowcontrol.APIGroupAll, "tak"},
								Resources:       []string{flowcontrol.ResourceAll, "tok"},
								NonResourceURLs: []string{flowcontrol.NonResourceAll},
							},
						},
					},
				},
			},
			expectedErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("rules").Index(0).Child("rule").Child("nonResourceURLs"), []string{"*"}, "rules cannot apply to both regular resources and non-resource URLs"),
			},
		},
		{
			name: "flow-schema mixes * non-resource URLs should fail",
			flowSchema: &flowcontrol.FlowSchema{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-foo",
				},
				Spec: flowcontrol.FlowSchemaSpec{
					PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
						Name: "system-bar",
					},
					Rules: []flowcontrol.PolicyRuleWithSubjects{
						{
							Subjects: []flowcontrol.Subject{
								{
									Kind:     flowcontrol.UserKind,
									APIGroup: flowcontrol.GroupName,
									Name:     "noxu",
								},
							},
							Rule: flowcontrol.PolicyRule{
								Verbs:           []string{"*"},
								NonResourceURLs: []string{flowcontrol.NonResourceAll, "tik"},
							},
						},
					},
				},
			},
			expectedErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("rules").Index(0).Child("rule").Child("nonResourceURLs"), []string{"*", "tik"}, "if '*' is present, must not specify other non-resource URLs"),
			},
		},
		{
			name: "normal flow-schema mixes user should fail",
			flowSchema: &flowcontrol.FlowSchema{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-foo",
				},
				Spec: flowcontrol.FlowSchemaSpec{
					PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
						Name: "system-bar",
					},
					Rules: []flowcontrol.PolicyRuleWithSubjects{
						{
							Subjects: []flowcontrol.Subject{
								{
									Kind:     flowcontrol.UserKind,
									APIGroup: flowcontrol.GroupName,
									Name:     "noxu",
								},
							},
							Rule: flowcontrol.PolicyRule{
								Verbs:           []string{"*"},
								NonResourceURLs: []string{flowcontrol.NonResourceAll, "tik"},
							},
						},
					},
				},
			},
			expectedErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("rules").Index(0).Child("rule").Child("nonResourceURLs"), []string{"*", "tik"}, "if '*' is present, must not specify other non-resource URLs"),
			},
		},
		{
			name: "flow-schema w/ invalid verb should fail",
			flowSchema: &flowcontrol.FlowSchema{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-foo",
				},
				Spec: flowcontrol.FlowSchemaSpec{
					PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
						Name: "system-bar",
					},
					Rules: []flowcontrol.PolicyRuleWithSubjects{
						{
							Subjects: []flowcontrol.Subject{
								{
									Kind:     flowcontrol.UserKind,
									APIGroup: flowcontrol.GroupName,
									Name:     "noxu",
								},
							},
							Rule: flowcontrol.PolicyRule{
								Verbs:     []string{"feed"},
								APIGroups: []string{flowcontrol.APIGroupAll},
								Resources: []string{flowcontrol.ResourceAll},
							},
						},
					},
				},
			},
			expectedErrors: field.ErrorList{
				field.NotSupported(field.NewPath("spec").Child("rules").Index(0).Child("rule").Child("verbs"), []string{"feed"}, supportedVerbs.List()),
			},
		},
		{
			name: "flow-schema w/ invalid priority level configuration name should fail",
			flowSchema: &flowcontrol.FlowSchema{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-foo",
				},
				Spec: flowcontrol.FlowSchemaSpec{
					PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
						Name: "system+++$$",
					},
					Rules: []flowcontrol.PolicyRuleWithSubjects{
						{
							Subjects: []flowcontrol.Subject{
								{
									Kind:     flowcontrol.UserKind,
									APIGroup: flowcontrol.GroupName,
									Name:     "noxu",
								},
							},
							Rule: flowcontrol.PolicyRule{
								Verbs:     []string{flowcontrol.VerbAll},
								APIGroups: []string{flowcontrol.APIGroupAll},
								Resources: []string{flowcontrol.ResourceAll},
							},
						},
					},
				},
			},
			expectedErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("priorityLevelConfiguration").Child("name"), "system+++$$", `a DNS-1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')`),
			},
		},
		{
			name: "flow-schema w/ service-account kind missing namespace should fail",
			flowSchema: &flowcontrol.FlowSchema{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-foo",
				},
				Spec: flowcontrol.FlowSchemaSpec{
					PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
						Name: "system-bar",
					},
					Rules: []flowcontrol.PolicyRuleWithSubjects{
						{
							Subjects: []flowcontrol.Subject{
								{
									Kind: flowcontrol.ServiceAccountKind,
									Name: "noxu",
								},
							},
							Rule: flowcontrol.PolicyRule{
								Verbs:     []string{flowcontrol.VerbAll},
								APIGroups: []string{flowcontrol.APIGroupAll},
								Resources: []string{flowcontrol.ResourceAll},
							},
						},
					},
				},
			},
			expectedErrors: field.ErrorList{
				field.Required(field.NewPath("spec").Child("rules").Index(0).Child("subjects").Index(0).Child("namespace"), "must specify namespace for service account"),
			},
		},
		{
			name: "flow-schema missing kind should fail",
			flowSchema: &flowcontrol.FlowSchema{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-foo",
				},
				Spec: flowcontrol.FlowSchemaSpec{
					PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
						Name: "system-bar",
					},
					Rules: []flowcontrol.PolicyRuleWithSubjects{
						{
							Subjects: []flowcontrol.Subject{
								{
									Kind: "",
								},
							},
							Rule: flowcontrol.PolicyRule{
								Verbs:     []string{flowcontrol.VerbAll},
								APIGroups: []string{flowcontrol.APIGroupAll},
								Resources: []string{flowcontrol.ResourceAll},
							},
						},
					},
				},
			},
			expectedErrors: field.ErrorList{
				field.Required(field.NewPath("spec").Child("rules").Index(0).Child("subjects").Index(0).Child("name"), ""),
				field.NotSupported(field.NewPath("spec").Child("rules").Index(0).Child("subjects").Index(0).Child("kind"), "", supportedSubjectKinds.List()),
			},
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			errs := ValidateFlowSchema(testCase.flowSchema)
			if !assert.ElementsMatch(t, testCase.expectedErrors, errs) {
				t.Logf("mismatch: %v", cmp.Diff(testCase.expectedErrors, errs))
			}
		})
	}
}

func TestPriorityLevelConfigurationValidation(t *testing.T) {
	testCases := []struct {
		name                       string
		priorityLevelConfiguration *flowcontrol.PriorityLevelConfiguration
		expectedErrors             field.ErrorList
	}{
		{
			name: "normal customized priority level should work",
			priorityLevelConfiguration: &flowcontrol.PriorityLevelConfiguration{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-foo",
				},
				Spec: flowcontrol.PriorityLevelConfigurationSpec{
					Exempt:                   false,
					AssuredConcurrencyShares: 100,
					Queues:                   512,
					HandSize:                 4,
					QueueLengthLimit:         100,
				},
			},
			expectedErrors: field.ErrorList{},
		},
		{
			name: "system low priority level w/ exempt should work",
			priorityLevelConfiguration: &flowcontrol.PriorityLevelConfiguration{
				ObjectMeta: metav1.ObjectMeta{
					Name: flowcontrol.PriorityLevelConfigurationNameExempt,
				},
				Spec: flowcontrol.PriorityLevelConfigurationSpec{
					Exempt: true,
				},
			},
			expectedErrors: field.ErrorList{},
		},
		{
			name: "customized priority level w/ empty spec should fail",
			priorityLevelConfiguration: &flowcontrol.PriorityLevelConfiguration{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-foo",
				},
				Spec: flowcontrol.PriorityLevelConfigurationSpec{},
			},
			expectedErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("assuredConcurrencyShares"), int32(0), "must be positive"),
				field.Invalid(field.NewPath("spec").Child("queueLengthLimit"), int32(0), "must be positive"),
				field.Invalid(field.NewPath("spec").Child("queues"), int32(0), "must be positive"),
				field.Invalid(field.NewPath("spec").Child("handSize"), int32(0), "handSize is not positive; deckSize is not positive"),
			},
		},
		{
			name: "customized priority level w/ overflowing handSize/queues should fail 1",
			priorityLevelConfiguration: &flowcontrol.PriorityLevelConfiguration{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-foo",
				},
				Spec: flowcontrol.PriorityLevelConfigurationSpec{
					AssuredConcurrencyShares: 100,
					QueueLengthLimit:         100,
					Queues:                   512,
					HandSize:                 8,
				},
			},
			expectedErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("handSize"), int32(8), "more than 60 bits of entropy required"),
			},
		},
		{
			name: "customized priority level w/ overflowing handSize/queues should fail 2",
			priorityLevelConfiguration: &flowcontrol.PriorityLevelConfiguration{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-foo",
				},
				Spec: flowcontrol.PriorityLevelConfigurationSpec{
					AssuredConcurrencyShares: 100,
					QueueLengthLimit:         100,
					Queues:                   128,
					HandSize:                 10,
				},
			},
			expectedErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("handSize"), int32(10), "more than 60 bits of entropy required"),
			},
		},
		{
			name: "customized priority level w/ overflowing handSize/queues should fail 3",
			priorityLevelConfiguration: &flowcontrol.PriorityLevelConfiguration{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-foo",
				},
				Spec: flowcontrol.PriorityLevelConfigurationSpec{
					AssuredConcurrencyShares: 100,
					QueueLengthLimit:         100,
					Queues:                   math.MaxInt32,
					HandSize:                 3,
				},
			},
			expectedErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("handSize"), int32(3), "more than 60 bits of entropy required"),
			},
		},
		{
			name: "customized priority level w/ handSize=2 and queues=max-int32 should work",
			priorityLevelConfiguration: &flowcontrol.PriorityLevelConfiguration{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-foo",
				},
				Spec: flowcontrol.PriorityLevelConfigurationSpec{
					AssuredConcurrencyShares: 100,
					QueueLengthLimit:         100,
					Queues:                   1<<30 - 1, // max integer in 30 bits
					HandSize:                 2,
				},
			},
			expectedErrors: field.ErrorList{},
		},
		{
			name: "customized priority level w/ handSize greater than queues should fail",
			priorityLevelConfiguration: &flowcontrol.PriorityLevelConfiguration{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-foo",
				},
				Spec: flowcontrol.PriorityLevelConfigurationSpec{
					AssuredConcurrencyShares: 100,
					QueueLengthLimit:         100,
					Queues:                   7,
					HandSize:                 8,
				},
			},
			expectedErrors: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("handSize"), int32(8), "should not be greater than queues (7)"),
				field.Invalid(field.NewPath("spec").Child("handSize"), int32(8), "handSize is greater than deckSize"),
			},
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			errs := ValidatePriorityLevelConfiguration(testCase.priorityLevelConfiguration)
			if !assert.ElementsMatch(t, testCase.expectedErrors, errs) {
				t.Logf("mismatch: %v", cmp.Diff(testCase.expectedErrors, errs))
			}
		})
	}
}
