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

package flowschema

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
)

func TestDeclarativeValidate(t *testing.T) {
	apiVersions := []string{"v1", "v1beta3", "v1beta2", "v1beta1"}
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "flowcontrol.apiserver.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "flowschemas",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		input        flowcontrol.FlowSchema
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidFlowSchema(),
		},
		"invalid empty priorityLevelConfiguration name": {
			input: mkValidFlowSchema(func(obj *flowcontrol.FlowSchema) {
				obj.Spec.PriorityLevelConfiguration.Name = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "priorityLevelConfiguration", "name"), ""),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	apiVersions := []string{"v1", "v1beta3", "v1beta2", "v1beta1"}
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	testCases := map[string]struct {
		oldObj       flowcontrol.FlowSchema
		updateObj    flowcontrol.FlowSchema
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidFlowSchema(func(obj *flowcontrol.FlowSchema) { obj.ResourceVersion = "1" }),
			updateObj: mkValidFlowSchema(func(obj *flowcontrol.FlowSchema) { obj.ResourceVersion = "1" }),
		},
		"invalid update empty priorityLevelConfiguration name": {
			oldObj: mkValidFlowSchema(func(obj *flowcontrol.FlowSchema) { obj.ResourceVersion = "1" }),
			updateObj: mkValidFlowSchema(func(obj *flowcontrol.FlowSchema) {
				obj.ResourceVersion = "1"
				obj.Spec.PriorityLevelConfiguration.Name = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "priorityLevelConfiguration", "name"), ""),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "flowcontrol.apiserver.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "flowschemas",
				Name:              "valid-flow-schema",
				IsResourceRequest: true,
				Verb:              "update",
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func mkValidFlowSchema(tweaks ...func(obj *flowcontrol.FlowSchema)) flowcontrol.FlowSchema {
	obj := flowcontrol.FlowSchema{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-flow-schema",
		},
		Spec: flowcontrol.FlowSchemaSpec{
			PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
				Name: "system-leader-election",
			},
			MatchingPrecedence: 1000,
			Rules: []flowcontrol.PolicyRulesWithSubjects{
				{
					Subjects: []flowcontrol.Subject{
						{
							Kind: flowcontrol.SubjectKindUser,
							User: &flowcontrol.UserSubject{
								Name: "system:kube-controller-manager",
							},
						},
					},
					ResourceRules: []flowcontrol.ResourcePolicyRule{
						{
							Verbs:      []string{"*"},
							APIGroups:  []string{"*"},
							Resources:  []string{"*"},
							Namespaces: []string{"*"},
						},
					},
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}
