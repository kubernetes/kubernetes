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

package flowschema

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	flowcontrol "k8s.io/kubernetes/pkg/apis/flowcontrol"
	registry "k8s.io/kubernetes/pkg/registry/flowcontrol/flowschema"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
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
		"user subject name is required": {
			input: mkValidFlowSchema(tweakUserSubjectName("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "rules").Index(0).Child("subjects").Index(0).Child("user.name"), "").MarkCoveredByDeclarative().MarkAlpha(),
			},
		},
		"subjects is required": {
			input: mkValidFlowSchema(clearSubjects()),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "rules").Index(0).Child("subjects"), ""),
			},
		},
		"subject kind is required": {
			input: mkValidFlowSchema(tweakSubjectKind("")),
			expectedErrs: field.ErrorList{
				field.NotSupported(field.NewPath("spec", "rules").Index(0).Child("subjects").Index(0).Child("kind"), "", []string{}),
			},
		},
		"subject kind does not match set member": {
			input: mkValidFlowSchema(tweakSubjectKind(flowcontrol.SubjectKindGroup)),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "rules").Index(0).Child("subjects").Index(0).Child("group"), ""),
				field.Forbidden(field.NewPath("spec", "rules").Index(0).Child("subjects").Index(0).Child("user"), ""),
			},
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, registry.Strategy, tc.expectedErrs)
		})
	}

	obj := mkValidFlowSchema()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	testCases := map[string]struct {
		oldObj       flowcontrol.FlowSchema
		updateObj    flowcontrol.FlowSchema
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidFlowSchema(),
			updateObj: mkValidFlowSchema(),
		},
		"user subject name is required": {
			oldObj:    mkValidFlowSchema(),
			updateObj: mkValidFlowSchema(tweakUserSubjectName("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "rules").Index(0).Child("subjects").Index(0).Child("user.name"), "").MarkCoveredByDeclarative().MarkAlpha(),
			},
		},
		"subjects is required": {
			oldObj:    mkValidFlowSchema(),
			updateObj: mkValidFlowSchema(clearSubjects()),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "rules").Index(0).Child("subjects"), ""),
			},
		},
		"subject kind is required": {
			oldObj:    mkValidFlowSchema(),
			updateObj: mkValidFlowSchema(tweakSubjectKind("")),
			expectedErrs: field.ErrorList{
				field.NotSupported(field.NewPath("spec", "rules").Index(0).Child("subjects").Index(0).Child("kind"), "", []string{}),
			},
		},
		"subject kind does not match set member": {
			oldObj:    mkValidFlowSchema(),
			updateObj: mkValidFlowSchema(tweakSubjectKind(flowcontrol.SubjectKindGroup)),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "rules").Index(0).Child("subjects").Index(0).Child("group"), ""),
				field.Forbidden(field.NewPath("spec", "rules").Index(0).Child("subjects").Index(0).Child("user"), ""),
			},
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			tc.oldObj.ResourceVersion = "1"
			tc.updateObj.ResourceVersion = "1"
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "flowcontrol.apiserver.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "flowschemas",
				Name:              "valid-obj",
				IsResourceRequest: true,
				Verb:              "update",
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, registry.Strategy, tc.expectedErrs)
		})
	}

	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "flowcontrol.apiserver.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "flowschemas",
		Name:              "valid-obj",
		IsResourceRequest: true,
		Verb:              "update",
	})

	updateObj := mkValidFlowSchema()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func mkValidFlowSchema(tweaks ...func(obj *flowcontrol.FlowSchema)) flowcontrol.FlowSchema {
	obj := flowcontrol.FlowSchema{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-obj",
		},
		Spec: flowcontrol.FlowSchemaSpec{
			MatchingPrecedence: 1000,
			PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
				Name: "valid-priority-level",
			},
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
							Verbs:      []string{"get", "list", "watch"},
							APIGroups:  []string{""},
							Resources:  []string{"pods"},
							Namespaces: []string{"production", "staging"},
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

func clearSubjects() func(*flowcontrol.FlowSchema) {
	return func(obj *flowcontrol.FlowSchema) {
		obj.Spec.Rules[0].Subjects = nil
	}
}

func tweakSubjectKind(kind flowcontrol.SubjectKind) func(*flowcontrol.FlowSchema) {
	return func(obj *flowcontrol.FlowSchema) {
		obj.Spec.Rules[0].Subjects[0].Kind = kind
	}
}

func tweakUserSubjectName(name string) func(*flowcontrol.FlowSchema) {
	return func(obj *flowcontrol.FlowSchema) {
		obj.Spec.Rules[0].Subjects[0].User.Name = name
	}
}
