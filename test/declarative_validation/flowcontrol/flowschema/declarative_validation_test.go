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
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(),
		&genericapirequest.RequestInfo{
			APIPrefix:         "apis",
			APIGroup:          "flowcontrol.apiserver.k8s.io",
			APIVersion:        apiVersion,
			Resource:          "flowschemas",
			IsResourceRequest: true,
			Verb:              "create",
		})

	obj := mkValidFlowSchema()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())

	for k, tc := range userSubjectTestCases() {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, registry.Strategy, tc.expectedErrs)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
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

	for k, tc := range userSubjectTestCases() {
		t.Run(k, func(t *testing.T) {
			old := mkValidFlowSchemaForUserSubject("test")
			old.ResourceVersion = "1"
			update := tc.input.DeepCopy()
			update.ResourceVersion = "1"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, update, &old, registry.Strategy, tc.expectedErrs)
		})
	}
}

type userSubjectTestCase struct {
	input        flowcontrol.FlowSchema
	expectedErrs field.ErrorList
}

func userSubjectTestCases() map[string]userSubjectTestCase {
	userPath := field.NewPath("spec", "rules").Index(0).Child("subjects").Index(0).Child("user")

	noUser := mkValidFlowSchemaForUserSubject("test")
	noUser.Spec.Rules[0].Subjects[0].User = nil

	return map[string]userSubjectTestCase{
		"valid user subject": {
			input: mkValidFlowSchemaForUserSubject("test"),
		},
		"user subject with empty name": {
			input: mkValidFlowSchemaForUserSubject(""),
			expectedErrs: field.ErrorList{
				field.Required(userPath.Child("name"), "").MarkAlpha(),
			},
		},
		"kind User without user": {
			input: noUser,
			expectedErrs: field.ErrorList{
				field.Required(userPath, "").MarkAlpha(),
			},
		},
	}
}

func mkValidFlowSchemaForUserSubject(userName string) flowcontrol.FlowSchema {
	return flowcontrol.FlowSchema{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-obj",
		},
		Spec: flowcontrol.FlowSchemaSpec{
			MatchingPrecedence: 1000,
			PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
				Name: "valid-priority-level",
			},
			Rules: []flowcontrol.PolicyRulesWithSubjects{{
				Subjects: []flowcontrol.Subject{{
					Kind: flowcontrol.SubjectKindUser,
					User: &flowcontrol.UserSubject{Name: userName},
				}},
				// At least one of resourceRules/nonResourceRules must be non-empty.
				NonResourceRules: []flowcontrol.NonResourcePolicyRule{{
					Verbs:           []string{flowcontrol.VerbAll},
					NonResourceURLs: []string{flowcontrol.NonResourceAll},
				}},
			}},
		},
	}
}

func mkValidFlowSchema() flowcontrol.FlowSchema {
	return flowcontrol.FlowSchema{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-obj",
		},
		Spec: flowcontrol.FlowSchemaSpec{
			MatchingPrecedence: 1000,
			PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
				Name: "valid-priority-level",
			},
		},
	}
}
