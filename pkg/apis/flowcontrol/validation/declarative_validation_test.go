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

package validation

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
)

func TestFlowSchemaDeclarativeValidation(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "flowcontrol.apiserver.k8s.io",
		APIVersion: "v1",
	})

	testCases := map[string]struct {
		input        flowcontrol.FlowSchema
		expectedErrs field.ErrorList
	}{
		"valid: PriorityLevelConfiguration.Name is set": {
			input: mkFlowSchema(),
		},
		"invalid: PriorityLevelConfiguration.Name is empty": {
			input: mkFlowSchema(clearPLCName),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "priorityLevelConfiguration", "name"), ""),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			errs := ValidateFlowSchema(&tc.input)
			tester := field.ErrorMatcher{}.ByType().ByField()
			tester.Test(t, tc.expectedErrs, errs)

			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, func(_ context.Context, obj runtime.Object) field.ErrorList {
				return ValidateFlowSchema(obj.(*flowcontrol.FlowSchema))
			}, tc.expectedErrs)
		})
	}
}

func mkFlowSchema(tweaks ...func(*flowcontrol.FlowSchema)) flowcontrol.FlowSchema {
	fs := flowcontrol.FlowSchema{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-flowschema",
		},
		Spec: flowcontrol.FlowSchemaSpec{
			MatchingPrecedence: 100,
			PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
				Name: "test-priority-level",
			},
			Rules: []flowcontrol.PolicyRulesWithSubjects{
				{
					Subjects: []flowcontrol.Subject{
						{
							Kind: flowcontrol.SubjectKindUser,
							User: &flowcontrol.UserSubject{Name: "test-user"},
						},
					},
					ResourceRules: []flowcontrol.ResourcePolicyRule{
						{
							Verbs:        []string{"get"},
							APIGroups:    []string{""},
							Resources:    []string{"pods"},
							ClusterScope: true,
							Namespaces:   []string{flowcontrol.NamespaceEvery},
						},
					},
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&fs)
	}
	return fs
}

func clearPLCName(fs *flowcontrol.FlowSchema) {
	fs.Spec.PriorityLevelConfiguration.Name = ""
}
