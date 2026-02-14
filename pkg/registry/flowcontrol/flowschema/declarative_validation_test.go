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
	apiVersions := []string{"v1", "v1beta3"}
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
		"invalid priorityLevelConfiguration.name (empty)": {
			input: mkValidFlowSchema(func(obj *flowcontrol.FlowSchema) {
				obj.Spec.PriorityLevelConfiguration.Name = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "priorityLevelConfiguration", "name"), ""),
			},
		},
		"invalid priorityLevelConfiguration.name (format)": {
			input: mkValidFlowSchema(func(obj *flowcontrol.FlowSchema) {
				obj.Spec.PriorityLevelConfiguration.Name = "ASD" // uppercase not allowed in DNS subdomain
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "priorityLevelConfiguration", "name"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func mkValidFlowSchema(tweaks ...func(obj *flowcontrol.FlowSchema)) flowcontrol.FlowSchema {
	obj := flowcontrol.FlowSchema{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-flow-schema",
		},
		Spec: flowcontrol.FlowSchemaSpec{
			MatchingPrecedence: 1000,
			PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
				Name: "exempt",
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}
