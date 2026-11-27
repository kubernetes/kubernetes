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
		"empty priorityLevelConfiguration.name": {
			input: mkValidFlowSchema(tweakPLC("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "priorityLevelConfiguration", "name"), "").MarkAlpha(),
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
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "flowcontrol.apiserver.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "flowschemas",
		Name:              "valid-obj",
		IsResourceRequest: true,
		Verb:              "update",
	})

	testCases := map[string]struct {
		oldObj       flowcontrol.FlowSchema
		updateObj    flowcontrol.FlowSchema
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidFlowSchema(withResourceVersion("1")),
			updateObj: mkValidFlowSchema(withResourceVersion("1"), tweakPLC("other-priority-level")),
		},
		"clear priorityLevelConfiguration.name": {
			oldObj:    mkValidFlowSchema(withResourceVersion("1")),
			updateObj: mkValidFlowSchema(withResourceVersion("1"), tweakPLC("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "priorityLevelConfiguration", "name"), "").MarkAlpha(),
			},
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, registry.Strategy, tc.expectedErrs)
		})
	}

	updateObj := mkValidFlowSchema()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func mkValidFlowSchema(tweaks ...func(*flowcontrol.FlowSchema)) flowcontrol.FlowSchema {
	fs := flowcontrol.FlowSchema{
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
	for _, tweak := range tweaks {
		tweak(&fs)
	}
	return fs
}

func tweakPLC(name string) func(*flowcontrol.FlowSchema) {
	return func(fs *flowcontrol.FlowSchema) {
		fs.Spec.PriorityLevelConfiguration.Name = name
	}
}

func withResourceVersion(rv string) func(*flowcontrol.FlowSchema) {
	return func(fs *flowcontrol.FlowSchema) {
		fs.ResourceVersion = rv
	}
}
