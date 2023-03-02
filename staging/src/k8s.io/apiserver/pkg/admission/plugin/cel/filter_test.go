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

package cel

import (
	"errors"
	"strings"
	"testing"

	celtypes "github.com/google/cel-go/common/types"
	"github.com/stretchr/testify/require"
	apiservercel "k8s.io/apiserver/pkg/cel"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
)

type condition struct {
	Expression string
}

func (c *condition) GetExpression() string {
	return c.Expression
}

func TestCompile(t *testing.T) {
	cases := []struct {
		name             string
		validation       []ExpressionAccessor
		errorExpressions map[string]string
	}{
		{
			name: "invalid syntax",
			validation: []ExpressionAccessor{
				&condition{
					Expression: "1 < 'asdf'",
				},
				&condition{
					Expression: "1 < 2",
				},
			},
			errorExpressions: map[string]string{
				"1 < 'asdf'": "found no matching overload for '_<_' applied to '(int, string)",
			},
		},
		{
			name: "valid syntax",
			validation: []ExpressionAccessor{
				&condition{
					Expression: "1 < 2",
				},
				&condition{
					Expression: "object.spec.string.matches('[0-9]+')",
				},
				&condition{
					Expression: "request.kind.group == 'example.com' && request.kind.version == 'v1' && request.kind.kind == 'Fake'",
				},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var c filterCompiler
			e := c.Compile(tc.validation, false)
			if e == nil {
				t.Fatalf("unexpected nil validator")
			}
			validations := tc.validation
			CompilationResults := e.(*filter).compilationResults
			require.Equal(t, len(validations), len(CompilationResults))

			meets := make([]bool, len(validations))
			for expr, expectErr := range tc.errorExpressions {
				for i, result := range CompilationResults {
					if validations[i].GetExpression() == expr {
						if result.Error == nil {
							t.Errorf("Expect expression '%s' to contain error '%v' but got no error", expr, expectErr)
						} else if !strings.Contains(result.Error.Error(), expectErr) {
							t.Errorf("Expected validations '%s' error to contain '%v' but got: %v", expr, expectErr, result.Error)
						}
						meets[i] = true
					}
				}
			}
			for i, meet := range meets {
				if !meet && CompilationResults[i].Error != nil {
					t.Errorf("Unexpected err '%v' for expression '%s'", CompilationResults[i].Error, validations[i].GetExpression())
				}
			}
		})
	}
}

func TestFilter(t *testing.T) {
	configMapParams := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Data: map[string]string{
			"fakeString": "fake",
		},
	}
	crdParams := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"spec": map[string]interface{}{
				"testSize": 10,
			},
		},
	}
	podObject := corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: corev1.PodSpec{
			NodeName: "testnode",
		},
	}

	var nilUnstructured *unstructured.Unstructured
	cases := []struct {
		name         string
		attributes   admission.Attributes
		params       runtime.Object
		validations  []ExpressionAccessor
		results      []EvaluationResult
		hasParamKind bool
	}{
		{
			name: "valid syntax for object",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "has(object.subsets) && object.subsets.size() < 2",
				},
			},
			attributes: newValidAttribute(nil, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
			},
			hasParamKind: false,
		},
		{
			name: "valid syntax for metadata",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "object.metadata.name == 'endpoints1'",
				},
			},
			attributes: newValidAttribute(nil, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
			},
			hasParamKind: false,
		},
		{
			name: "valid syntax for oldObject",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "oldObject == null",
				},
				&condition{
					Expression: "object != null",
				},
			},
			attributes: newValidAttribute(nil, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
				{
					EvalResult: celtypes.True,
				},
			},
			hasParamKind: false,
		},
		{
			name: "valid syntax for request",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "request.operation == 'CREATE'",
				},
			},
			attributes: newValidAttribute(nil, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
			},
			hasParamKind: false,
		},
		{
			name: "valid syntax for configMap",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "request.namespace != params.data.fakeString",
				},
			},
			attributes: newValidAttribute(nil, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
			},
			hasParamKind: true,
			params:       configMapParams,
		},
		{
			name: "test failure",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "object.subsets.size() > 2",
				},
			},
			attributes: newValidAttribute(nil, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.False,
				},
			},
			hasParamKind: true,
			params: &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Data: map[string]string{
					"fakeString": "fake",
				},
			},
		},
		{
			name: "test failure with multiple validations",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "has(object.subsets)",
				},
				&condition{
					Expression: "object.subsets.size() > 2",
				},
			},
			attributes: newValidAttribute(nil, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
				{
					EvalResult: celtypes.False,
				},
			},
			hasParamKind: true,
			params:       configMapParams,
		},
		{
			name: "test failure policy with multiple failed validations",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "oldObject != null",
				},
				&condition{
					Expression: "object.subsets.size() > 2",
				},
			},
			attributes: newValidAttribute(nil, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.False,
				},
				{
					EvalResult: celtypes.False,
				},
			},
			hasParamKind: true,
			params:       configMapParams,
		},
		{
			name: "test Object null in delete",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "oldObject != null",
				},
				&condition{
					Expression: "object == null",
				},
			},
			attributes: newValidAttribute(nil, true),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
				{
					EvalResult: celtypes.True,
				},
			},
			hasParamKind: true,
			params:       configMapParams,
		},
		{
			name: "test runtime error",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "oldObject.x == 100",
				},
			},
			attributes: newValidAttribute(nil, true),
			results: []EvaluationResult{
				{
					Error: errors.New("expression 'oldObject.x == 100' resulted in error"),
				},
			},
			hasParamKind: true,
			params:       configMapParams,
		},
		{
			name: "test against crd param",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "object.subsets.size() < params.spec.testSize",
				},
			},
			attributes: newValidAttribute(nil, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
			},
			hasParamKind: true,
			params:       crdParams,
		},
		{
			name: "test compile failure",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "fail to compile test",
				},
				&condition{
					Expression: "object.subsets.size() > params.spec.testSize",
				},
			},
			attributes: newValidAttribute(nil, false),
			results: []EvaluationResult{
				{
					Error: errors.New("compilation error"),
				},
				{
					EvalResult: celtypes.False,
				},
			},
			hasParamKind: true,
			params:       crdParams,
		},
		{
			name: "test pod",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "object.spec.nodeName == 'testnode'",
				},
			},
			attributes: newValidAttribute(&podObject, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
			},
			hasParamKind: true,
			params:       crdParams,
		},
		{
			name: "test deny paramKind without paramRef",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "params != null",
				},
			},
			attributes: newValidAttribute(&podObject, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.False,
				},
			},
			hasParamKind: true,
		},
		{
			name: "test allow paramKind without paramRef",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "params == null",
				},
			},
			attributes: newValidAttribute(&podObject, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
			},
			hasParamKind: true,
			params:       runtime.Object(nilUnstructured),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := filterCompiler{}
			f := c.Compile(tc.validations, tc.hasParamKind)
			if f == nil {
				t.Fatalf("unexpected nil validator")
			}
			validations := tc.validations
			CompilationResults := f.(*filter).compilationResults
			require.Equal(t, len(validations), len(CompilationResults))

			versionedAttr, err := generic.NewVersionedAttributes(tc.attributes, tc.attributes.GetKind(), newObjectInterfacesForTest())
			if err != nil {
				t.Fatalf("unexpected error on conversion: %v", err)
			}

			evalResults, err := f.ForInput(versionedAttr, tc.params, CreateAdmissionRequest(versionedAttr.Attributes))
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			require.Equal(t, len(evalResults), len(tc.results))
			for i, result := range tc.results {
				if result.EvalResult != evalResults[i].EvalResult {
					t.Errorf("Expected result '%v' but got '%v'", result.EvalResult, evalResults[i].EvalResult)
				}
				if result.Error != nil && !strings.Contains(evalResults[i].Error.Error(), result.Error.Error()) {
					t.Errorf("Expected result '%v' but got '%v'", result.Error, evalResults[i].Error)
				}
			}
		})
	}
}

// newObjectInterfacesForTest returns an ObjectInterfaces appropriate for test cases in this file.
func newObjectInterfacesForTest() admission.ObjectInterfaces {
	scheme := runtime.NewScheme()
	corev1.AddToScheme(scheme)
	return admission.NewObjectInterfacesFromScheme(scheme)
}

func newValidAttribute(object runtime.Object, isDelete bool) admission.Attributes {
	var oldObject runtime.Object
	if !isDelete {
		if object == nil {
			object = &corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name: "endpoints1",
				},
				Subsets: []corev1.EndpointSubset{
					{
						Addresses: []corev1.EndpointAddress{{IP: "127.0.0.0"}},
					},
				},
			}
		}
	} else {
		object = nil
		oldObject = &corev1.Endpoints{
			Subsets: []corev1.EndpointSubset{
				{
					Addresses: []corev1.EndpointAddress{{IP: "127.0.0.0"}},
				},
			},
		}
	}
	return admission.NewAttributesRecord(object, oldObject, schema.GroupVersionKind{}, "default", "foo", schema.GroupVersionResource{}, "", admission.Create, &metav1.CreateOptions{}, false, nil)

}

func TestCompilationErrors(t *testing.T) {
	cases := []struct {
		name     string
		results  []CompilationResult
		expected []error
	}{
		{
			name:     "no errors, empty list",
			results:  []CompilationResult{},
			expected: []error{},
		},
		{
			name: "no errors, several results",
			results: []CompilationResult{
				{}, {}, {},
			},
			expected: []error{},
		},
		{
			name: "all errors",
			results: []CompilationResult{
				{
					Error: &apiservercel.Error{
						Detail: "error1",
					},
				},
				{
					Error: &apiservercel.Error{
						Detail: "error2",
					},
				},
				{
					Error: &apiservercel.Error{
						Detail: "error3",
					},
				},
			},
			expected: []error{
				errors.New("error1"),
				errors.New("error2"),
				errors.New("error3"),
			},
		},
		{
			name: "mixed errors and non errors",
			results: []CompilationResult{
				{},
				{
					Error: &apiservercel.Error{
						Detail: "error1",
					},
				},
				{},
				{
					Error: &apiservercel.Error{
						Detail: "error2",
					},
				},
				{},
				{},
				{
					Error: &apiservercel.Error{
						Detail: "error3",
					},
				},
				{},
			},
			expected: []error{
				errors.New("error1"),
				errors.New("error2"),
				errors.New("error3"),
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e := filter{
				compilationResults: tc.results,
			}
			compilationErrors := e.CompilationErrors()
			if compilationErrors == nil {
				t.Fatalf("unexpected nil value returned")
			}
			require.Equal(t, len(compilationErrors), len(tc.expected))

			for i, expectedError := range tc.expected {
				if expectedError.Error() != compilationErrors[i].Error() {
					t.Errorf("Expected error '%v' but got '%v'", expectedError.Error(), compilationErrors[i].Error())
				}
			}
		})
	}
}
