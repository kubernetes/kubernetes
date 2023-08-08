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
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	celgo "github.com/google/cel-go/cel"
	celtypes "github.com/google/cel-go/common/types"
	"github.com/stretchr/testify/require"

	"k8s.io/utils/pointer"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
)

type condition struct {
	Expression string
}

func (c *condition) GetExpression() string {
	return c.Expression
}

func (v *condition) ReturnTypes() []*celgo.Type {
	return []*celgo.Type{celgo.BoolType}
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
			c := filterCompiler{compiler: NewCompiler(environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion()))}
			e := c.Compile(tc.validation, OptionalVariableDeclarations{HasParams: false, HasAuthorizer: false}, environment.NewExpressions)
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

	nsObject := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test",
			Labels: map[string]string{
				"env": "test",
				"foo": "demo",
			},
			Annotations: map[string]string{
				"annotation1": "testAnnotation1",
			},
			Finalizers: []string{"f1"},
		},
		Spec: corev1.NamespaceSpec{
			Finalizers: []corev1.FinalizerName{
				corev1.FinalizerKubernetes,
			},
		},
		Status: corev1.NamespaceStatus{
			Phase: corev1.NamespaceActive,
		},
	}

	var nilUnstructured *unstructured.Unstructured
	cases := []struct {
		name             string
		attributes       admission.Attributes
		params           runtime.Object
		validations      []ExpressionAccessor
		results          []EvaluationResult
		hasParamKind     bool
		authorizer       authorizer.Authorizer
		testPerCallLimit uint64
		namespaceObject  *corev1.Namespace
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
		{
			name: "test authorizer allow resource check",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "authorizer.group('').resource('endpoints').check('create').allowed()",
				},
				&condition{
					Expression: "authorizer.group('').resource('endpoints').check('create').errored()",
				},
			},
			attributes: newValidAttribute(&podObject, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
				{
					EvalResult: celtypes.False,
				},
			},
			authorizer: newAuthzAllowMatch(authorizer.AttributesRecord{
				ResourceRequest: true,
				Resource:        "endpoints",
				Verb:            "create",
				APIVersion:      "*",
			}),
		},
		{
			name: "test authorizer allow resource check with all fields",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "authorizer.group('apps').resource('deployments').subresource('status').namespace('test').name('backend').check('create').allowed()",
				},
			},
			attributes: newValidAttribute(&podObject, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
			},
			authorizer: newAuthzAllowMatch(authorizer.AttributesRecord{
				ResourceRequest: true,
				APIGroup:        "apps",
				Resource:        "deployments",
				Subresource:     "status",
				Namespace:       "test",
				Name:            "backend",
				Verb:            "create",
				APIVersion:      "*",
			}),
		},
		{
			name: "test authorizer not allowed resource check one incorrect field",
			validations: []ExpressionAccessor{
				&condition{

					Expression: "authorizer.group('apps').resource('deployments').subresource('status').namespace('test').name('backend').check('create').allowed()",
				},
			},
			attributes: newValidAttribute(&podObject, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.False,
				},
			},
			authorizer: newAuthzAllowMatch(authorizer.AttributesRecord{
				ResourceRequest: true,
				APIGroup:        "apps",
				Resource:        "deployments-xxxx",
				Subresource:     "status",
				Namespace:       "test",
				Name:            "backend",
				Verb:            "create",
				APIVersion:      "*",
			}),
		},
		{
			name: "test authorizer reason",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "authorizer.group('').resource('endpoints').check('create').reason() == 'fake reason'",
				},
			},
			attributes: newValidAttribute(&podObject, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
			},
			authorizer: denyAll,
		},
		{
			name: "test authorizer error",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "authorizer.group('').resource('endpoints').check('create').errored()",
				},
				&condition{
					Expression: "authorizer.group('').resource('endpoints').check('create').error() == 'fake authz error'",
				},
				&condition{
					Expression: "authorizer.group('').resource('endpoints').check('create').allowed()",
				},
			},
			attributes: newValidAttribute(&podObject, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
				{
					EvalResult: celtypes.True,
				},
				{
					EvalResult: celtypes.False,
				},
			},
			authorizer: errorAll,
		},
		{
			name: "test authorizer allow path check",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "authorizer.path('/healthz').check('get').allowed()",
				},
			},
			attributes: newValidAttribute(&podObject, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
			},
			authorizer: newAuthzAllowMatch(authorizer.AttributesRecord{
				Path: "/healthz",
				Verb: "get",
			}),
		},
		{
			name: "test authorizer decision is denied path check",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "authorizer.path('/healthz').check('get').allowed() == false",
				},
			},
			attributes: newValidAttribute(&podObject, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
			},
			authorizer: denyAll,
		},
		{
			name: "test request resource authorizer allow check",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "authorizer.requestResource.check('custom-verb').allowed()",
				},
			},
			attributes: endpointCreateAttributes(),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
			},
			authorizer: newAuthzAllowMatch(authorizer.AttributesRecord{
				ResourceRequest: true,
				APIGroup:        "",
				Resource:        "endpoints",
				Subresource:     "",
				Namespace:       "default",
				Name:            "endpoints1",
				Verb:            "custom-verb",
				APIVersion:      "*",
			}),
		},
		{
			name: "test subresource request resource authorizer allow check",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "authorizer.requestResource.check('custom-verb').allowed()",
				},
			},
			attributes: endpointStatusUpdateAttributes(),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
			},
			authorizer: newAuthzAllowMatch(authorizer.AttributesRecord{
				ResourceRequest: true,
				APIGroup:        "",
				Resource:        "endpoints",
				Subresource:     "status",
				Namespace:       "default",
				Name:            "endpoints1",
				Verb:            "custom-verb",
				APIVersion:      "*",
			}),
		},
		{
			name: "test serviceAccount authorizer allow check",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "authorizer.serviceAccount('default', 'test-serviceaccount').group('').resource('endpoints').namespace('default').name('endpoints1').check('custom-verb').allowed()",
				},
			},
			attributes: endpointCreateAttributes(),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
			},
			authorizer: newAuthzAllowMatch(authorizer.AttributesRecord{
				User: &user.DefaultInfo{
					Name:   "system:serviceaccount:default:test-serviceaccount",
					Groups: []string{"system:serviceaccounts", "system:serviceaccounts:default"},
				},
				ResourceRequest: true,
				APIGroup:        "",
				Resource:        "endpoints",
				Namespace:       "default",
				Name:            "endpoints1",
				Verb:            "custom-verb",
				APIVersion:      "*",
			}),
		},
		{
			name: "test perCallLimit exceed",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "object.subsets.size() < params.spec.testSize",
				},
			},
			attributes: newValidAttribute(nil, false),
			results: []EvaluationResult{
				{
					Error: errors.New(fmt.Sprintf("operation cancelled: actual cost limit exceeded")),
				},
			},
			hasParamKind:     true,
			params:           crdParams,
			testPerCallLimit: 1,
		},
		{
			name: "test namespaceObject",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "namespaceObject.metadata.name == 'test'",
				},
				&condition{
					Expression: "'env' in namespaceObject.metadata.labels && namespaceObject.metadata.labels.env == 'test'",
				},
				&condition{
					Expression: "('fake' in namespaceObject.metadata.labels) && namespaceObject.metadata.labels.fake == 'test'",
				},
				&condition{
					Expression: "namespaceObject.spec.finalizers[0] == 'kubernetes'",
				},
				&condition{
					Expression: "namespaceObject.status.phase == 'Active'",
				},
				&condition{
					Expression: "size(namespaceObject.metadata.managedFields) == 1",
				},
				&condition{
					Expression: "size(namespaceObject.metadata.ownerReferences) == 1",
				},
				&condition{
					Expression: "'env' in namespaceObject.metadata.annotations",
				},
			},
			attributes: newValidAttribute(&podObject, false),
			results: []EvaluationResult{
				{
					EvalResult: celtypes.True,
				},
				{
					EvalResult: celtypes.True,
				},
				{
					EvalResult: celtypes.False,
				},
				{
					EvalResult: celtypes.True,
				},
				{
					EvalResult: celtypes.True,
				},
				{
					Error: errors.New("undefined field 'managedFields'"),
				},
				{
					Error: errors.New("undefined field 'ownerReferences'"),
				},
				{
					EvalResult: celtypes.False,
				},
			},
			hasParamKind:    false,
			namespaceObject: nsObject,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.testPerCallLimit == 0 {
				tc.testPerCallLimit = celconfig.PerCallLimit
			}
			env, err := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion()).Extend(
				environment.VersionedOptions{
					IntroducedVersion: environment.DefaultCompatibilityVersion(),
					ProgramOptions:    []celgo.ProgramOption{celgo.CostLimit(tc.testPerCallLimit)},
				},
			)
			if err != nil {
				t.Fatal(err)
			}
			c := NewFilterCompiler(env)
			f := c.Compile(tc.validations, OptionalVariableDeclarations{HasParams: tc.hasParamKind, HasAuthorizer: tc.authorizer != nil}, environment.NewExpressions)
			if f == nil {
				t.Fatalf("unexpected nil validator")
			}
			validations := tc.validations
			CompilationResults := f.(*filter).compilationResults
			require.Equal(t, len(validations), len(CompilationResults))

			versionedAttr, err := admission.NewVersionedAttributes(tc.attributes, tc.attributes.GetKind(), newObjectInterfacesForTest())
			if err != nil {
				t.Fatalf("unexpected error on conversion: %v", err)
			}

			optionalVars := OptionalVariableBindings{VersionedParams: tc.params, Authorizer: tc.authorizer}
			ctx := context.TODO()
			evalResults, _, err := f.ForInput(ctx, versionedAttr, CreateAdmissionRequest(versionedAttr.Attributes, metav1.GroupVersionResource(versionedAttr.GetResource()), metav1.GroupVersionKind(versionedAttr.VersionedKind)), optionalVars, CreateNamespaceObject(tc.namespaceObject), celconfig.RuntimeCELCostBudget)
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

func TestRuntimeCELCostBudget(t *testing.T) {
	configMapParams := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Data: map[string]string{
			"fakeString": "fake",
		},
	}

	cases := []struct {
		name                     string
		attributes               admission.Attributes
		params                   runtime.Object
		validations              []ExpressionAccessor
		hasParamKind             bool
		authorizer               authorizer.Authorizer
		testRuntimeCELCostBudget int64
		exceedBudget             bool
		expectRemainingBudget    *int64
	}{
		{
			name: "expression exceed RuntimeCELCostBudget at fist expression",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "has(object.subsets) && object.subsets.size() < 2",
				},
				&condition{
					Expression: "has(object.subsets)",
				},
			},
			attributes:               newValidAttribute(nil, false),
			hasParamKind:             false,
			testRuntimeCELCostBudget: 1,
			exceedBudget:             true,
		},
		{
			name: "expression exceed RuntimeCELCostBudget at last expression",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "has(object.subsets) && object.subsets.size() < 2",
				},
				&condition{
					Expression: "object.subsets.size() > 2",
				},
			},
			attributes:               newValidAttribute(nil, false),
			hasParamKind:             true,
			params:                   configMapParams,
			testRuntimeCELCostBudget: 5,
			exceedBudget:             true,
		},
		{
			name: "test RuntimeCELCostBudge is not exceed",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "oldObject != null",
				},
				&condition{
					Expression: "object.subsets.size() > 2",
				},
			},
			attributes:               newValidAttribute(nil, false),
			hasParamKind:             true,
			params:                   configMapParams,
			exceedBudget:             false,
			testRuntimeCELCostBudget: 10,
			expectRemainingBudget:    pointer.Int64(4), // 10 - 6
		},
		{
			name: "test RuntimeCELCostBudge exactly covers",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "oldObject != null",
				},
				&condition{
					Expression: "object.subsets.size() > 2",
				},
			},
			attributes:               newValidAttribute(nil, false),
			hasParamKind:             true,
			params:                   configMapParams,
			exceedBudget:             false,
			testRuntimeCELCostBudget: 6,
			expectRemainingBudget:    pointer.Int64(0),
		},
		{
			name: "test RuntimeCELCostBudge exactly covers then constant",
			validations: []ExpressionAccessor{
				&condition{
					Expression: "oldObject != null",
				},
				&condition{
					Expression: "object.subsets.size() > 2",
				},
				&condition{
					Expression: "true", // zero cost
				},
			},
			attributes:               newValidAttribute(nil, false),
			hasParamKind:             true,
			params:                   configMapParams,
			exceedBudget:             false,
			testRuntimeCELCostBudget: 6,
			expectRemainingBudget:    pointer.Int64(0),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := filterCompiler{compiler: NewCompiler(environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion()))}
			f := c.Compile(tc.validations, OptionalVariableDeclarations{HasParams: tc.hasParamKind, HasAuthorizer: false}, environment.NewExpressions)
			if f == nil {
				t.Fatalf("unexpected nil validator")
			}
			validations := tc.validations
			CompilationResults := f.(*filter).compilationResults
			require.Equal(t, len(validations), len(CompilationResults))

			versionedAttr, err := admission.NewVersionedAttributes(tc.attributes, tc.attributes.GetKind(), newObjectInterfacesForTest())
			if err != nil {
				t.Fatalf("unexpected error on conversion: %v", err)
			}

			if tc.testRuntimeCELCostBudget == 0 {
				tc.testRuntimeCELCostBudget = celconfig.RuntimeCELCostBudget
			}
			optionalVars := OptionalVariableBindings{VersionedParams: tc.params, Authorizer: tc.authorizer}
			ctx := context.TODO()
			evalResults, remaining, err := f.ForInput(ctx, versionedAttr, CreateAdmissionRequest(versionedAttr.Attributes, metav1.GroupVersionResource(versionedAttr.GetResource()), metav1.GroupVersionKind(versionedAttr.VersionedKind)), optionalVars, nil, tc.testRuntimeCELCostBudget)
			if tc.exceedBudget && err == nil {
				t.Errorf("Expected RuntimeCELCostBudge to be exceeded but got nil")
			}
			if tc.exceedBudget && !strings.Contains(err.Error(), "validation failed due to running out of cost budget, no further validation rules will be run") {
				t.Errorf("Expected RuntimeCELCostBudge exceeded error but got: %v", err)
			}
			if err != nil && remaining != -1 {
				t.Errorf("expected -1 remaining when error, but got %d", remaining)
			}
			if err != nil && !tc.exceedBudget {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.exceedBudget && len(evalResults) != 0 {
				t.Fatalf("unexpected result returned: %v", evalResults)
			}
			if tc.expectRemainingBudget != nil && *tc.expectRemainingBudget != remaining {
				t.Errorf("wrong remaining budget, expect %d, but got %d", *tc.expectRemainingBudget, remaining)
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

var denyAll = fakeAuthorizer{defaultResult: authorizerResult{decision: authorizer.DecisionDeny, reason: "fake reason", err: nil}}
var errorAll = fakeAuthorizer{defaultResult: authorizerResult{decision: authorizer.DecisionNoOpinion, reason: "", err: fmt.Errorf("fake authz error")}}

func newAuthzAllowMatch(match authorizer.AttributesRecord) fakeAuthorizer {
	return fakeAuthorizer{
		match: &authorizerMatch{
			match:            match,
			authorizerResult: authorizerResult{decision: authorizer.DecisionAllow, reason: "", err: nil},
		},
		defaultResult: authorizerResult{decision: authorizer.DecisionDeny, reason: "", err: nil},
	}
}

type fakeAuthorizer struct {
	match         *authorizerMatch
	defaultResult authorizerResult
}

type authorizerResult struct {
	decision authorizer.Decision
	reason   string
	err      error
}

type authorizerMatch struct {
	authorizerResult
	match authorizer.AttributesRecord
}

func (f fakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	if f.match != nil {
		other, ok := a.(*authorizer.AttributesRecord)
		if !ok {
			panic(fmt.Sprintf("unsupported type: %T", a))
		}
		if reflect.DeepEqual(f.match.match, *other) {
			return f.match.decision, f.match.reason, f.match.err
		}
	}
	return f.defaultResult.decision, f.defaultResult.reason, f.defaultResult.err
}

func endpointCreateAttributes() admission.Attributes {
	name := "endpoints1"
	namespace := "default"
	var object, oldObject runtime.Object
	object = &corev1.Endpoints{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Endpoints",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Subsets: []corev1.EndpointSubset{
			{
				Addresses: []corev1.EndpointAddress{{IP: "127.0.0.0"}},
			},
		},
	}
	gvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Endpoints"}
	gvr := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "endpoints"}
	return admission.NewAttributesRecord(object, oldObject, gvk, namespace, name, gvr, "", admission.Create, &metav1.CreateOptions{}, false, nil)
}

func endpointStatusUpdateAttributes() admission.Attributes {
	attrs := endpointCreateAttributes()
	return admission.NewAttributesRecord(
		attrs.GetObject(), attrs.GetObject(), attrs.GetKind(), attrs.GetNamespace(), attrs.GetName(),
		attrs.GetResource(), "status", admission.Update, &metav1.UpdateOptions{}, false, nil)
}
