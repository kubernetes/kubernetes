/*
Copyright 2022 The Kubernetes Authors.

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
	"math/rand"
	"strings"
	"testing"

	celgo "github.com/google/cel-go/cel"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/cel/library"
)

func TestCompileValidatingPolicyExpression(t *testing.T) {
	cases := []struct {
		name             string
		expressions      []string
		hasParams        bool
		hasAuthorizer    bool
		errorExpressions map[string]string
		envType          environment.Type
	}{
		{
			name: "invalid syntax",
			errorExpressions: map[string]string{
				"1 < 'asdf'":          "found no matching overload for '_<_' applied to '(int, string)",
				"'asdf'.contains('x'": "Syntax error: missing ')' at",
			},
		},
		{
			name:        "with params",
			expressions: []string{"object.foo < params.x"},
			hasParams:   true,
		},
		{
			name:        "namespaceObject",
			expressions: []string{"namespaceObject.metadata.name.startsWith('test')"},
			hasParams:   true,
		},
		{
			name:             "without params",
			errorExpressions: map[string]string{"object.foo < params.x": "undeclared reference to 'params'"},
			hasParams:        false,
		},
		{
			name:        "oldObject comparison",
			expressions: []string{"object.foo == oldObject.foo"},
		},
		{
			name: "object null checks",
			// since object and oldObject are CEL variable, has() cannot be used (it works only on fields),
			// so we always populate it to allow for a null check in the case of CREATE, where oldObject is
			// null, and DELETE, where object is null.
			expressions: []string{"object == null || oldObject == null || object.foo == oldObject.foo"},
		},
		{
			name:             "invalid root var",
			errorExpressions: map[string]string{"object.foo < invalid.x": "undeclared reference to 'invalid'"},
			hasParams:        false,
		},
		{
			name: "function library",
			// sanity check that functions of the various libraries are available
			expressions: []string{
				"object.spec.string.matches('[0-9]+')",                      // strings extension lib
				"object.spec.string.findAll('[0-9]+').size() > 0",           // kubernetes string lib
				"object.spec.list.isSorted()",                               // kubernetes list lib
				"url(object.spec.endpoint).getHostname() in ['ok1', 'ok2']", // kubernetes url lib
			},
		},
		{
			name: "valid request",
			expressions: []string{
				"request.kind.group == 'example.com' && request.kind.version == 'v1' && request.kind.kind == 'Fake'",
				"request.resource.group == 'example.com' && request.resource.version == 'v1' && request.resource.resource == 'fake' && request.subResource == 'scale'",
				"request.requestKind.group == 'example.com' && request.requestKind.version == 'v1' && request.requestKind.kind == 'Fake'",
				"request.requestResource.group == 'example.com' && request.requestResource.version == 'v1' && request.requestResource.resource == 'fake' && request.requestSubResource == 'scale'",
				"request.name == 'fake-name'",
				"request.namespace == 'fake-namespace'",
				"request.operation == 'CREATE'",
				"request.userInfo.username == 'admin'",
				"request.userInfo.uid == '014fbff9a07c'",
				"request.userInfo.groups == ['system:authenticated', 'my-admin-group']",
				"request.userInfo.extra == {'some-key': ['some-value1', 'some-value2']}",
				"request.dryRun == false",
				"request.options == {'whatever': 'you want'}",
			},
		},
		{
			name: "invalid request",
			errorExpressions: map[string]string{
				"request.foo1 == 'nope'":                 "undefined field 'foo1'",
				"request.resource.foo2 == 'nope'":        "undefined field 'foo2'",
				"request.requestKind.foo3 == 'nope'":     "undefined field 'foo3'",
				"request.requestResource.foo4 == 'nope'": "undefined field 'foo4'",
				"request.userInfo.foo5 == 'nope'":        "undefined field 'foo5'",
			},
		},
		{
			name:          "with authorizer",
			hasAuthorizer: true,
			expressions: []string{
				"authorizer.group('') != null",
			},
		},
		{
			name: "without authorizer",
			errorExpressions: map[string]string{
				"authorizer.group('') != null": "undeclared reference to 'authorizer'",
			},
		},
		{
			name: "compile with storage environment should recognize functions available only in the storage environment",
			expressions: []string{
				"test() == true",
			},
			envType: environment.StoredExpressions,
		},
		{
			name: "compile with supported environment should not recognize functions available only in the storage environment",
			errorExpressions: map[string]string{
				"test() == true": "undeclared reference to 'test'",
			},
			envType: environment.NewExpressions,
		},
		{
			name: "valid namespaceObject",
			expressions: []string{
				"namespaceObject.metadata != null",
				"namespaceObject.metadata.name == 'test'",
				"namespaceObject.metadata.generateName == 'test'",
				"namespaceObject.metadata.namespace == 'testns'",
				"'test' in namespaceObject.metadata.labels",
				"'test' in namespaceObject.metadata.annotations",
				"namespaceObject.metadata.UID == '12345'",
				"type(namespaceObject.metadata.creationTimestamp) == google.protobuf.Timestamp",
				"type(namespaceObject.metadata.deletionTimestamp) == google.protobuf.Timestamp",
				"namespaceObject.metadata.deletionGracePeriodSeconds == 5",
				"namespaceObject.metadata.generation == 2",
				"namespaceObject.metadata.resourceVersion == 'v1'",
				"namespaceObject.metadata.finalizers[0] == 'testEnv'",
				"namespaceObject.spec.finalizers[0] == 'testEnv'",
				"namespaceObject.status.phase == 'Active'",
				"namespaceObject.status.conditions[0].status == 'True'",
				"namespaceObject.status.conditions[0].type == 'NamespaceDeletionDiscoveryFailure'",
				"type(namespaceObject.status.conditions[0].lastTransitionTime) == google.protobuf.Timestamp",
				"namespaceObject.status.conditions[0].message == 'Unknow'",
				"namespaceObject.status.conditions[0].reason == 'Invalid'",
			},
		},
		{
			name: "invalid namespaceObject",
			errorExpressions: map[string]string{
				"namespaceObject.foo1 == 'nope'":                      "undefined field 'foo1'",
				"namespaceObject.metadata.foo2 == 'nope'":             "undefined field 'foo2'",
				"namespaceObject.spec.foo3 == 'nope'":                 "undefined field 'foo3'",
				"namespaceObject.status.foo4 == 'nope'":               "undefined field 'foo4'",
				"namespaceObject.status.conditions[0].foo5 == 'nope'": "undefined field 'foo5'",
			},
		},
	}

	// Include the test library, which includes the test() function in the storage environment during test
	base := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion())
	extended, err := base.Extend(environment.VersionedOptions{
		IntroducedVersion: version.MajorMinor(1, 999),
		EnvOptions:        []celgo.EnvOption{library.Test()},
	})
	if err != nil {
		t.Fatal(err)
	}
	compiler := NewCompiler(extended)

	for _, tc := range cases {
		envType := tc.envType
		if envType == "" {
			envType = environment.NewExpressions
		}
		t.Run(tc.name, func(t *testing.T) {
			for _, expr := range tc.expressions {
				t.Run(expr, func(t *testing.T) {
					t.Run("expression", func(t *testing.T) {
						options := OptionalVariableDeclarations{HasParams: tc.hasParams, HasAuthorizer: tc.hasAuthorizer}

						result := compiler.CompileCELExpression(&fakeValidationCondition{
							Expression: expr,
						}, options, envType)
						if result.Error != nil {
							t.Errorf("Unexpected error: %v", result.Error)
						}
					})
					t.Run("auditAnnotation.valueExpression", func(t *testing.T) {
						// Test audit annotation compilation by casting the result to a string
						options := OptionalVariableDeclarations{HasParams: tc.hasParams, HasAuthorizer: tc.hasAuthorizer}
						result := compiler.CompileCELExpression(&fakeAuditAnnotationCondition{
							ValueExpression: "string(" + expr + ")",
						}, options, envType)
						if result.Error != nil {
							t.Errorf("Unexpected error: %v", result.Error)
						}
					})
				})
			}
			for expr, expectErr := range tc.errorExpressions {
				t.Run(expr, func(t *testing.T) {
					t.Run("expression", func(t *testing.T) {
						options := OptionalVariableDeclarations{HasParams: tc.hasParams, HasAuthorizer: tc.hasAuthorizer}
						result := compiler.CompileCELExpression(&fakeValidationCondition{
							Expression: expr,
						}, options, envType)
						if result.Error == nil {
							t.Errorf("Expected expression '%s' to contain '%v' but got no error", expr, expectErr)
							return
						}
						if !strings.Contains(result.Error.Error(), expectErr) {
							t.Errorf("Expected validation '%s' error to contain '%v' but got: %v", expr, expectErr, result.Error)
						}
					})
					t.Run("auditAnnotation.valueExpression", func(t *testing.T) {
						// Test audit annotation compilation by casting the result to a string
						options := OptionalVariableDeclarations{HasParams: tc.hasParams, HasAuthorizer: tc.hasAuthorizer}
						result := compiler.CompileCELExpression(&fakeAuditAnnotationCondition{
							ValueExpression: "string(" + expr + ")",
						}, options, envType)
						if result.Error == nil {
							t.Errorf("Expected expression '%s' to contain '%v' but got no error", expr, expectErr)
							return
						}
						if !strings.Contains(result.Error.Error(), expectErr) {
							t.Errorf("Expected validation '%s' error to contain '%v' but got: %v", expr, expectErr, result.Error)
						}
					})
				})
			}
		})
	}
}

func BenchmarkCompile(b *testing.B) {
	compiler := NewCompiler(environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion()))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		options := OptionalVariableDeclarations{HasParams: rand.Int()%2 == 0, HasAuthorizer: rand.Int()%2 == 0}

		result := compiler.CompileCELExpression(&fakeValidationCondition{
			Expression: "object.foo < object.bar",
		}, options, environment.StoredExpressions)
		if result.Error != nil {
			b.Fatal(result.Error)
		}
	}
}

type fakeValidationCondition struct {
	Expression string
}

func (v *fakeValidationCondition) GetExpression() string {
	return v.Expression
}

func (v *fakeValidationCondition) ReturnTypes() []*celgo.Type {
	return []*celgo.Type{celgo.BoolType}
}

type fakeAuditAnnotationCondition struct {
	ValueExpression string
}

func (v *fakeAuditAnnotationCondition) GetExpression() string {
	return v.ValueExpression
}

func (v *fakeAuditAnnotationCondition) ReturnTypes() []*celgo.Type {
	return []*celgo.Type{celgo.StringType, celgo.NullType}
}
