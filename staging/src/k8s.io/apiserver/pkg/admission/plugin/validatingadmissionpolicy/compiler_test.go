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

package validatingadmissionpolicy

import (
	"strings"
	"testing"
)

func TestCompileValidatingPolicyExpression(t *testing.T) {
	cases := []struct {
		name             string
		expressions      []string
		hasParams        bool
		errorExpressions map[string]string
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
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for _, expr := range tc.expressions {
				result := CompileValidatingPolicyExpression(expr, tc.hasParams)
				if result.Error != nil {
					t.Errorf("Unexpected error: %v", result.Error)
				}
			}
			for expr, expectErr := range tc.errorExpressions {
				result := CompileValidatingPolicyExpression(expr, tc.hasParams)
				if result.Error == nil {
					t.Errorf("Expected expression '%s' to contain '%v' but got no error", expr, expectErr)
					continue
				}
				if !strings.Contains(result.Error.Error(), expectErr) {
					t.Errorf("Expected validation '%s' error to contain '%v' but got: %v", expr, expectErr, result.Error)
				}
				continue
			}
		})
	}
}
