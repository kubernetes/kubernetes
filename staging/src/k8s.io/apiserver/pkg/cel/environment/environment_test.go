/*
Copyright 2023 The Kubernetes Authors.

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

package environment

import (
	"context"
	"fmt"
	"testing"

	"github.com/google/cel-go/cel"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/library"
)

type envTypeAndVersion struct {
	version *version.Version
	envType Type
}

func TestBaseEnvironment(t *testing.T) {
	widgetsType := apiservercel.NewObjectType("Widget",
		map[string]*apiservercel.DeclField{
			"x": {
				Name: "x",
				Type: apiservercel.StringType,
			},
		})

	cases := []struct {
		name                    string
		typeVersionCombinations []envTypeAndVersion
		validExpressions        []string
		invalidExpressions      []string
		activation              any
		opts                    []VersionedOptions
	}{
		{
			name: "core settings enabled",
			typeVersionCombinations: []envTypeAndVersion{
				{version.MajorMinor(1, 23), NewExpressions},
				{version.MajorMinor(1, 23), StoredExpressions},
			},
			validExpressions: []string{
				"[1, 2, 3].indexOf(2) == 1",      // lists
				"'abc'.contains('bc')",           //strings
				"isURL('http://example.com')",    // urls
				"'a 1 b 2'.find('[0-9]') == '1'", // regex
			},
		},
		{
			name: "authz disabled",
			typeVersionCombinations: []envTypeAndVersion{
				{version.MajorMinor(1, 26), NewExpressions},
				// always enabled for StoredExpressions
			},
			invalidExpressions: []string{"authorizer.path('/healthz').check('get').allowed()"},
			activation:         map[string]any{"authorizer": library.NewAuthorizerVal(nil, fakeAuthorizer{decision: authorizer.DecisionAllow})},
			opts: []VersionedOptions{
				{IntroducedVersion: version.MajorMinor(1, 27), EnvOptions: []cel.EnvOption{cel.Variable("authorizer", library.AuthorizerType)}},
			},
		},
		{
			name: "authz enabled",
			typeVersionCombinations: []envTypeAndVersion{
				{version.MajorMinor(1, 27), NewExpressions},
				{version.MajorMinor(1, 26), StoredExpressions},
			},
			validExpressions: []string{"authorizer.path('/healthz').check('get').allowed()"},
			activation:       map[string]any{"authorizer": library.NewAuthorizerVal(nil, fakeAuthorizer{decision: authorizer.DecisionAllow})},
			opts: []VersionedOptions{
				{IntroducedVersion: version.MajorMinor(1, 27), EnvOptions: []cel.EnvOption{cel.Variable("authorizer", library.AuthorizerType)}},
			},
		},
		{
			name: "cross numeric comparisons disabled",
			typeVersionCombinations: []envTypeAndVersion{
				{version.MajorMinor(1, 27), NewExpressions},
				// always enabled for StoredExpressions
			},
			invalidExpressions: []string{"1.5 > 1"},
		},
		{
			name: "cross numeric comparisons enabled",
			typeVersionCombinations: []envTypeAndVersion{
				{version.MajorMinor(1, 28), NewExpressions},
				{version.MajorMinor(1, 27), StoredExpressions},
			},
			validExpressions: []string{"1.5 > 1"},
		},
		{
			name: "user defined variable disabled",
			typeVersionCombinations: []envTypeAndVersion{
				{version.MajorMinor(1, 27), NewExpressions},
				// always enabled for StoredExpressions
			},
			invalidExpressions: []string{"fizz == 'buzz'"},
			activation:         map[string]any{"fizz": "buzz"},
			opts: []VersionedOptions{
				{IntroducedVersion: version.MajorMinor(1, 28), EnvOptions: []cel.EnvOption{cel.Variable("fizz", cel.StringType)}},
			},
		},
		{
			name: "user defined variable enabled",
			typeVersionCombinations: []envTypeAndVersion{
				{version.MajorMinor(1, 28), NewExpressions},
				{version.MajorMinor(1, 27), StoredExpressions},
			},
			validExpressions: []string{"fizz == 'buzz'"},
			activation:       map[string]any{"fizz": "buzz"},
			opts: []VersionedOptions{
				{IntroducedVersion: version.MajorMinor(1, 28), EnvOptions: []cel.EnvOption{cel.Variable("fizz", cel.StringType)}},
			},
		},
		{
			name: "declared type enabled before removed",
			typeVersionCombinations: []envTypeAndVersion{
				{version.MajorMinor(1, 28), NewExpressions},
				// always disabled for StoredExpressions
			},
			validExpressions: []string{"widget.x == 'buzz'"},
			activation:       map[string]any{"widget": map[string]any{"x": "buzz"}},
			opts: []VersionedOptions{
				{
					IntroducedVersion: version.MajorMinor(1, 28),
					RemovedVersion:    version.MajorMinor(1, 29),
					DeclTypes:         []*apiservercel.DeclType{widgetsType},
					EnvOptions: []cel.EnvOption{
						cel.Variable("widget", cel.ObjectType("Widget")),
					},
				},
			},
		},
		{
			name: "declared type disabled after removed",
			typeVersionCombinations: []envTypeAndVersion{
				{version.MajorMinor(1, 29), NewExpressions},
				{version.MajorMinor(1, 29), StoredExpressions},
			},
			invalidExpressions: []string{"widget.x == 'buzz'"},
			activation:         map[string]any{"widget": map[string]any{"x": "buzz"}},
			opts: []VersionedOptions{
				{
					IntroducedVersion: version.MajorMinor(1, 28),
					RemovedVersion:    version.MajorMinor(1, 29),
					DeclTypes:         []*apiservercel.DeclType{widgetsType},
					EnvOptions: []cel.EnvOption{
						cel.Variable("widget", cel.ObjectType("Widget")),
					},
				},
			},
		},
		{
			name: "declared type disabled",
			typeVersionCombinations: []envTypeAndVersion{
				{version.MajorMinor(1, 27), NewExpressions},
				// always enabled for StoredExpressions
			},
			invalidExpressions: []string{"widget.x == 'buzz'"},
			activation:         map[string]any{"widget": map[string]any{"x": "buzz"}},
			opts: []VersionedOptions{
				{
					IntroducedVersion: version.MajorMinor(1, 28),
					DeclTypes:         []*apiservercel.DeclType{widgetsType},
					EnvOptions: []cel.EnvOption{
						cel.Variable("widget", widgetsType.CelType()),
					},
				},
			},
		},
		{
			name: "declared type enabled",
			typeVersionCombinations: []envTypeAndVersion{
				{version.MajorMinor(1, 28), NewExpressions},
				{version.MajorMinor(1, 27), StoredExpressions},
			},
			validExpressions: []string{"widget.x == 'buzz'"},
			activation:       map[string]any{"widget": map[string]any{"x": "buzz"}},
			opts: []VersionedOptions{
				{
					IntroducedVersion: version.MajorMinor(1, 28),
					DeclTypes:         []*apiservercel.DeclType{widgetsType},
					EnvOptions: []cel.EnvOption{
						cel.Variable("widget", widgetsType.CelType()),
					},
				},
			},
		},
		{
			name: "library version 0 enabled, version 1 disabled",
			typeVersionCombinations: []envTypeAndVersion{
				{version.MajorMinor(1, 27), NewExpressions},
				// version 1 always enabled for StoredExpressions
			},
			validExpressions:   []string{"test() == true"},
			invalidExpressions: []string{"testV1() == true"},
			opts: []VersionedOptions{
				{
					IntroducedVersion: version.MajorMinor(1, 27),
					RemovedVersion:    version.MajorMinor(1, 28),
					EnvOptions: []cel.EnvOption{
						library.Test(library.TestVersion(0)),
					},
				},
				{
					IntroducedVersion: version.MajorMinor(1, 28),
					EnvOptions: []cel.EnvOption{
						library.Test(library.TestVersion(1)),
					},
				},
			},
		},
		{
			name: "library version 0 disabled, version 1 enabled",
			typeVersionCombinations: []envTypeAndVersion{
				{version.MajorMinor(1, 28), NewExpressions},
				{version.MajorMinor(1, 26), StoredExpressions},
				{version.MajorMinor(1, 27), StoredExpressions},
				{version.MajorMinor(1, 28), StoredExpressions},
			},
			validExpressions: []string{"test() == false", "testV1() == true"},
			opts: []VersionedOptions{
				{
					IntroducedVersion: version.MajorMinor(1, 27),
					RemovedVersion:    version.MajorMinor(1, 28),
					EnvOptions: []cel.EnvOption{
						library.Test(library.TestVersion(0)),
					},
				},
				{
					IntroducedVersion: version.MajorMinor(1, 28),
					EnvOptions: []cel.EnvOption{
						library.Test(library.TestVersion(1)),
					},
				},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			activation := tc.activation
			if activation == nil {
				activation = map[string]any{}
			}
			for _, tv := range tc.typeVersionCombinations {
				t.Run(fmt.Sprintf("version=%s,envType=%s", tv.version.String(), tv.envType), func(t *testing.T) {

					envSet := MustBaseEnvSet(tv.version, true)
					if tc.opts != nil {
						var err error
						envSet, err = envSet.Extend(tc.opts...)
						if err != nil {
							t.Errorf("unexpected error extending environment %v", err)
						}
					}

					envType := NewExpressions
					if len(tv.envType) > 0 {
						envType = tv.envType
					}

					validationEnv, err := envSet.Env(envType)
					if err != nil {
						t.Fatal(err)
					}
					for _, valid := range tc.validExpressions {
						if ok, err := isValid(validationEnv, valid, activation); !ok {
							if err != nil {
								t.Errorf("expected expression to be valid but got %v", err)
							}
							t.Error("expected expression to return true")
						}
					}
					for _, invalid := range tc.invalidExpressions {
						if ok, _ := isValid(validationEnv, invalid, activation); ok {
							t.Errorf("expected invalid expression to result in error")
						}
					}
				})
			}
		})
	}
}

func isValid(env *cel.Env, expr string, activation any) (bool, error) {
	ast, issues := env.Compile(expr)
	if len(issues.Errors()) > 0 {
		return false, issues.Err()
	}
	prog, err := env.Program(ast)
	if err != nil {
		return false, err
	}
	result, _, err := prog.Eval(activation)
	if err != nil {
		return false, err
	}
	return result.Value() == true, nil
}

type fakeAuthorizer struct {
	decision authorizer.Decision
	reason   string
	err      error
}

func (f fakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	return f.decision, f.reason, f.err
}
