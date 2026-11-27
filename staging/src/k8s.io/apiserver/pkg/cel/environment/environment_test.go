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

	"cel.dev/cel-go/cel"
	"cel.dev/cel-go/ext"

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

	// The escaping happens while construct declType hence we use escaped format here directly.
	gadgetsType := apiservercel.NewObjectType("Gadget",
		map[string]*apiservercel.DeclField{
			"__namespace__": {
				Name: "__namespace__",
				Type: apiservercel.StringType,
			},
		})

	// Gated functions like 'includes' become available at 1.37. When the default compatibility version rolls over to 1.37 or later, 'includes' will be valid. We dynamically select valid/invalid expressions depending on the current DefaultCompatibilityVersion to prevent test failures during Kubernetes version bump.
	var defaultCompatibilityInvalid []string
	var defaultCompatibilityValid []string
	if DefaultCompatibilityVersion().AtLeast(version.MajorMinor(1, 37)) {
		defaultCompatibilityValid = []string{
			"[1, 2].includes(2) == true",
		}
	} else {
		defaultCompatibilityInvalid = []string{
			"[1, 2].includes(2)",
		}
	}

	// networkLibraryExpressions must be valid with both the kubernetes IP and CIDR libraries
	// (before 1.38) and the cel-go network library that replaces them (from 1.38).
	networkLibraryExpressions := []string{
		"ip('192.168.0.1').family() == 4",
		"isIP('2001:db8::68') && !isIP('192.168.0.1.0')",
		"ip('::1').isLoopback()",
		"string(ip('192.168.0.1')) == '192.168.0.1'",
		"ip('192.168.0.1') == ip('192.168.0.1')",
		"type(ip('192.168.0.1')) == net.IP",
		"isCIDR('192.168.0.0/24') && !isCIDR('192.168.0.0/33')",
		"cidr('192.168.0.0/24').containsIP(ip('192.168.0.1'))",
		"cidr('192.168.0.0/16').containsCIDR('192.168.1.0/24')",
		"cidr('192.168.0.1/24').masked() == cidr('192.168.0.0/24')",
		"cidr('192.168.0.0/24').prefixLength() == 24",
		"type(cidr('192.168.0.0/24')) == net.CIDR",
	}

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
		{
			name: "recognizeKeywordAsFieldName disabled",
			typeVersionCombinations: []envTypeAndVersion{
				{version.MajorMinor(1, 30), NewExpressions},
				// always enabled for StoredExpressions
			},
			invalidExpressions: []string{"gadget.namespace == 'buzz'"},
			activation:         map[string]any{"gadget": map[string]any{"namespace": "buzz"}},
			opts: []VersionedOptions{
				{
					IntroducedVersion: version.MajorMinor(1, 28),
					DeclTypes:         []*apiservercel.DeclType{gadgetsType},
					EnvOptions: []cel.EnvOption{
						cel.Variable("gadget", cel.ObjectType("Gadget")),
					},
				},
			},
		},
		{
			name: "recognizeKeywordAsFieldName enabled",
			typeVersionCombinations: []envTypeAndVersion{
				{version.MajorMinor(1, 31), NewExpressions},
				{version.MajorMinor(1, 30), StoredExpressions},
			},
			validExpressions: []string{"gadget.namespace == 'buzz'"},
			activation:       map[string]any{"gadget": map[string]any{"namespace": "buzz"}},
			opts: []VersionedOptions{
				{
					IntroducedVersion: version.MajorMinor(1, 28),
					DeclTypes:         []*apiservercel.DeclType{gadgetsType},
					EnvOptions: []cel.EnvOption{
						cel.Variable("gadget", cel.ObjectType("Gadget")),
					},
				},
			},
		},
		{
			name: "includes function version gating (default compatibility version)",
			typeVersionCombinations: []envTypeAndVersion{
				{version: DefaultCompatibilityVersion(), envType: NewExpressions},
			},
			invalidExpressions: defaultCompatibilityInvalid,
			validExpressions:   defaultCompatibilityValid,
		},
		{
			name: "includes function version gating (emulated version 1.36)",
			typeVersionCombinations: []envTypeAndVersion{
				{version: version.MajorMinor(1, 36), envType: NewExpressions},
			},
			invalidExpressions: []string{
				"[1, 2].includes(2)",
			},
		},
		{
			name: "includes function available (emulated version 1.37)",
			typeVersionCombinations: []envTypeAndVersion{
				{version: version.MajorMinor(1, 37), envType: NewExpressions},
			},
			validExpressions: []string{
				"[1, 2].includes(2) == true",
			},
		},
		{
			name: "includes function available (stored expressions for default/1.36)",
			typeVersionCombinations: []envTypeAndVersion{
				{version: DefaultCompatibilityVersion(), envType: StoredExpressions},
				{version: version.MajorMinor(1, 36), envType: StoredExpressions},
			},
			validExpressions: []string{
				"[1, 2].includes(2) == true",
			},
		},
		{
			name: "kubernetes IP and CIDR libraries (before 1.38)",
			typeVersionCombinations: []envTypeAndVersion{
				{version: version.MajorMinor(1, 30), envType: NewExpressions},
				{version: version.MajorMinor(1, 37), envType: NewExpressions},
			},
			validExpressions: networkLibraryExpressions,
			invalidExpressions: []string{
				"cidr('192.168.0.0/24').isMask()",
			},
		},
		{
			name: "cel-go network library (from 1.38, and stored expressions)",
			typeVersionCombinations: []envTypeAndVersion{
				{version: version.MajorMinor(1, 38), envType: NewExpressions},
				{version: version.MajorMinor(1, 30), envType: StoredExpressions},
				{version: version.MajorMinor(1, 37), envType: StoredExpressions},
			},
			validExpressions: append(append([]string{}, networkLibraryExpressions...),
				"cidr('192.168.0.0/24').isMask()",
				"!cidr('192.168.0.1/24').isMask()",
			),
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

					envSet := MustBaseEnvSet(tv.version)
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

// TestNetworkLibraryMigration verifies that the kubernetes IP and CIDR libraries are used for
// new expressions before 1.38, and that the cel-go network library is used from 1.38 and for
// stored expressions.
func TestNetworkLibraryMigration(t *testing.T) {
	cases := []struct {
		name    string
		version *version.Version
		envType Type
		celGo   bool
	}{
		{name: "new expressions 1.30", version: version.MajorMinor(1, 30), envType: NewExpressions, celGo: false},
		{name: "new expressions 1.37", version: version.MajorMinor(1, 37), envType: NewExpressions, celGo: false},
		{name: "new expressions 1.38", version: version.MajorMinor(1, 38), envType: NewExpressions, celGo: true},
		{name: "stored expressions 1.37", version: version.MajorMinor(1, 37), envType: StoredExpressions, celGo: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			env, err := MustBaseEnvSet(tc.version).Env(tc.envType)
			if err != nil {
				t.Fatal(err)
			}

			eval := func(expr string) any {
				t.Helper()
				ast, issues := env.Compile(expr)
				if issues.Err() != nil {
					t.Fatalf("unexpected compile error for %q: %v", expr, issues.Err())
				}
				prog, err := env.Program(ast)
				if err != nil {
					t.Fatalf("unexpected program error for %q: %v", expr, err)
				}
				result, _, err := prog.Eval(map[string]any{})
				if err != nil {
					t.Fatalf("unexpected eval error for %q: %v", expr, err)
				}
				return result
			}

			ipVal := eval("ip('192.168.0.1')")
			cidrVal := eval("cidr('192.168.0.0/24')")
			if tc.celGo {
				if _, ok := ipVal.(ext.IP); !ok {
					t.Errorf("expected cel-go ext.IP, got %T", ipVal)
				}
				if _, ok := cidrVal.(ext.CIDR); !ok {
					t.Errorf("expected cel-go ext.CIDR, got %T", cidrVal)
				}
			} else {
				if _, ok := ipVal.(apiservercel.IP); !ok {
					t.Errorf("expected apiservercel.IP, got %T", ipVal)
				}
				if _, ok := cidrVal.(apiservercel.CIDR); !ok {
					t.Errorf("expected apiservercel.CIDR, got %T", cidrVal)
				}
			}

			// The cel-go network library validates ip() and cidr() string literals at compile
			// time, the kubernetes libraries report invalid literals at runtime.
			for _, expr := range []string{"ip('192.168.0.1.0')", "cidr('192.168.0.0/33')"} {
				_, issues := env.Compile(expr)
				if gotCompileErr := issues.Err() != nil; gotCompileErr != tc.celGo {
					t.Errorf("expected compile error for %q: %v, got: %v", expr, tc.celGo, issues.Err())
				}
			}
		})
	}
}

type fakeAuthorizer struct {
	decision authorizer.Decision
	reason   string
	err      error
}

func (f fakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	return f.decision, f.reason, f.err
}
