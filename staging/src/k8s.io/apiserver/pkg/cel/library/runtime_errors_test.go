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

package library_test

import (
	"context"
	"strings"
	"testing"

	"github.com/google/cel-go/cel"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/cel/library"
)

// TestRuntimeErrors checks that functions return an error instead of panicking
// for values that type checking cannot rule out, such as dyn values whose
// runtime type matches no overload. Admission policies declare object,
// oldObject and params as dyn, so these values come from user input.
func TestRuntimeErrors(t *testing.T) {
	envSet, err := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion()).Extend(
		environment.VersionedOptions{
			IntroducedVersion: version.MajorMinor(1, 0),
			EnvOptions: []cel.EnvOption{
				cel.Variable("object", cel.DynType),
				cel.Variable("authorizer", library.AuthorizerType),
				library.JSONPatch(),
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	env, err := envSet.Env(environment.StoredExpressions)
	if err != nil {
		t.Fatal(err)
	}
	allowAll := authorizer.AuthorizerFunc(func(context.Context, authorizer.Attributes) (authorizer.Decision, string, error) {
		return authorizer.DecisionAllow, "", nil
	})
	vars := map[string]any{
		"object": map[string]any{
			"spec": map[string]any{
				"replicas": int64(3),
				"pattern":  "[0-9]+",
				"weights":  []any{int64(1), 2.5},
				"nulls":    []any{int64(1), nil},
			},
		},
		"authorizer": library.NewAuthorizerVal(&user.DefaultInfo{Name: "alice"}, allowAll),
	}

	const noSuchOverload = "no such overload"
	cases := []struct {
		expr    string
		wantErr string
	}{
		// Fields of object
		{expr: "quantity(object.spec.replicas)", wantErr: noSuchOverload},
		{expr: "string(object.spec.weights)", wantErr: noSuchOverload},
		{expr: "object.spec.replicas.find('[0-9]+')", wantErr: noSuchOverload},
		{expr: "object.spec.weights.sum()", wantErr: noSuchOverload},
		{expr: "object.spec.nulls.isSorted()", wantErr: noSuchOverload},

		// Authz
		{expr: "authorizer.path(dyn(1))", wantErr: noSuchOverload},
		{expr: "dyn(1).path('/healthz')", wantErr: noSuchOverload},
		{expr: "authorizer.group(dyn(1))", wantErr: noSuchOverload},
		{expr: "authorizer.serviceAccount('default', dyn(1))", wantErr: noSuchOverload},
		{expr: "authorizer.group('').resource(dyn(1))", wantErr: noSuchOverload},
		{expr: "authorizer.group('').resource('pods').subresource(dyn(1))", wantErr: noSuchOverload},
		{expr: "authorizer.group('').resource('pods').namespace(dyn(1))", wantErr: noSuchOverload},
		{expr: "authorizer.group('').resource('pods').name(dyn(1))", wantErr: noSuchOverload},
		{expr: "authorizer.group('').resource('pods').fieldSelector(dyn(1))", wantErr: noSuchOverload},
		{expr: "authorizer.group('').resource('pods').labelSelector(dyn(1))", wantErr: noSuchOverload},
		{expr: "authorizer.group('').resource('pods').check(dyn(1))", wantErr: noSuchOverload},
		{expr: "authorizer.path('/healthz').check(dyn(1))", wantErr: noSuchOverload},
		{expr: "dyn(1).check('get')", wantErr: noSuchOverload},
		{expr: "dyn(1).allowed()", wantErr: noSuchOverload},
		{expr: "dyn(1).reason()", wantErr: noSuchOverload},
		{expr: "dyn(1).errored()", wantErr: noSuchOverload},
		{expr: "dyn(1).error()", wantErr: noSuchOverload},

		// CIDR
		{expr: "cidr(dyn(1))", wantErr: noSuchOverload},
		{expr: "isCIDR(dyn(1))", wantErr: noSuchOverload},
		{expr: "cidr('10.0.0.0/8').containsIP(dyn(1))", wantErr: noSuchOverload},
		{expr: "dyn(1).containsIP(ip('10.0.0.1'))", wantErr: noSuchOverload},
		{expr: "cidr('10.0.0.0/8').containsIP('not-an-ip')", wantErr: "parse error"},
		{expr: "cidr('10.0.0.0/8').containsCIDR(dyn(1))", wantErr: noSuchOverload},
		{expr: "dyn(1).containsCIDR(cidr('10.0.0.0/8'))", wantErr: noSuchOverload},
		{expr: "cidr('10.0.0.0/8').containsCIDR('not-a-cidr')", wantErr: "parse error"},
		{expr: "dyn(1).ip()", wantErr: noSuchOverload},
		{expr: "dyn(1).prefixLength()", wantErr: noSuchOverload},
		{expr: "dyn(1).masked()", wantErr: noSuchOverload},
		{expr: "dyn(1).isMask()", wantErr: noSuchOverload},

		// Format
		{expr: "format.named(dyn(1))", wantErr: noSuchOverload},
		{expr: "format.dns1123Label().validate(dyn(1))", wantErr: noSuchOverload},
		{expr: "dyn(1).validate('a')", wantErr: noSuchOverload},

		// IP
		{expr: "ip(dyn(1))", wantErr: noSuchOverload},
		{expr: "isIP(dyn(1))", wantErr: noSuchOverload},
		{expr: "ip.isCanonical(dyn(1))", wantErr: noSuchOverload},
		{expr: "dyn(1).family()", wantErr: noSuchOverload},
		{expr: "dyn(1).isUnspecified()", wantErr: noSuchOverload},
		{expr: "dyn(1).isLoopback()", wantErr: noSuchOverload},
		{expr: "dyn(1).isLinkLocalMulticast()", wantErr: noSuchOverload},
		{expr: "dyn(1).isLinkLocalUnicast()", wantErr: noSuchOverload},
		{expr: "dyn(1).isGlobalUnicast()", wantErr: noSuchOverload},

		// JSONPatch
		{expr: "jsonpatch.escapeKey(dyn(1))", wantErr: noSuchOverload},

		// Lists
		{expr: "dyn(1).isSorted()", wantErr: noSuchOverload},
		{expr: "[dyn(1), dyn(null)].isSorted()", wantErr: noSuchOverload},
		{expr: "dyn(1).sum()", wantErr: noSuchOverload},
		{expr: "[dyn(1), dyn(null)].sum()", wantErr: noSuchOverload},
		{expr: "[dyn(1), dyn('a')].sum()", wantErr: noSuchOverload},
		{expr: "[9223372036854775807, 1].sum()", wantErr: "integer overflow"},
		{expr: "dyn(1).min()", wantErr: noSuchOverload},
		{expr: "[dyn(1), dyn(null)].min()", wantErr: noSuchOverload},
		{expr: "dyn(1).max()", wantErr: noSuchOverload},
		{expr: "[dyn(1), dyn(null)].max()", wantErr: noSuchOverload},
		{expr: "dyn([]).max()", wantErr: "called on empty list"},
		{expr: "dyn(1).indexOf(1)", wantErr: noSuchOverload},
		{expr: "dyn(1).lastIndexOf(1)", wantErr: noSuchOverload},

		// Quantity
		{expr: "quantity(dyn(1))", wantErr: noSuchOverload},
		{expr: "isQuantity(dyn(1))", wantErr: noSuchOverload},
		{expr: "sign(dyn(1))", wantErr: noSuchOverload},
		{expr: "quantity('1').isGreaterThan(dyn(1))", wantErr: noSuchOverload},
		{expr: "quantity('1').isLessThan(dyn(1))", wantErr: noSuchOverload},
		{expr: "quantity('1').compareTo(dyn(1))", wantErr: noSuchOverload},
		{expr: "dyn(1).asApproximateFloat()", wantErr: noSuchOverload},
		{expr: "dyn(1).asInteger()", wantErr: noSuchOverload},
		{expr: "dyn(1).isInteger()", wantErr: noSuchOverload},
		{expr: "quantity('1').add(dyn('a'))", wantErr: noSuchOverload},
		{expr: "dyn('a').add(1)", wantErr: noSuchOverload},
		{expr: "quantity('1').sub(dyn('a'))", wantErr: noSuchOverload},
		{expr: "dyn('a').sub(1)", wantErr: noSuchOverload},

		// Regex, with constant patterns optimized and object.spec.pattern not
		{expr: "dyn(1).find('[0-9]+')", wantErr: noSuchOverload},
		{expr: "dyn(1).find(object.spec.pattern)", wantErr: noSuchOverload},
		{expr: "dyn(1).findAll('[0-9]+')", wantErr: noSuchOverload},
		{expr: "dyn(1).findAll(object.spec.pattern)", wantErr: noSuchOverload},
		{expr: "'123'.findAll('[0-9]+', dyn('a'))", wantErr: noSuchOverload},
		{expr: "'123'.findAll(object.spec.pattern, dyn('a'))", wantErr: noSuchOverload},

		// Semver
		{expr: "semver(dyn(1))", wantErr: noSuchOverload},
		{expr: "semver('1.0.0', dyn(1))", wantErr: noSuchOverload},
		{expr: "isSemver(dyn(1))", wantErr: noSuchOverload},
		{expr: "isSemver('1.0.0', dyn(1))", wantErr: noSuchOverload},
		{expr: "dyn(1).major()", wantErr: noSuchOverload},
		{expr: "dyn(1).minor()", wantErr: noSuchOverload},
		{expr: "dyn(1).patch()", wantErr: noSuchOverload},
		{expr: "semver('1.0.0').isGreaterThan(dyn(1))", wantErr: noSuchOverload},
		{expr: "semver('1.0.0').isLessThan(dyn(1))", wantErr: noSuchOverload},
		{expr: "semver('1.0.0').compareTo(dyn(1))", wantErr: noSuchOverload},

		// URLs
		{expr: "url(dyn(1))", wantErr: noSuchOverload},
		{expr: "isURL(dyn(1))", wantErr: noSuchOverload},
		{expr: "dyn(1).getScheme()", wantErr: noSuchOverload},
		{expr: "dyn(1).getHost()", wantErr: noSuchOverload},
		{expr: "dyn(1).getHostname()", wantErr: noSuchOverload},
		{expr: "dyn(1).getPort()", wantErr: noSuchOverload},
		{expr: "dyn(1).getEscapedPath()", wantErr: noSuchOverload},
		{expr: "dyn(1).getQuery()", wantErr: noSuchOverload},
	}
	for _, tc := range cases {
		t.Run(tc.expr, func(t *testing.T) {
			ast, issues := env.Compile(tc.expr)
			if issues.Err() != nil {
				t.Fatalf("unexpected compile error: %v", issues.Err())
			}
			prg, err := env.Program(ast)
			if err != nil {
				t.Fatalf("unexpected program error: %v", err)
			}
			_, _, err = prg.Eval(vars)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("got error %v, want error containing %q", err, tc.wantErr)
			}
		})
	}
}
