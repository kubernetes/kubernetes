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
	"testing"

	"cel.dev/cel-go/cel"
	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/common/types/ref"
	"cel.dev/cel-go/ext"

	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/library"
)

// TestNetworkLibraryCompatibility verifies that cel-go's network library, which replaces the
// Kubernetes IP and CIDR libraries in the base environment from Kubernetes 1.38, is a drop-in
// replacement: every expression that compiles and evaluates with the Kubernetes libraries
// compiles and evaluates to the same value with cel-go's network library.
//
// The expressions below mirror the cases in ip_test.go and cidr_test.go. The expected values
// are not repeated here; the Kubernetes libraries are the reference implementation and are
// covered by those tests. This test only checks that both libraries agree.
//
// The one known difference is that cel-go's network library validates ip() and cidr() string
// literals at compile time, while the Kubernetes libraries fail at runtime. Such cases are
// marked with invalidLiteral and are expected to fail at a different stage, but still fail.
func TestNetworkLibraryCompatibility(t *testing.T) {
	cases := []struct {
		expr string
		// invalidLiteral marks expressions with an invalid ip() or cidr() string literal, which
		// the Kubernetes libraries reject at runtime and cel-go's network library at compile time.
		invalidLiteral bool
	}{
		// ip() / isIP() / ip.isCanonical()
		{expr: `ip("192.168.0.1")`},
		{expr: `ip("192.168.0.1.0")`, invalidLiteral: true},
		{expr: `isIP("192.168.0.1")`},
		{expr: `isIP("192.168.0.1.0")`},
		{expr: `ip.isCanonical("127.0.0.1")`},
		{expr: `ip.isCanonical("127.0.0.1.0")`},
		{expr: `ip("2001:db8::68")`},
		{expr: `ip("2001:db8:::68")`, invalidLiteral: true},
		{expr: `isIP("2001:db8::68")`},
		{expr: `isIP("2001:db8:::68")`},
		{expr: `ip.isCanonical("2001:db8::68")`},
		{expr: `ip.isCanonical("2001:DB8::68")`},
		{expr: `ip.isCanonical("2001:db8:::68")`},
		// IP accessors
		{expr: `ip("192.168.0.1").family()`},
		{expr: `ip("2001:db8::68").family()`},
		{expr: `ip("0.0.0.0").isUnspecified()`},
		{expr: `ip("127.0.0.1").isUnspecified()`},
		{expr: `ip("::").isUnspecified()`},
		{expr: `ip("::1").isUnspecified()`},
		{expr: `ip("127.0.0.1").isLoopback()`},
		{expr: `ip("1.2.3.4").isLoopback()`},
		{expr: `ip("::1").isLoopback()`},
		{expr: `ip("2001:db8::abcd").isLoopback()`},
		{expr: `ip("224.0.0.1").isLinkLocalMulticast()`},
		{expr: `ip("224.0.1.1").isLinkLocalMulticast()`},
		{expr: `ip("ff02::1").isLinkLocalMulticast()`},
		{expr: `ip("fd00::1").isLinkLocalMulticast()`},
		{expr: `ip("169.254.169.254").isLinkLocalUnicast()`},
		{expr: `ip("192.168.0.1").isLinkLocalUnicast()`},
		{expr: `ip("fe80::1").isLinkLocalUnicast()`},
		{expr: `ip("fd80::1").isLinkLocalUnicast()`},
		{expr: `ip("192.168.0.1").isGlobalUnicast()`},
		{expr: `ip("255.255.255.255").isGlobalUnicast()`},
		{expr: `ip("2001:db8::abcd").isGlobalUnicast()`},
		{expr: `ip("ff00::1").isGlobalUnicast()`},
		{expr: `string(ip("192.168.0.1"))`},
		{expr: `string(ip("2001:db8::68"))`},
		{expr: `type(ip("192.168.0.1")) == net.IP`},
		{expr: `ip("192.168.0.1") == ip("192.168.0.1")`},
		{expr: `ip("192.168.0.1") != ip("192.168.0.2")`},
		// cidr() / isCIDR()
		{expr: `cidr("192.168.0.0/24")`},
		{expr: `cidr("192.168.0.1/24")`},
		{expr: `cidr("192.168.0.0/")`, invalidLiteral: true},
		{expr: `cidr("2001:db8::/32")`},
		{expr: `cidr("2001:db8::1/32")`},
		{expr: `cidr("2001:db8::/")`, invalidLiteral: true},
		{expr: `isCIDR("192.168.0.0/24")`},
		{expr: `isCIDR("192.168.0.0/")`},
		{expr: `isCIDR("2001:db8::/32")`},
		{expr: `isCIDR("2001:db8::1/32")`},
		// CIDR accessors
		{expr: `cidr("192.168.0.0/24").containsIP(ip("192.168.0.1"))`},
		{expr: `cidr("192.168.0.0/24").containsIP(ip("192.168.1.1"))`},
		{expr: `cidr("192.168.0.0/24").containsIP("192.168.0.1")`},
		{expr: `cidr("192.168.0.0/24").containsIP("192.168.1.1")`},
		{expr: `cidr("2001:db8::/32").containsIP(ip("2001:db8::1"))`},
		{expr: `cidr("2001:db8::/32").containsIP(ip("2001:dc8::1"))`},
		{expr: `cidr("2001:db8::/32").containsIP("2001:db8::1")`},
		{expr: `cidr("2001:db8::/32").containsIP("2001:dc8::1")`},
		{expr: `cidr("192.168.0.0/24").containsCIDR(cidr("192.168.0.0/25"))`},
		{expr: `cidr("192.168.0.0/24").containsCIDR(cidr("192.168.0.0/23"))`},
		{expr: `cidr("192.168.0.0/24").containsCIDR("192.168.0.0/25")`},
		{expr: `cidr("192.168.0.0/24").containsCIDR("192.168.0.0/23")`},
		{expr: `cidr("2001:db8::/32").containsCIDR(cidr("2001:db8::/33"))`},
		{expr: `cidr("2001:db8::/32").containsCIDR(cidr("2001:db8::/31"))`},
		{expr: `cidr("2001:db8::/32").containsCIDR("2001:db8::/33")`},
		{expr: `cidr("2001:db8::/32").containsCIDR("2001:db8::/31")`},
		{expr: `cidr("192.168.0.0/24").ip()`},
		{expr: `cidr("2001:db8::/32").ip()`},
		{expr: `cidr("192.168.0.0/24").isMask()`},
		{expr: `cidr("192.168.0.1/24").isMask()`},
		{expr: `cidr("2001:db8::/32").isMask()`},
		{expr: `cidr("2001:db8:1::/32").isMask()`},
		{expr: `cidr("192.168.0.0/24").masked()`},
		{expr: `cidr("192.168.0.1/24").masked()`},
		{expr: `cidr("2001:db8::/32").masked()`},
		{expr: `cidr("2001:db8:1::/32").masked()`},
		{expr: `cidr("192.168.0.0/24").masked() == cidr("192.168.0.0/24")`},
		{expr: `cidr("192.168.0.1/24").masked() == cidr("192.168.0.1/24")`},
		{expr: `cidr("2001:db8::1/32").masked() == cidr("2001:db8::/32")`},
		{expr: `cidr("192.168.0.0/24").prefixLength()`},
		{expr: `cidr("2001:db8::/32").prefixLength()`},
		{expr: `string(cidr("192.168.0.0/24"))`},
		{expr: `string(cidr("2001:db8::/32"))`},
		{expr: `type(cidr("192.168.0.0/24")) == net.CIDR`},
		{expr: `cidr("192.168.0.0/24") == cidr("192.168.0.0/24")`},
		{expr: `cidr("192.168.0.0/24") != cidr("192.168.0.0/25")`},
		// Type mismatches are compile errors with both libraries.
		{expr: `isIP(cidr("192.168.0.0/24"))`},
		{expr: `isCIDR(ip("192.168.0.1"))`},
		{expr: `cidr("192.168.0.0/24").containsIP(cidr("192.168.0.0/25"))`},
	}

	kubernetesEnv, err := cel.NewEnv(library.IP(), library.CIDR(library.CIDRVersion(1)))
	if err != nil {
		t.Fatal(err)
	}
	celGoEnv, err := cel.NewEnv(ext.Network(ext.NetworkVersion(1)))
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range cases {
		t.Run(tc.expr, func(t *testing.T) {
			kubernetes := evalNetworkExpr(t, kubernetesEnv, tc.expr)
			celGo := evalNetworkExpr(t, celGoEnv, tc.expr)

			if tc.invalidLiteral {
				// Known difference: the Kubernetes libraries fail at runtime, cel-go at compile time.
				if kubernetes.compileErr != nil || kubernetes.evalErr == nil {
					t.Errorf("expected the kubernetes libraries to fail at runtime, got compile error %v, runtime error %v", kubernetes.compileErr, kubernetes.evalErr)
				}
				if celGo.compileErr == nil {
					t.Errorf("expected the cel-go network library to fail at compile time, got compile error %v, runtime error %v", celGo.compileErr, celGo.evalErr)
				}
				return
			}

			if (kubernetes.compileErr == nil) != (celGo.compileErr == nil) {
				t.Fatalf("compile result differs: kubernetes=%v, cel-go=%v", kubernetes.compileErr, celGo.compileErr)
			}
			if kubernetes.compileErr != nil {
				return
			}
			if (kubernetes.evalErr == nil) != (celGo.evalErr == nil) {
				t.Fatalf("eval result differs: kubernetes=%v, cel-go=%v", kubernetes.evalErr, celGo.evalErr)
			}
			if kubernetes.evalErr != nil {
				return
			}
			if celGo.result.Equal(toCELGoValue(kubernetes.result)) != types.True {
				t.Errorf("value differs: kubernetes=%v (%T), cel-go=%v (%T)", kubernetes.result.Value(), kubernetes.result, celGo.result.Value(), celGo.result)
			}
		})
	}
}

// TestNetworkLibraryTypeNames verifies that cel-go's network library uses the same type names as
// the Kubernetes IP and CIDR libraries, so that stored expressions referring to net.IP and
// net.CIDR keep compiling when the base environment switches libraries.
func TestNetworkLibraryTypeNames(t *testing.T) {
	if got, want := ext.IPType.TypeName(), apiservercel.IPType.TypeName(); got != want {
		t.Errorf("IP type name: cel-go=%q, kubernetes=%q", got, want)
	}
	if got, want := ext.CIDRType.TypeName(), apiservercel.CIDRType.TypeName(); got != want {
		t.Errorf("CIDR type name: cel-go=%q, kubernetes=%q", got, want)
	}
}

type networkEvalResult struct {
	compileErr error
	evalErr    error
	result     ref.Val
}

func evalNetworkExpr(t *testing.T, env *cel.Env, expr string) networkEvalResult {
	t.Helper()
	compiled, issues := env.Compile(expr)
	if issues != nil && issues.Err() != nil {
		return networkEvalResult{compileErr: issues.Err()}
	}
	prog, err := env.Program(compiled)
	if err != nil {
		t.Fatalf("%v", err)
	}
	res, _, err := prog.Eval(map[string]interface{}{})
	if err != nil {
		return networkEvalResult{evalErr: err}
	}
	return networkEvalResult{result: res}
}

// toCELGoValue converts Kubernetes IP and CIDR values to their cel-go equivalents so that the
// results of both libraries can be compared with Equal.
func toCELGoValue(v ref.Val) ref.Val {
	switch v := v.(type) {
	case apiservercel.IP:
		return ext.IP{Addr: v.Addr}
	case apiservercel.CIDR:
		return ext.CIDR{Prefix: v.Prefix}
	}
	return v
}
