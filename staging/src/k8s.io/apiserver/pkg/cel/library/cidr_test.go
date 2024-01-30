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

package library_test

import (
	"net/netip"
	"regexp"
	"testing"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/library"
)

func testCIDR(t *testing.T, expr string, expectResult ref.Val, expectRuntimeErr string, expectCompileErrs []string) {
	env, err := cel.NewEnv(
		library.IP(),
		library.CIDR(),
	)
	if err != nil {
		t.Fatalf("%v", err)
	}

	compiled, issues := env.Compile(expr)

	if len(expectCompileErrs) > 0 {
		missingCompileErrs := []string{}
		matchedCompileErrs := sets.New[int]()
		for _, expectedCompileErr := range expectCompileErrs {
			compiledPattern, err := regexp.Compile(expectedCompileErr)
			if err != nil {
				t.Fatalf("failed to compile expected err regex: %v", err)
			}

			didMatch := false

			for i, compileError := range issues.Errors() {
				if compiledPattern.Match([]byte(compileError.Message)) {
					didMatch = true
					matchedCompileErrs.Insert(i)
				}
			}

			if !didMatch {
				missingCompileErrs = append(missingCompileErrs, expectedCompileErr)
			} else if len(matchedCompileErrs) != len(issues.Errors()) {
				unmatchedErrs := []cel.Error{}
				for i, issue := range issues.Errors() {
					if !matchedCompileErrs.Has(i) {
						unmatchedErrs = append(unmatchedErrs, *issue)
					}
				}
				require.Empty(t, unmatchedErrs, "unexpected compilation errors")
			}
		}

		require.Empty(t, missingCompileErrs, "expected compilation errors")
		return
	} else if len(issues.Errors()) > 0 {
		t.Fatalf("%v", issues.Errors())
	}

	prog, err := env.Program(compiled)
	if err != nil {
		t.Fatalf("%v", err)
	}
	res, _, err := prog.Eval(map[string]interface{}{})
	if len(expectRuntimeErr) > 0 {
		if err == nil {
			t.Fatalf("no runtime error thrown. Expected: %v", expectRuntimeErr)
		} else if expectRuntimeErr != err.Error() {
			t.Fatalf("unexpected err: %v", err)
		}
	} else if err != nil {
		t.Fatalf("%v", err)
	} else if expectResult != nil {
		converted := res.Equal(expectResult).Value().(bool)
		require.True(t, converted, "expectation not equal to output")
	} else {
		t.Fatal("expected result must not be nil")
	}
}

func TestCIDR(t *testing.T) {
	ipv4CIDR, _ := netip.ParsePrefix("192.168.0.0/24")
	ipv4Addr, _ := netip.ParseAddr("192.168.0.0")

	ipv6CIDR, _ := netip.ParsePrefix("2001:db8::/32")
	ipv6Addr, _ := netip.ParseAddr("2001:db8::")

	trueVal := types.Bool(true)
	falseVal := types.Bool(false)

	cases := []struct {
		name              string
		expr              string
		expectResult      ref.Val
		expectRuntimeErr  string
		expectCompileErrs []string
	}{
		{
			name:         "parse ipv4",
			expr:         `cidr("192.168.0.0/24")`,
			expectResult: apiservercel.CIDR{Prefix: ipv4CIDR},
		},
		{
			name:             "parse invalid ipv4",
			expr:             `cidr("192.168.0.0/")`,
			expectRuntimeErr: "network address parse error during conversion from string: network address parse error during conversion from string: netip.ParsePrefix(\"192.168.0.0/\"): bad bits after slash: \"\"",
		},
		{
			name:         "contains IP ipv4 (IP)",
			expr:         `cidr("192.168.0.0/24").containsIP(ip("192.168.0.1"))`,
			expectResult: trueVal,
		},
		{
			name:         "does not contain IP ipv4 (IP)",
			expr:         `cidr("192.168.0.0/24").containsIP(ip("192.168.1.1"))`,
			expectResult: falseVal,
		},
		{
			name:         "contains IP ipv4 (string)",
			expr:         `cidr("192.168.0.0/24").containsIP("192.168.0.1")`,
			expectResult: trueVal,
		},
		{
			name:         "does not contain IP ipv4 (string)",
			expr:         `cidr("192.168.0.0/24").containsIP("192.168.1.1")`,
			expectResult: falseVal,
		},
		{
			name:         "contains CIDR ipv4 (CIDR)",
			expr:         `cidr("192.168.0.0/24").containsCIDR(cidr("192.168.0.0/25"))`,
			expectResult: trueVal,
		},
		{
			name:         "does not contain IP ipv4 (CIDR)",
			expr:         `cidr("192.168.0.0/24").containsCIDR(cidr("192.168.0.0/23"))`,
			expectResult: falseVal,
		},
		{
			name:         "contains CIDR ipv4 (string)",
			expr:         `cidr("192.168.0.0/24").containsCIDR("192.168.0.0/25")`,
			expectResult: trueVal,
		},
		{
			name:         "does not contain CIDR ipv4 (string)",
			expr:         `cidr("192.168.0.0/24").containsCIDR("192.168.0.0/23")`,
			expectResult: falseVal,
		},
		{
			name:         "returns IP ipv4",
			expr:         `cidr("192.168.0.0/24").ip()`,
			expectResult: apiservercel.IP{Addr: ipv4Addr},
		},
		{
			name:         "masks masked ipv4",
			expr:         `cidr("192.168.0.0/24").masked()`,
			expectResult: apiservercel.CIDR{Prefix: netip.PrefixFrom(ipv4Addr, 24)},
		},
		{
			name:         "masks unmasked ipv4",
			expr:         `cidr("192.168.0.1/24").masked()`,
			expectResult: apiservercel.CIDR{Prefix: netip.PrefixFrom(ipv4Addr, 24)},
		},
		{
			name:         "returns prefix length ipv4",
			expr:         `cidr("192.168.0.0/24").prefixLength()`,
			expectResult: types.Int(24),
		},
		{
			name:         "parse ipv6",
			expr:         `cidr("2001:db8::/32")`,
			expectResult: apiservercel.CIDR{Prefix: ipv6CIDR},
		},
		{
			name:             "parse invalid ipv6",
			expr:             `cidr("2001:db8::/")`,
			expectRuntimeErr: "network address parse error during conversion from string: network address parse error during conversion from string: netip.ParsePrefix(\"2001:db8::/\"): bad bits after slash: \"\"",
		},
		{
			name:         "contains IP ipv6 (IP)",
			expr:         `cidr("2001:db8::/32").containsIP(ip("2001:db8::1"))`,
			expectResult: trueVal,
		},
		{
			name:         "does not contain IP ipv6 (IP)",
			expr:         `cidr("2001:db8::/32").containsIP(ip("2001:dc8::1"))`,
			expectResult: falseVal,
		},
		{
			name:         "contains IP ipv6 (string)",
			expr:         `cidr("2001:db8::/32").containsIP("2001:db8::1")`,
			expectResult: trueVal,
		},
		{
			name:         "does not contain IP ipv6 (string)",
			expr:         `cidr("2001:db8::/32").containsIP("2001:dc8::1")`,
			expectResult: falseVal,
		},
		{
			name:         "contains CIDR ipv6 (CIDR)",
			expr:         `cidr("2001:db8::/32").containsCIDR(cidr("2001:db8::/33"))`,
			expectResult: trueVal,
		},
		{
			name:         "does not contain IP ipv6 (CIDR)",
			expr:         `cidr("2001:db8::/32").containsCIDR(cidr("2001:db8::/31"))`,
			expectResult: falseVal,
		},
		{
			name:         "contains CIDR ipv6 (string)",
			expr:         `cidr("2001:db8::/32").containsCIDR("2001:db8::/33")`,
			expectResult: trueVal,
		},
		{
			name:         "does not contain CIDR ipv6 (string)",
			expr:         `cidr("2001:db8::/32").containsCIDR("2001:db8::/31")`,
			expectResult: falseVal,
		},
		{
			name:         "returns IP ipv6",
			expr:         `cidr("2001:db8::/32").ip()`,
			expectResult: apiservercel.IP{Addr: ipv6Addr},
		},
		{
			name:         "masks masked ipv6",
			expr:         `cidr("2001:db8::/32").masked()`,
			expectResult: apiservercel.CIDR{Prefix: netip.PrefixFrom(ipv6Addr, 32)},
		},
		{
			name:         "masks unmasked ipv6",
			expr:         `cidr("2001:db8:1::/32").masked()`,
			expectResult: apiservercel.CIDR{Prefix: netip.PrefixFrom(ipv6Addr, 32)},
		},
		{
			name:         "returns prefix length ipv6",
			expr:         `cidr("2001:db8::/32").prefixLength()`,
			expectResult: types.Int(32),
		},
		{
			name:         "converting a CIDR to a string",
			expr:         `string(cidr("192.168.0.0/24"))`,
			expectResult: types.String("192.168.0.0/24"),
		},
		{
			name:         "type of CIDR is net.CIDR",
			expr:         `type(cidr("192.168.0.0/24")) == net.CIDR`,
			expectResult: trueVal,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testCIDR(t, tc.expr, tc.expectResult, tc.expectRuntimeErr, tc.expectCompileErrs)
		})
	}
}
