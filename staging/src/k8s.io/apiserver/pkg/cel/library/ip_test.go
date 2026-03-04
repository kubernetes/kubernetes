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

func testIP(t *testing.T, expr string, expectResult ref.Val, expectRuntimeErr string, expectCompileErrs []string) {
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

func TestIP(t *testing.T) {
	ipv4Addr, _ := netip.ParseAddr("192.168.0.1")
	int4 := types.Int(4)

	ipv6Addr, _ := netip.ParseAddr("2001:db8::68")
	int6 := types.Int(6)

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
			expr:         `ip("192.168.0.1")`,
			expectResult: apiservercel.IP{Addr: ipv4Addr},
		},
		{
			name:             "parse invalid ipv4",
			expr:             `ip("192.168.0.1.0")`,
			expectRuntimeErr: "IP Address \"192.168.0.1.0\" parse error during conversion from string: ParseAddr(\"192.168.0.1.0\"): IPv4 address too long",
		},
		{
			name:         "isIP valid ipv4",
			expr:         `isIP("192.168.0.1")`,
			expectResult: trueVal,
		},
		{
			name:         "isIP invalid ipv4",
			expr:         `isIP("192.168.0.1.0")`,
			expectResult: falseVal,
		},
		{
			name:         "ip.isCanonical valid ipv4",
			expr:         `ip.isCanonical("127.0.0.1")`,
			expectResult: trueVal,
		},
		{
			name:             "ip.isCanonical invalid ipv4",
			expr:             `ip.isCanonical("127.0.0.1.0")`,
			expectRuntimeErr: "IP Address \"127.0.0.1.0\" parse error during conversion from string: ParseAddr(\"127.0.0.1.0\"): IPv4 address too long",
		},
		{
			name:         "ipv4 family",
			expr:         `ip("192.168.0.1").family()`,
			expectResult: int4,
		},
		{
			name:         "ipv4 isUnspecified true",
			expr:         `ip("0.0.0.0").isUnspecified()`,
			expectResult: trueVal,
		},
		{
			name:         "ipv4 isUnspecified false",
			expr:         `ip("127.0.0.1").isUnspecified()`,
			expectResult: falseVal,
		},
		{
			name:         "ipv4 isLoopback true",
			expr:         `ip("127.0.0.1").isLoopback()`,
			expectResult: trueVal,
		},
		{
			name:         "ipv4 isLoopback false",
			expr:         `ip("1.2.3.4").isLoopback()`,
			expectResult: falseVal,
		},
		{
			name:         "ipv4 isLinkLocalMulticast true",
			expr:         `ip("224.0.0.1").isLinkLocalMulticast()`,
			expectResult: trueVal,
		},
		{
			name:         "ipv4 isLinkLocalMulticast false",
			expr:         `ip("224.0.1.1").isLinkLocalMulticast()`,
			expectResult: falseVal,
		},
		{
			name:         "ipv4 isLinkLocalUnicast true",
			expr:         `ip("169.254.169.254").isLinkLocalUnicast()`,
			expectResult: trueVal,
		},
		{
			name:         "ipv4 isLinkLocalUnicast false",
			expr:         `ip("192.168.0.1").isLinkLocalUnicast()`,
			expectResult: falseVal,
		},
		{
			name:         "ipv4 isGlobalUnicast true",
			expr:         `ip("192.168.0.1").isGlobalUnicast()`,
			expectResult: trueVal,
		},
		{
			name:         "ipv4 isGlobalUnicast false",
			expr:         `ip("255.255.255.255").isGlobalUnicast()`,
			expectResult: falseVal,
		},
		{
			name:         "parse ipv6",
			expr:         `ip("2001:db8::68")`,
			expectResult: apiservercel.IP{Addr: ipv6Addr},
		},
		{
			name:             "parse invalid ipv6",
			expr:             `ip("2001:db8:::68")`,
			expectRuntimeErr: "IP Address \"2001:db8:::68\" parse error during conversion from string: ParseAddr(\"2001:db8:::68\"): each colon-separated field must have at least one digit (at \":68\")",
		},
		{
			name:         "isIP valid ipv6",
			expr:         `isIP("2001:db8::68")`,
			expectResult: trueVal,
		},
		{
			name:         "isIP invalid ipv4",
			expr:         `isIP("2001:db8:::68")`,
			expectResult: falseVal,
		},
		{
			name:         "ip.isCanonical valid ipv6",
			expr:         `ip.isCanonical("2001:db8::68")`,
			expectResult: trueVal,
		},
		{
			name:         "ip.isCanonical non-canonical ipv6",
			expr:         `ip.isCanonical("2001:DB8::68")`,
			expectResult: falseVal,
		},
		{
			name:             "ip.isCanonical invalid ipv6",
			expr:             `ip.isCanonical("2001:db8:::68")`,
			expectRuntimeErr: "IP Address \"2001:db8:::68\" parse error during conversion from string: ParseAddr(\"2001:db8:::68\"): each colon-separated field must have at least one digit (at \":68\")",
		},
		{
			name:         "ipv6 family",
			expr:         `ip("2001:db8::68").family()`,
			expectResult: int6,
		},
		{
			name:         "ipv6 isUnspecified true",
			expr:         `ip("::").isUnspecified()`,
			expectResult: trueVal,
		},
		{
			name:         "ipv6 isUnspecified false",
			expr:         `ip("::1").isUnspecified()`,
			expectResult: falseVal,
		},
		{
			name:         "ipv6 isLoopback true",
			expr:         `ip("::1").isLoopback()`,
			expectResult: trueVal,
		},
		{
			name:         "ipv6 isLoopback false",
			expr:         `ip("2001:db8::abcd").isLoopback()`,
			expectResult: falseVal,
		},
		{
			name:         "ipv6 isLinkLocalMulticast true",
			expr:         `ip("ff02::1").isLinkLocalMulticast()`,
			expectResult: trueVal,
		},
		{
			name:         "ipv6 isLinkLocalMulticast false",
			expr:         `ip("fd00::1").isLinkLocalMulticast()`,
			expectResult: falseVal,
		},
		{
			name:         "ipv6 isLinkLocalUnicast true",
			expr:         `ip("fe80::1").isLinkLocalUnicast()`,
			expectResult: trueVal,
		},
		{
			name:         "ipv6 isLinkLocalUnicast false",
			expr:         `ip("fd80::1").isLinkLocalUnicast()`,
			expectResult: falseVal,
		},
		{
			name:         "ipv6 isGlobalUnicast true",
			expr:         `ip("2001:db8::abcd").isGlobalUnicast()`,
			expectResult: trueVal,
		},
		{
			name:         "ipv6 isGlobalUnicast false",
			expr:         `ip("ff00::1").isGlobalUnicast()`,
			expectResult: falseVal,
		},
		{
			name:              "passing cidr into isIP returns compile error",
			expr:              `isIP(cidr("192.168.0.0/24"))`,
			expectCompileErrs: []string{"found no matching overload for 'isIP' applied to '\\(net.CIDR\\)'"},
		},
		{
			name:         "converting an IP address to a string",
			expr:         `string(ip("192.168.0.1"))`,
			expectResult: types.String("192.168.0.1"),
		},
		{
			name:         "type of IP is net.IP",
			expr:         `type(ip("192.168.0.1")) == net.IP`,
			expectResult: trueVal,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testIP(t, tc.expr, tc.expectResult, tc.expectRuntimeErr, tc.expectCompileErrs)
		})
	}
}
