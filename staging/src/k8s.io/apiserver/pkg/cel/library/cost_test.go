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

package library

import (
	"context"
	"fmt"
	"testing"

	"github.com/google/cel-go/common/types/ref"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker"
	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/ext"
	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"

	"k8s.io/apiserver/pkg/authorization/authorizer"
	apiservercel "k8s.io/apiserver/pkg/cel"
)

const (
	intListLiteral       = "[1, 2, 3, 4, 5]"
	uintListLiteral      = "[uint(1), uint(2), uint(3), uint(4), uint(5)]"
	doubleListLiteral    = "[1.0, 2.0, 3.0, 4.0, 5.0]"
	boolListLiteral      = "[false, true, false, true, false]"
	stringListLiteral    = "['012345678901', '012345678901', '012345678901', '012345678901', '012345678901']"
	bytesListLiteral     = "[bytes('012345678901'), bytes('012345678901'), bytes('012345678901'), bytes('012345678901'), bytes('012345678901')]"
	durationListLiteral  = "[duration('1s'), duration('2s'), duration('3s'), duration('4s'), duration('5s')]"
	timestampListLiteral = "[timestamp('2011-01-01T00:00:00.000+01:00'), timestamp('2011-01-02T00:00:00.000+01:00'), " +
		"timestamp('2011-01-03T00:00:00.000+01:00'), timestamp('2011-01-04T00:00:00.000+01:00'), " +
		"timestamp('2011-01-05T00:00:00.000+01:00')]"
	stringLiteral = "'01234567890123456789012345678901234567890123456789'"
)

type comparableCost struct {
	comparableLiteral     string
	expectedEstimatedCost checker.CostEstimate
	expectedRuntimeCost   uint64

	param string
}

func TestListsCost(t *testing.T) {
	cases := []struct {
		opts  []string
		costs []comparableCost
	}{
		{
			opts: []string{".sum()"},
			// 10 cost for the list declaration, the rest is the due to the function call
			costs: []comparableCost{
				{
					comparableLiteral:     intListLiteral,
					expectedEstimatedCost: checker.CostEstimate{Min: 15, Max: 15}, expectedRuntimeCost: 15,
				},
				{
					comparableLiteral:     uintListLiteral,
					expectedEstimatedCost: checker.CostEstimate{Min: 20, Max: 20}, expectedRuntimeCost: 20, // +5 for casts
				},
				{
					comparableLiteral:     doubleListLiteral,
					expectedEstimatedCost: checker.CostEstimate{Min: 15, Max: 15}, expectedRuntimeCost: 15,
				},
				{
					comparableLiteral:     durationListLiteral,
					expectedEstimatedCost: checker.CostEstimate{Min: 20, Max: 20}, expectedRuntimeCost: 20, // +5 for casts
				},
			},
		},
		{
			opts: []string{".isSorted()", ".max()", ".min()"},
			// 10 cost for the list declaration, the rest is the due to the function call
			costs: []comparableCost{
				{
					comparableLiteral:     intListLiteral,
					expectedEstimatedCost: checker.CostEstimate{Min: 15, Max: 15}, expectedRuntimeCost: 15,
				},
				{
					comparableLiteral:     uintListLiteral,
					expectedEstimatedCost: checker.CostEstimate{Min: 20, Max: 20}, expectedRuntimeCost: 20, // +5 for numeric casts
				},
				{
					comparableLiteral:     doubleListLiteral,
					expectedEstimatedCost: checker.CostEstimate{Min: 15, Max: 15}, expectedRuntimeCost: 15,
				},
				{
					comparableLiteral:     boolListLiteral,
					expectedEstimatedCost: checker.CostEstimate{Min: 15, Max: 15}, expectedRuntimeCost: 15,
				},
				{
					comparableLiteral:     stringListLiteral,
					expectedEstimatedCost: checker.CostEstimate{Min: 15, Max: 25}, expectedRuntimeCost: 15, // +5 for string comparisons
				},
				{
					comparableLiteral:     bytesListLiteral,
					expectedEstimatedCost: checker.CostEstimate{Min: 25, Max: 35}, expectedRuntimeCost: 25, // +10 for casts from string to byte, +5 for byte comparisons
				},
				{
					comparableLiteral:     durationListLiteral,
					expectedEstimatedCost: checker.CostEstimate{Min: 20, Max: 20}, expectedRuntimeCost: 20, // +5 for numeric casts
				},
				{
					comparableLiteral:     timestampListLiteral,
					expectedEstimatedCost: checker.CostEstimate{Min: 20, Max: 20}, expectedRuntimeCost: 20, // +5 for casts
				},
			},
		},
	}
	for _, tc := range cases {
		for _, op := range tc.opts {
			for _, typ := range tc.costs {
				t.Run(typ.comparableLiteral+op, func(t *testing.T) {
					e := typ.comparableLiteral + op
					testCost(t, e, typ.expectedEstimatedCost, typ.expectedRuntimeCost)
				})
			}
		}
	}
}

func TestIndexOfCost(t *testing.T) {
	cases := []struct {
		opts  []string
		costs []comparableCost
	}{
		{
			opts: []string{".indexOf(%s)", ".lastIndexOf(%s)"},
			// 10 cost for the list declaration, the rest is the due to the function call
			costs: []comparableCost{
				{
					comparableLiteral: intListLiteral, param: "3",
					expectedEstimatedCost: checker.CostEstimate{Min: 15, Max: 15}, expectedRuntimeCost: 15,
				},
				{
					comparableLiteral: uintListLiteral, param: "uint(3)",
					expectedEstimatedCost: checker.CostEstimate{Min: 21, Max: 21}, expectedRuntimeCost: 21, // +5 for numeric casts
				},
				{
					comparableLiteral: doubleListLiteral, param: "3.0",
					expectedEstimatedCost: checker.CostEstimate{Min: 15, Max: 15}, expectedRuntimeCost: 15,
				},
				{
					comparableLiteral: boolListLiteral, param: "true",
					expectedEstimatedCost: checker.CostEstimate{Min: 15, Max: 15}, expectedRuntimeCost: 15,
				},
				{
					comparableLiteral: stringListLiteral, param: "'x'",
					expectedEstimatedCost: checker.CostEstimate{Min: 15, Max: 25}, expectedRuntimeCost: 15, // +5 for string comparisons
				},
				{
					comparableLiteral: bytesListLiteral, param: "bytes('x')",
					expectedEstimatedCost: checker.CostEstimate{Min: 26, Max: 36}, expectedRuntimeCost: 26, // +11 for casts from string to byte, +5 for byte comparisons
				},
				{
					comparableLiteral: durationListLiteral, param: "duration('3s')",
					expectedEstimatedCost: checker.CostEstimate{Min: 21, Max: 21}, expectedRuntimeCost: 21, // +6 for casts from duration to byte
				},
				{
					comparableLiteral: timestampListLiteral, param: "timestamp('2011-01-03T00:00:00.000+01:00')",
					expectedEstimatedCost: checker.CostEstimate{Min: 21, Max: 21}, expectedRuntimeCost: 21, // +6 for casts from timestamp to byte
				},

				// index of operations are also defined for strings
				{
					comparableLiteral: stringLiteral, param: "'123'",
					expectedEstimatedCost: checker.CostEstimate{Min: 5, Max: 5}, expectedRuntimeCost: 5,
				},
			},
		},
	}
	for _, tc := range cases {
		for _, op := range tc.opts {
			for _, typ := range tc.costs {
				opWithParam := fmt.Sprintf(op, typ.param)
				t.Run(typ.comparableLiteral+opWithParam, func(t *testing.T) {
					e := typ.comparableLiteral + opWithParam
					testCost(t, e, typ.expectedEstimatedCost, typ.expectedRuntimeCost)
				})
			}
		}
	}
}

func TestURLsCost(t *testing.T) {
	cases := []struct {
		ops                []string
		expectEsimatedCost checker.CostEstimate
		expectRuntimeCost  uint64
	}{
		{
			ops:                []string{".getScheme()", ".getHostname()", ".getHost()", ".getPort()", ".getEscapedPath()", ".getQuery()"},
			expectEsimatedCost: checker.CostEstimate{Min: 4, Max: 4},
			expectRuntimeCost:  4,
		},
		{
			ops:                []string{" == url('https:://kubernetes.io/')"},
			expectEsimatedCost: checker.CostEstimate{Min: 7, Max: 9},
			expectRuntimeCost:  7,
		},
		{
			ops:                []string{" == url('http://x.b')"},
			expectEsimatedCost: checker.CostEstimate{Min: 5, Max: 5},
			expectRuntimeCost:  5,
		},
	}

	for _, tc := range cases {
		for _, op := range tc.ops {
			t.Run("url."+op, func(t *testing.T) {
				testCost(t, "url('https:://kubernetes.io/')"+op, tc.expectEsimatedCost, tc.expectRuntimeCost)
			})
		}
	}
}

func TestIPCost(t *testing.T) {
	ipv4 := "ip('192.168.0.1')"
	ipv4BaseEstimatedCost := checker.CostEstimate{Min: 2, Max: 2}
	ipv4BaseRuntimeCost := uint64(2)

	ipv6 := "ip('2001:db8:3333:4444:5555:6666:7777:8888')"
	ipv6BaseEstimatedCost := checker.CostEstimate{Min: 4, Max: 4}
	ipv6BaseRuntimeCost := uint64(4)

	testCases := []struct {
		ops                []string
		expectEsimatedCost func(checker.CostEstimate) checker.CostEstimate
		expectRuntimeCost  func(uint64) uint64
	}{
		{
			// For just parsing the IP, the cost is expected to be the base.
			ops:                []string{""},
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate { return c },
			expectRuntimeCost:  func(c uint64) uint64 { return c },
		},
		{
			ops: []string{".family()", ".isUnspecified()", ".isLoopback()", ".isLinkLocalMulticast()", ".isLinkLocalUnicast()", ".isGlobalUnicast()"},
			// For most other operations, the cost is expected to be the base + 1.
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 1, Max: c.Max + 1}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 1 },
		},
		{
			ops: []string{" == ip('192.168.0.1')"},
			// For most other operations, the cost is expected to be the base + 1.
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return c.Add(ipv4BaseEstimatedCost).Add(checker.CostEstimate{Min: 1, Max: 1})
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + ipv4BaseRuntimeCost + 1 },
		},
	}

	for _, tc := range testCases {
		for _, op := range tc.ops {
			t.Run(ipv4+op, func(t *testing.T) {
				testCost(t, ipv4+op, tc.expectEsimatedCost(ipv4BaseEstimatedCost), tc.expectRuntimeCost(ipv4BaseRuntimeCost))
			})

			t.Run(ipv6+op, func(t *testing.T) {
				testCost(t, ipv6+op, tc.expectEsimatedCost(ipv6BaseEstimatedCost), tc.expectRuntimeCost(ipv6BaseRuntimeCost))
			})
		}
	}
}

func TestIPIsCanonicalCost(t *testing.T) {
	testCases := []struct {
		op                 string
		expectEsimatedCost checker.CostEstimate
		expectRuntimeCost  uint64
	}{
		{
			op:                 "ip.isCanonical('192.168.0.1')",
			expectEsimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:  3,
		},
		{
			op:                 "ip.isCanonical('2001:db8:3333:4444:5555:6666:7777:8888')",
			expectEsimatedCost: checker.CostEstimate{Min: 8, Max: 8},
			expectRuntimeCost:  8,
		},
		{
			op:                 "ip.isCanonical('2001:db8::abcd')",
			expectEsimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:  3,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.op, func(t *testing.T) {
			testCost(t, tc.op, tc.expectEsimatedCost, tc.expectRuntimeCost)
		})
	}
}

func TestCIDRCost(t *testing.T) {
	ipv4 := "cidr('192.168.0.0/16')"
	ipv4BaseEstimatedCost := checker.CostEstimate{Min: 2, Max: 2}
	ipv4BaseRuntimeCost := uint64(2)

	ipv6 := "cidr('2001:db8::/32')"
	ipv6BaseEstimatedCost := checker.CostEstimate{Min: 2, Max: 2}
	ipv6BaseRuntimeCost := uint64(2)

	type testCase struct {
		ops                []string
		expectEsimatedCost func(checker.CostEstimate) checker.CostEstimate
		expectRuntimeCost  func(uint64) uint64
	}

	cases := []testCase{
		{
			// For just parsing the IP, the cost is expected to be the base.
			ops:                []string{""},
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate { return c },
			expectRuntimeCost:  func(c uint64) uint64 { return c },
		},
		{
			ops: []string{".ip()", ".prefixLength()", ".masked()"},
			// For most other operations, the cost is expected to be the base + 1.
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 1, Max: c.Max + 1}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 1 },
		},
		{
			ops: []string{" == cidr('2001:db8::/32')"},
			// For most other operations, the cost is expected to be the base + 1.
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return c.Add(ipv6BaseEstimatedCost).Add(checker.CostEstimate{Min: 1, Max: 1})
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + ipv6BaseRuntimeCost + 1 },
		},
	}

	//nolint:gocritic
	ipv4Cases := append(cases, []testCase{
		{
			ops: []string{".containsCIDR(cidr('192.0.0.0/30'))"},
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 5, Max: c.Max + 9}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 5 },
		},
		{
			ops: []string{".containsCIDR(cidr('192.168.0.0/16'))"},
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 5, Max: c.Max + 9}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 5 },
		},
		{
			ops: []string{".containsCIDR('192.0.0.0/30')"},
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 5, Max: c.Max + 9}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 5 },
		},
		{
			ops: []string{".containsCIDR('192.168.0.0/16')"},
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 5, Max: c.Max + 9}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 5 },
		},
		{
			ops: []string{".containsIP(ip('192.0.0.1'))"},
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 2, Max: c.Max + 5}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 2 },
		},
		{
			ops: []string{".containsIP(ip('192.169.0.1'))"},
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 3, Max: c.Max + 6}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 3 },
		},
		{
			ops: []string{".containsIP(ip('192.169.169.250'))"},
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 3, Max: c.Max + 6}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 3 },
		},
		{
			ops: []string{".containsIP('192.0.0.1')"},
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 2, Max: c.Max + 5}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 2 },
		},
		{
			ops: []string{".containsIP('192.169.0.1')"},
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 3, Max: c.Max + 6}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 3 },
		},
	}...)

	//nolint:gocritic
	ipv6Cases := append(cases, []testCase{
		{
			ops: []string{".containsCIDR(cidr('2001:db8::/126'))"},
			// For operations like checking if an IP is in a CIDR, the cost is expected to higher.
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 5, Max: c.Max + 9}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 5 },
		},
		{
			ops: []string{".containsCIDR(cidr('2001:db8::/32'))"},
			// For operations like checking if an IP is in a CIDR, the cost is expected to higher.
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 5, Max: c.Max + 9}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 5 },
		},
		{
			ops: []string{".containsCIDR('2001:db8::/126')"},
			// For operations like checking if an IP is in a CIDR, the cost is expected to higher.
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 5, Max: c.Max + 9}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 5 },
		},
		{
			ops: []string{".containsCIDR('2001:db8::/32')"},
			// For operations like checking if an IP is in a CIDR, the cost is expected to higher.
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 5, Max: c.Max + 9}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 5 },
		},
		{
			ops: []string{".containsIP(ip('2001:db8:3333:4444:5555:6666:7777:8888'))"},
			// For operations like checking if an IP is in a CIDR, the cost is expected to higher.
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 5, Max: c.Max + 8}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 5 },
		},
		{
			ops: []string{".containsIP(ip('2001:db8::1'))"},
			// For operations like checking if an IP is in a CIDR, the cost is expected to higher.
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 3, Max: c.Max + 6}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 3 },
		},
		{
			ops: []string{".containsIP('2001:db8:3333:4444:5555:6666:7777:8888')"},
			// For operations like checking if an IP is in a CIDR, the cost is expected to higher.
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 5, Max: c.Max + 8}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 5 },
		},
		{
			ops: []string{".containsIP('2001:db8::1')"},
			// For operations like checking if an IP is in a CIDR, the cost is expected to higher.
			expectEsimatedCost: func(c checker.CostEstimate) checker.CostEstimate {
				return checker.CostEstimate{Min: c.Min + 3, Max: c.Max + 6}
			},
			expectRuntimeCost: func(c uint64) uint64 { return c + 3 },
		},
	}...)

	for _, tc := range ipv4Cases {
		for _, op := range tc.ops {
			t.Run(ipv4+op, func(t *testing.T) {
				testCost(t, ipv4+op, tc.expectEsimatedCost(ipv4BaseEstimatedCost), tc.expectRuntimeCost(ipv4BaseRuntimeCost))
			})
		}
	}

	for _, tc := range ipv6Cases {
		for _, op := range tc.ops {
			t.Run(ipv6+op, func(t *testing.T) {
				testCost(t, ipv6+op, tc.expectEsimatedCost(ipv6BaseEstimatedCost), tc.expectRuntimeCost(ipv6BaseRuntimeCost))
			})
		}
	}
}

func TestStringLibrary(t *testing.T) {
	cases := []struct {
		name               string
		expr               string
		expectEsimatedCost checker.CostEstimate
		expectRuntimeCost  uint64
	}{
		{
			name:               "lowerAscii",
			expr:               "'ABCDEFGHIJ abcdefghij'.lowerAscii()",
			expectEsimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:  3,
		},
		{
			name:               "lowerAsciiEquals",
			expr:               "'ABCDEFGHIJ abcdefghij'.lowerAscii() == 'abcdefghij ABCDEFGHIJ'.lowerAscii()",
			expectEsimatedCost: checker.CostEstimate{Min: 7, Max: 9},
			expectRuntimeCost:  9,
		},
		{
			name:               "upperAscii",
			expr:               "'ABCDEFGHIJ abcdefghij'.upperAscii()",
			expectEsimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:  3,
		},
		{
			name:               "upperAsciiEquals",
			expr:               "'ABCDEFGHIJ abcdefghij'.upperAscii() == 'abcdefghij ABCDEFGHIJ'.upperAscii()",
			expectEsimatedCost: checker.CostEstimate{Min: 7, Max: 9},
			expectRuntimeCost:  9,
		},
		{
			name:               "quote",
			expr:               "strings.quote('ABCDEFGHIJ abcdefghij')",
			expectEsimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:  3,
		},
		{
			name:               "quoteEquals",
			expr:               "strings.quote('ABCDEFGHIJ abcdefghij') == strings.quote('ABCDEFGHIJ abcdefghij')",
			expectEsimatedCost: checker.CostEstimate{Min: 7, Max: 11},
			expectRuntimeCost:  9,
		},
		{
			name:               "replace",
			expr:               "'abc 123 def 123'.replace('123', '456')",
			expectEsimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:  3,
		},
		{
			name:               "replace between all chars",
			expr:               "'abc 123 def 123'.replace('', 'x')",
			expectEsimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:  3,
		},
		{
			name:               "replace with empty",
			expr:               "'abc 123 def 123'.replace('123', '')",
			expectEsimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:  3,
		},
		{
			name:               "replace with limit",
			expr:               "'abc 123 def 123'.replace('123', '456', 1)",
			expectEsimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:  3,
		},
		{
			name:               "split",
			expr:               "'abc 123 def 123'.split(' ')",
			expectEsimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:  3,
		},
		{
			name:               "split with limit",
			expr:               "'abc 123 def 123'.split(' ', 1)",
			expectEsimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:  3,
		},
		{
			name:               "substring",
			expr:               "'abc 123 def 123'.substring(5)",
			expectEsimatedCost: checker.CostEstimate{Min: 2, Max: 2},
			expectRuntimeCost:  2,
		},
		{
			name:               "substring with end",
			expr:               "'abc 123 def 123'.substring(5, 8)",
			expectEsimatedCost: checker.CostEstimate{Min: 2, Max: 2},
			expectRuntimeCost:  2,
		},
		{
			name:               "trim",
			expr:               "'  abc 123 def 123  '.trim()",
			expectEsimatedCost: checker.CostEstimate{Min: 2, Max: 2},
			expectRuntimeCost:  2,
		},
		{
			name:               "join with separator",
			expr:               "['aa', 'bb', 'cc', 'd', 'e', 'f', 'g', 'h', 'i', 'j'].join(' ')",
			expectEsimatedCost: checker.CostEstimate{Min: 11, Max: 23},
			expectRuntimeCost:  15,
		},
		{
			name:               "join",
			expr:               "['aa', 'bb', 'cc', 'd', 'e', 'f', 'g', 'h', 'i', 'j'].join()",
			expectEsimatedCost: checker.CostEstimate{Min: 10, Max: 22},
			expectRuntimeCost:  13,
		},
		{
			name:               "find",
			expr:               "'abc 123 def 123'.find('123')",
			expectEsimatedCost: checker.CostEstimate{Min: 2, Max: 2},
			expectRuntimeCost:  2,
		},
		{
			name:               "findAll",
			expr:               "'abc 123 def 123'.findAll('123')",
			expectEsimatedCost: checker.CostEstimate{Min: 2, Max: 2},
			expectRuntimeCost:  2,
		},
		{
			name:               "findAll with limit",
			expr:               "'abc 123 def 123'.findAll('123', 1)",
			expectEsimatedCost: checker.CostEstimate{Min: 2, Max: 2},
			expectRuntimeCost:  2,
		},
		{
			name:               "jsonpatch.escapeKey",
			expr:               "jsonpatch.escapeKey('abc/def~ abc/def~')",
			expectEsimatedCost: checker.CostEstimate{Min: 2, Max: 2},
			expectRuntimeCost:  2,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testCost(t, tc.expr, tc.expectEsimatedCost, tc.expectRuntimeCost)
		})
	}
}

func TestAuthzLibrary(t *testing.T) {
	cases := []struct {
		name                string
		expr                string
		expectEstimatedCost checker.CostEstimate
		expectRuntimeCost   uint64
	}{
		{
			name:                "path",
			expr:                "authorizer.path('/healthz')",
			expectEstimatedCost: checker.CostEstimate{Min: 2, Max: 2},
			expectRuntimeCost:   2,
		},
		{
			name:                "resource",
			expr:                "authorizer.group('apps').resource('deployments').subresource('status').namespace('test').name('backend')",
			expectEstimatedCost: checker.CostEstimate{Min: 6, Max: 6},
			expectRuntimeCost:   6,
		},
		{
			name:                "fieldSelector",
			expr:                "authorizer.group('').resource('pods').fieldSelector('spec.nodeName=example-node-name.fully.qualified.domain.name.example.com')",
			expectEstimatedCost: checker.CostEstimate{Min: 1821, Max: 1821},
			expectRuntimeCost:   1821, // authorizer(1) + group(1) + resource(1) + fieldSelector(10 + ceil(71/2)*50=1800 + ceil(71*.1)=8)
		},
		{
			name:                "labelSelector",
			expr:                "authorizer.group('').resource('pods').labelSelector('spec.nodeName=example-node-name.fully.qualified.domain.name.example.com')",
			expectEstimatedCost: checker.CostEstimate{Min: 1821, Max: 1821},
			expectRuntimeCost:   1821, // authorizer(1) + group(1) + resource(1) + fieldSelector(10 + ceil(71/2)*50=1800 + ceil(71*.1)=8)
		},
		{
			name:                "path check allowed",
			expr:                "authorizer.path('/healthz').check('get').allowed()",
			expectEstimatedCost: checker.CostEstimate{Min: 350003, Max: 350003},
			expectRuntimeCost:   350003,
		},
		{
			name:                "resource check allowed",
			expr:                "authorizer.group('apps').resource('deployments').subresource('status').namespace('test').name('backend').check('create').allowed()",
			expectEstimatedCost: checker.CostEstimate{Min: 350007, Max: 350007},
			expectRuntimeCost:   350007,
		},
		{
			name:                "resource check reason",
			expr:                "authorizer.group('apps').resource('deployments').subresource('status').namespace('test').name('backend').check('create').allowed()",
			expectEstimatedCost: checker.CostEstimate{Min: 350007, Max: 350007},
			expectRuntimeCost:   350007,
		},
		{
			name:                "resource check errored",
			expr:                "authorizer.group('apps').resource('deployments').subresource('status').namespace('test').name('backend').check('create').errored()",
			expectEstimatedCost: checker.CostEstimate{Min: 350007, Max: 350007},
			expectRuntimeCost:   350007,
		},
		{
			name:                "resource check error",
			expr:                "authorizer.group('apps').resource('deployments').subresource('status').namespace('test').name('backend').check('create').error()",
			expectEstimatedCost: checker.CostEstimate{Min: 350007, Max: 350007},
			expectRuntimeCost:   350007,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testCost(t, tc.expr, tc.expectEstimatedCost, tc.expectRuntimeCost)
		})
	}
}

func TestQuantityCost(t *testing.T) {
	cases := []struct {
		name                string
		expr                string
		expectEstimatedCost checker.CostEstimate
		expectRuntimeCost   uint64
	}{
		{
			name:                "path",
			expr:                `quantity("12Mi")`,
			expectEstimatedCost: checker.CostEstimate{Min: 1, Max: 1},
			expectRuntimeCost:   1,
		},
		{
			name:                "isQuantity",
			expr:                `isQuantity("20")`,
			expectEstimatedCost: checker.CostEstimate{Min: 1, Max: 1},
			expectRuntimeCost:   1,
		},
		{
			name:                "isQuantity_megabytes",
			expr:                `isQuantity("20M")`,
			expectEstimatedCost: checker.CostEstimate{Min: 1, Max: 1},
			expectRuntimeCost:   1,
		},
		{
			name:                "equality_reflexivity",
			expr:                `quantity("200M") == quantity("200M")`,
			expectEstimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:   3,
		},
		{
			name:                "equality_symmetry",
			expr:                `quantity("200M") == quantity("0.2G") && quantity("0.2G") == quantity("200M")`,
			expectEstimatedCost: checker.CostEstimate{Min: 3, Max: 6},
			expectRuntimeCost:   6,
		},
		{
			name:                "equality_transitivity",
			expr:                `quantity("2M") == quantity("0.002G") && quantity("2000k") == quantity("2M") && quantity("0.002G") == quantity("2000k")`,
			expectEstimatedCost: checker.CostEstimate{Min: 3, Max: 9},
			expectRuntimeCost:   9,
		},
		{
			name:                "quantity_less",
			expr:                `quantity("50M").isLessThan(quantity("50Mi"))`,
			expectEstimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:   3,
		},
		{
			name:                "quantity_greater",
			expr:                `quantity("50Mi").isGreaterThan(quantity("50M"))`,
			expectEstimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:   3,
		},
		{
			name:                "compare_equal",
			expr:                `quantity("200M").compareTo(quantity("0.2G")) > 0`,
			expectEstimatedCost: checker.CostEstimate{Min: 4, Max: 4},
			expectRuntimeCost:   4,
		},
		{
			name:                "add_quantity",
			expr:                `quantity("50k").add(quantity("20")) == quantity("50.02k")`,
			expectEstimatedCost: checker.CostEstimate{Min: 5, Max: 5},
			expectRuntimeCost:   5,
		},
		{
			name:                "sub_quantity",
			expr:                `quantity("50k").sub(quantity("20")) == quantity("49.98k")`,
			expectEstimatedCost: checker.CostEstimate{Min: 5, Max: 5},
			expectRuntimeCost:   5,
		},
		{
			name:                "sub_int",
			expr:                `quantity("50k").sub(20) == quantity("49980")`,
			expectEstimatedCost: checker.CostEstimate{Min: 4, Max: 4},
			expectRuntimeCost:   4,
		},
		{
			name:                "arith_chain_1",
			expr:                `quantity("50k").add(20).sub(quantity("100k")).asInteger() > 0`,
			expectEstimatedCost: checker.CostEstimate{Min: 6, Max: 6},
			expectRuntimeCost:   6,
		},
		{
			name:                "arith_chain",
			expr:                `quantity("50k").add(20).sub(quantity("100k")).sub(-50000).asInteger() > 0`,
			expectEstimatedCost: checker.CostEstimate{Min: 7, Max: 7},
			expectRuntimeCost:   7,
		},
		{
			name:                "as_integer",
			expr:                `quantity("50k").asInteger() > 0`,
			expectEstimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:   3,
		},
		{
			name:                "is_integer",
			expr:                `quantity("50").isInteger()`,
			expectEstimatedCost: checker.CostEstimate{Min: 2, Max: 2},
			expectRuntimeCost:   2,
		},
		{
			name:                "as_float",
			expr:                `quantity("50.703k").asApproximateFloat() > 0.0`,
			expectEstimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:   3,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testCost(t, tc.expr, tc.expectEstimatedCost, tc.expectRuntimeCost)
		})
	}
}

func TestNameFormatCost(t *testing.T) {
	cases := []struct {
		name                string
		expr                string
		expectEstimatedCost checker.CostEstimate
		expectRuntimeCost   uint64
	}{
		{
			name:                "format.named",
			expr:                `format.named("dns1123subdomain")`,
			expectEstimatedCost: checker.CostEstimate{Min: 1, Max: 1},
			expectRuntimeCost:   1,
		},
		{
			name: "format.dns1123Subdomain.validate",
			expr: `format.named("dns1123Subdomain").value().validate("my-name")`,
			// Estimated cost doesnt know value at runtime so it is
			// using an estimated maximum regex length
			expectEstimatedCost: checker.CostEstimate{Min: 34, Max: 34},
			expectRuntimeCost:   17,
		},
		{
			name:                "format.dns1123label.validate",
			expr:                `format.named("dns1123Label").value().validate("my-name")`,
			expectEstimatedCost: checker.CostEstimate{Min: 34, Max: 34},
			expectRuntimeCost:   10,
		},
		{
			name:                "format.dns1123label.validate",
			expr:                `format.named("dns1123Label").value().validate("my-name")`,
			expectEstimatedCost: checker.CostEstimate{Min: 34, Max: 34},
			expectRuntimeCost:   10,
		},
		{
			name:                "format.dns1123label.validate",
			expr:                `format.named("dns1123Label").value() == format.named("dns1123Label").value()`,
			expectEstimatedCost: checker.CostEstimate{Min: 5, Max: 11},
			expectRuntimeCost:   5,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testCost(t, tc.expr, tc.expectEstimatedCost, tc.expectRuntimeCost)
		})
	}
}

func TestSetsCost(t *testing.T) {
	cases := []struct {
		name                string
		expr                string
		expectEstimatedCost checker.CostEstimate
		expectRuntimeCost   uint64
	}{
		{
			name:                "sets",
			expr:                `sets.contains([], [])`,
			expectEstimatedCost: checker.CostEstimate{Min: 21, Max: 21},
			expectRuntimeCost:   21,
		},
		{
			expr:                `sets.contains([1], [])`,
			expectEstimatedCost: checker.CostEstimate{Min: 21, Max: 21},
			expectRuntimeCost:   21,
		},
		{
			expr:                `sets.contains([1], [1])`,
			expectEstimatedCost: checker.CostEstimate{Min: 22, Max: 22},
			expectRuntimeCost:   22,
		},
		{
			expr:                `sets.contains([1], [1, 1])`,
			expectEstimatedCost: checker.CostEstimate{Min: 23, Max: 23},
			expectRuntimeCost:   23,
		},
		{
			expr:                `sets.contains([1, 1], [1])`,
			expectEstimatedCost: checker.CostEstimate{Min: 23, Max: 23},
			expectRuntimeCost:   23,
		},
		{
			expr:                `sets.contains([2, 1], [1])`,
			expectEstimatedCost: checker.CostEstimate{Min: 23, Max: 23},
			expectRuntimeCost:   23,
		},
		{
			expr:                `sets.contains([1, 2, 3, 4], [2, 3])`,
			expectEstimatedCost: checker.CostEstimate{Min: 29, Max: 29},
			expectRuntimeCost:   29,
		},
		{
			expr:                `sets.contains([1], [1.0, 1])`,
			expectEstimatedCost: checker.CostEstimate{Min: 23, Max: 23},
			expectRuntimeCost:   23,
		},
		{
			expr:                `sets.contains([1, 2], [2u, 2.0])`,
			expectEstimatedCost: checker.CostEstimate{Min: 25, Max: 25},
			expectRuntimeCost:   25,
		},
		{
			expr:                `sets.contains([1, 2u], [2, 2.0])`,
			expectEstimatedCost: checker.CostEstimate{Min: 25, Max: 25},
			expectRuntimeCost:   25,
		},
		{
			expr:                `sets.contains([1, 2.0, 3u], [1.0, 2u, 3])`,
			expectEstimatedCost: checker.CostEstimate{Min: 30, Max: 30},
			expectRuntimeCost:   30,
		},
		{
			expr: `sets.contains([[1], [2, 3]], [[2, 3.0]])`,
			// 10 for each list creation, top-level list sizes are 2, 1
			expectEstimatedCost: checker.CostEstimate{Min: 53, Max: 53},
			expectRuntimeCost:   53,
		},
		{
			expr:                `!sets.contains([1], [2])`,
			expectEstimatedCost: checker.CostEstimate{Min: 23, Max: 23},
			expectRuntimeCost:   23,
		},
		{
			expr:                `!sets.contains([1], [1, 2])`,
			expectEstimatedCost: checker.CostEstimate{Min: 24, Max: 24},
			expectRuntimeCost:   24,
		},
		{
			expr:                `!sets.contains([1], ["1", 1])`,
			expectEstimatedCost: checker.CostEstimate{Min: 24, Max: 24},
			expectRuntimeCost:   24,
		},
		{
			expr:                `!sets.contains([1], [1.1, 1u])`,
			expectEstimatedCost: checker.CostEstimate{Min: 24, Max: 24},
			expectRuntimeCost:   24,
		},

		// set equivalence (note the cost factor is higher as it's basically two contains checks)
		{
			expr:                `sets.equivalent([], [])`,
			expectEstimatedCost: checker.CostEstimate{Min: 21, Max: 21},
			expectRuntimeCost:   21,
		},
		{
			expr:                `sets.equivalent([1], [1])`,
			expectEstimatedCost: checker.CostEstimate{Min: 23, Max: 23},
			expectRuntimeCost:   23,
		},
		{
			expr:                `sets.equivalent([1], [1, 1])`,
			expectEstimatedCost: checker.CostEstimate{Min: 25, Max: 25},
			expectRuntimeCost:   25,
		},
		{
			expr:                `sets.equivalent([1, 1], [1])`,
			expectEstimatedCost: checker.CostEstimate{Min: 25, Max: 25},
			expectRuntimeCost:   25,
		},
		{
			expr:                `sets.equivalent([1], [1u, 1.0])`,
			expectEstimatedCost: checker.CostEstimate{Min: 25, Max: 25},
			expectRuntimeCost:   25,
		},
		{
			expr:                `sets.equivalent([1], [1u, 1.0])`,
			expectEstimatedCost: checker.CostEstimate{Min: 25, Max: 25},
			expectRuntimeCost:   25,
		},
		{
			expr:                `sets.equivalent([1, 2, 3], [3u, 2.0, 1])`,
			expectEstimatedCost: checker.CostEstimate{Min: 39, Max: 39},
			expectRuntimeCost:   39,
		},
		{
			expr:                `sets.equivalent([[1.0], [2, 3]], [[1], [2, 3.0]])`,
			expectEstimatedCost: checker.CostEstimate{Min: 69, Max: 69},
			expectRuntimeCost:   69,
		},
		{
			expr:                `!sets.equivalent([2, 1], [1])`,
			expectEstimatedCost: checker.CostEstimate{Min: 26, Max: 26},
			expectRuntimeCost:   26,
		},
		{
			expr:                `!sets.equivalent([1], [1, 2])`,
			expectEstimatedCost: checker.CostEstimate{Min: 26, Max: 26},
			expectRuntimeCost:   26,
		},
		{
			expr:                `!sets.equivalent([1, 2], [2u, 2, 2.0])`,
			expectEstimatedCost: checker.CostEstimate{Min: 34, Max: 34},
			expectRuntimeCost:   34,
		},
		{
			expr:                `!sets.equivalent([1, 2], [1u, 2, 2.3])`,
			expectEstimatedCost: checker.CostEstimate{Min: 34, Max: 34},
			expectRuntimeCost:   34,
		},
		{
			expr:                `sets.intersects([1], [1])`,
			expectEstimatedCost: checker.CostEstimate{Min: 22, Max: 22},
			expectRuntimeCost:   22,
		},
		{
			expr:                `sets.intersects([1], [1, 1])`,
			expectEstimatedCost: checker.CostEstimate{Min: 23, Max: 23},
			expectRuntimeCost:   23,
		},
		{
			expr:                `sets.intersects([1, 1], [1])`,
			expectEstimatedCost: checker.CostEstimate{Min: 23, Max: 23},
			expectRuntimeCost:   23,
		},
		{
			expr:                `sets.intersects([2, 1], [1])`,
			expectEstimatedCost: checker.CostEstimate{Min: 23, Max: 23},
			expectRuntimeCost:   23,
		},
		{
			expr:                `sets.intersects([1], [1, 2])`,
			expectEstimatedCost: checker.CostEstimate{Min: 23, Max: 23},
			expectRuntimeCost:   23,
		},
		{
			expr:                `sets.intersects([1], [1.0, 2])`,
			expectEstimatedCost: checker.CostEstimate{Min: 23, Max: 23},
			expectRuntimeCost:   23,
		},
		{
			expr:                `sets.intersects([1, 2], [2u, 2, 2.0])`,
			expectEstimatedCost: checker.CostEstimate{Min: 27, Max: 27},
			expectRuntimeCost:   27,
		},
		{
			expr:                `sets.intersects([1, 2], [1u, 2, 2.3])`,
			expectEstimatedCost: checker.CostEstimate{Min: 27, Max: 27},
			expectRuntimeCost:   27,
		},
		{
			expr:                `sets.intersects([[1], [2, 3]], [[1, 2], [2, 3.0]])`,
			expectEstimatedCost: checker.CostEstimate{Min: 65, Max: 65},
			expectRuntimeCost:   65,
		},
		{
			expr:                `!sets.intersects([], [])`,
			expectEstimatedCost: checker.CostEstimate{Min: 22, Max: 22},
			expectRuntimeCost:   22,
		},
		{
			expr:                `!sets.intersects([1], [])`,
			expectEstimatedCost: checker.CostEstimate{Min: 22, Max: 22},
			expectRuntimeCost:   22,
		},
		{
			expr:                `!sets.intersects([1], [2])`,
			expectEstimatedCost: checker.CostEstimate{Min: 23, Max: 23},
			expectRuntimeCost:   23,
		},
		{
			expr:                `!sets.intersects([1], ["1", 2])`,
			expectEstimatedCost: checker.CostEstimate{Min: 24, Max: 24},
			expectRuntimeCost:   24,
		},
		{
			expr:                `!sets.intersects([1], [1.1, 2u])`,
			expectEstimatedCost: checker.CostEstimate{Min: 24, Max: 24},
			expectRuntimeCost:   24,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testCost(t, tc.expr, tc.expectEstimatedCost, tc.expectRuntimeCost)
		})
	}
}

func TestSemverCost(t *testing.T) {
	cases := []struct {
		name                string
		expr                string
		expectEstimatedCost checker.CostEstimate
		expectRuntimeCost   uint64
	}{
		{
			name:                "semver",
			expr:                `semver("1.0.0")`,
			expectEstimatedCost: checker.CostEstimate{Min: 1, Max: 1},
			expectRuntimeCost:   1,
		},
		{
			name:                "semver long input",
			expr:                `semver("1234.56789012345.67890123456789")`,
			expectEstimatedCost: checker.CostEstimate{Min: 4, Max: 4},
			expectRuntimeCost:   4,
		},
		{
			name:                "isSemver",
			expr:                `isSemver("1.0.0")`,
			expectEstimatedCost: checker.CostEstimate{Min: 1, Max: 1},
			expectRuntimeCost:   1,
		},
		{
			name:                "isSemver long input",
			expr:                `isSemver("1234.56789012345.67890123456789")`,
			expectEstimatedCost: checker.CostEstimate{Min: 4, Max: 4},
			expectRuntimeCost:   4,
		},
		// major(), minor(), patch()
		{
			name:                "major",
			expr:                `semver("1.2.3").major()`,
			expectEstimatedCost: checker.CostEstimate{Min: 2, Max: 2},
			expectRuntimeCost:   2,
		},
		{
			name:                "minor",
			expr:                `semver("1.2.3").minor()`,
			expectEstimatedCost: checker.CostEstimate{Min: 2, Max: 2},
			expectRuntimeCost:   2,
		},
		{
			name:                "patch",
			expr:                `semver("1.2.3").patch()`,
			expectEstimatedCost: checker.CostEstimate{Min: 2, Max: 2},
			expectRuntimeCost:   2,
		},
		// isLessThan
		{
			name:                "isLessThan",
			expr:                `semver("1.0.0").isLessThan(semver("1.1.0"))`,
			expectEstimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:   3,
		},
		// isGreaterThan
		{
			name:                "isGreaterThan",
			expr:                `semver("1.1.0").isGreaterThan(semver("1.0.0"))`,
			expectEstimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:   3,
		},
		// compareTo
		{
			name:                "compareTo",
			expr:                `semver("1.0.0").compareTo(semver("1.2.3"))`,
			expectEstimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:   3,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testCost(t, tc.expr, tc.expectEstimatedCost, tc.expectRuntimeCost)
		})
	}
}

func TestTwoVariableComprehensionCost(t *testing.T) {
	cases := []struct {
		name                string
		expr                string
		expectEstimatedCost checker.CostEstimate
		expectRuntimeCost   uint64
	}{
		{
			name:                "map all",
			expr:                `{'a': 1, 'b': 2}.all(k, v, v > 0)`,
			expectEstimatedCost: checker.CostEstimate{Min: 37, Max: 41},
			expectRuntimeCost:   41,
		},
		{
			name:                "map exists",
			expr:                `{'a': 1, 'b': 2}.exists(k, v, v > 0)`,
			expectEstimatedCost: checker.CostEstimate{Min: 39, Max: 43},
			expectRuntimeCost:   40,
		},
		{
			name:                "map existsOne",
			expr:                `{'a': 1, 'b': 2}.existsOne(k, v, v > 0)`,
			expectEstimatedCost: checker.CostEstimate{Min: 38, Max: 40},
			expectRuntimeCost:   40,
		},
		{
			name:                "map transformMap",
			expr:                `{'a': 1, 'b': 2}.transformMap(k, v, v + 1)`,
			expectEstimatedCost: checker.CostEstimate{Min: 71, Max: 71},
			expectRuntimeCost:   71,
		},
		{
			name:                "map transformMap with filter",
			expr:                `{'a': 1, 'b': 2}.transformMap(k, v, v < 5, v + 1)`,
			expectEstimatedCost: checker.CostEstimate{Min: 67, Max: 75},
			expectRuntimeCost:   75,
		},
		{
			name:                "map transformMapEntry",
			expr:                `{'a': 1, 'b': 2}.transformMapEntry(k, v, {k: v + 1})`,
			expectEstimatedCost: checker.CostEstimate{Min: 131, Max: 131},
			expectRuntimeCost:   131,
		},
		{
			name:                "map transformMapEntry with filter",
			expr:                `{'a': 1, 'b': 2}.transformMapEntry(k, v, v < 5, {k: v + 1})`,
			expectEstimatedCost: checker.CostEstimate{Min: 67, Max: 135},
			expectRuntimeCost:   135,
		},

		{
			name:                "list all",
			expr:                `[1, 2].all(i, v, v > 0)`,
			expectEstimatedCost: checker.CostEstimate{Min: 17, Max: 21},
			expectRuntimeCost:   21,
		},
		{
			name:                "list exists",
			expr:                `[1, 2].exists(i, v, v > 0)`,
			expectEstimatedCost: checker.CostEstimate{Min: 19, Max: 23},
			expectRuntimeCost:   20,
		},
		{
			name:                "list existsOne",
			expr:                `[1, 2].existsOne(i, v, v > 0)`,
			expectEstimatedCost: checker.CostEstimate{Min: 18, Max: 20},
			expectRuntimeCost:   20,
		},
		{
			name:                "list transformList",
			expr:                `[1, 2].transformList(i, v, v + 1)`,
			expectEstimatedCost: checker.CostEstimate{Min: 49, Max: 49},
			expectRuntimeCost:   49,
		},
		{
			name:                "list transformList with filter",
			expr:                `[1, 2].transformList(i, v, v < 5, v + 1)`,
			expectEstimatedCost: checker.CostEstimate{Min: 27, Max: 53},
			expectRuntimeCost:   53,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testCost(t, tc.expr, tc.expectEstimatedCost, tc.expectRuntimeCost)
		})
	}
}

func testCost(t *testing.T, expr string, expectEsimatedCost checker.CostEstimate, expectRuntimeCost uint64) {
	originalPanicOnUnknown := panicOnUnknown
	panicOnUnknown = true
	t.Cleanup(func() { panicOnUnknown = originalPanicOnUnknown })

	est := &CostEstimator{SizeEstimator: &testCostEstimator{}}
	env, err := cel.NewEnv(
		ext.Strings(ext.StringsVersion(2)),
		URLs(),
		Regex(),
		Lists(),
		Authz(),
		AuthzSelectors(),
		Quantity(),
		ext.Sets(),
		IP(),
		CIDR(),
		Format(),
		JSONPatch(),
		cel.OptionalTypes(),
		// cel-go v0.17.7 introduced CostEstimatorOptions.
		// Previous the presence has a cost of 0 but cel fixed it to 1. We still set to 0 here to avoid breaking changes.
		cel.CostEstimatorOptions(checker.PresenceTestHasCost(false)),
		ext.TwoVarComprehensions(),
		SemverLib(SemverVersion(1)),
	)
	if err != nil {
		t.Fatalf("%v", err)
	}
	env, err = env.Extend(cel.Variable("authorizer", AuthorizerType))
	if err != nil {
		t.Fatalf("%v", err)
	}
	compiled, issues := env.Compile(expr)
	if len(issues.Errors()) > 0 {
		var errList []string
		for _, issue := range issues.Errors() {
			errList = append(errList, issue.ToDisplayString(common.NewTextSource(expr)))
		}
		t.Fatalf("%v", errList)
	}
	estCost, err := env.EstimateCost(compiled, est)
	if err != nil {
		t.Fatalf("%v", err)
	}
	if estCost.Min != expectEsimatedCost.Min || estCost.Max != expectEsimatedCost.Max {
		t.Errorf("Expected estimated cost of %d..%d but got %d..%d", expectEsimatedCost.Min, expectEsimatedCost.Max, estCost.Min, estCost.Max)
	}
	prog, err := env.Program(compiled, cel.CostTracking(est))
	if err != nil {
		t.Fatalf("%v", err)
	}
	_, details, err := prog.Eval(map[string]interface{}{"authorizer": NewAuthorizerVal(nil, alwaysAllowAuthorizer{})})
	if err != nil {
		t.Fatalf("%v", err)
	}
	cost := details.ActualCost()
	if *cost != expectRuntimeCost {
		t.Errorf("Expected cost of %d but got %d", expectRuntimeCost, *cost)
	}
}

func TestSize(t *testing.T) {
	exactSize := func(size int) checker.SizeEstimate {
		return checker.SizeEstimate{Min: uint64(size), Max: uint64(size)}
	}
	exactSizes := func(sizes ...int) []checker.SizeEstimate {
		results := make([]checker.SizeEstimate, len(sizes))
		for i, size := range sizes {
			results[i] = exactSize(size)
		}
		return results
	}
	cases := []struct {
		name       string
		function   string
		overload   string
		targetSize checker.SizeEstimate
		argSizes   []checker.SizeEstimate
		expectSize checker.SizeEstimate
	}{
		{
			name:       "replace empty with char",
			function:   "replace",
			targetSize: exactSize(3),     // e.g. abc
			argSizes:   exactSizes(0, 1), // e.g. replace "" with "_"
			expectSize: exactSize(7),     // e.g. _a_b_c_
		},
		{
			name:       "maybe replace char with empty",
			function:   "replace",
			targetSize: exactSize(3),
			argSizes:   exactSizes(1, 0),
			expectSize: checker.SizeEstimate{Min: 0, Max: 3},
		},
		{
			name:       "maybe replace repeated",
			function:   "replace",
			targetSize: exactSize(4),
			argSizes:   exactSizes(2, 4),
			expectSize: checker.SizeEstimate{Min: 4, Max: 8},
		},
		{
			name:       "maybe replace empty",
			function:   "replace",
			targetSize: exactSize(4),
			argSizes:   []checker.SizeEstimate{{Min: 0, Max: 1}, {Min: 0, Max: 2}},
			expectSize: checker.SizeEstimate{Min: 0, Max: 14}, // len(__a__a__a__a__) == 14
		},
		{
			name:       "replace non-empty size range, maybe larger",
			function:   "replace",
			targetSize: exactSize(4),
			argSizes:   []checker.SizeEstimate{{Min: 1, Max: 1}, {Min: 1, Max: 2}},
			expectSize: checker.SizeEstimate{Min: 4, Max: 8},
		},
		{
			name:       "replace non-empty size range, maybe smaller",
			function:   "replace",
			targetSize: exactSize(4),
			argSizes:   []checker.SizeEstimate{{Min: 1, Max: 2}, {Min: 1, Max: 1}},
			expectSize: checker.SizeEstimate{Min: 2, Max: 4},
		},
	}

	originalPanicOnUnknown := panicOnUnknown
	panicOnUnknown = true
	t.Cleanup(func() { panicOnUnknown = originalPanicOnUnknown })

	est := &CostEstimator{SizeEstimator: &testCostEstimator{}}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var targetNode checker.AstNode = testNode{size: tc.targetSize}
			argNodes := make([]checker.AstNode, len(tc.argSizes))
			for i, arg := range tc.argSizes {
				argNodes[i] = testNode{size: arg}
			}
			result := est.EstimateCallCost(tc.function, tc.overload, &targetNode, argNodes)
			if result.ResultSize == nil {
				t.Fatalf("Expected ResultSize but got none")
			}
			if *result.ResultSize != tc.expectSize {
				t.Fatalf("Expected %+v but got %+v", tc.expectSize, *result.ResultSize)
			}
		})
	}
}

// TestTypeEquality ensures that cost is tested for all custom types used by Kubernetes libraries.
func TestTypeEquality(t *testing.T) {
	examples := map[string]ref.Val{
		// Add example ref.Val's for custom types in Kubernetes here:
		"kubernetes.authorization.Authorizer":    authorizerVal{},
		"kubernetes.authorization.PathCheck":     pathCheckVal{},
		"kubernetes.authorization.GroupCheck":    groupCheckVal{},
		"kubernetes.authorization.ResourceCheck": resourceCheckVal{},
		"kubernetes.authorization.Decision":      decisionVal{},
		"kubernetes.URL":                         apiservercel.URL{},
		"kubernetes.Quantity":                    apiservercel.Quantity{},
		"net.IP":                                 apiservercel.IP{},
		"net.CIDR":                               apiservercel.CIDR{},
		"kubernetes.NamedFormat":                 apiservercel.Format{},
		"kubernetes.Semver":                      apiservercel.Semver{},
	}

	originalPanicOnUnknown := panicOnUnknown
	panicOnUnknown = true
	t.Cleanup(func() { panicOnUnknown = originalPanicOnUnknown })
	est := &CostEstimator{SizeEstimator: &testCostEstimator{}}

	for _, lib := range KnownLibraries() {
		for _, kt := range lib.Types() {
			t.Run(kt.TypeName(), func(t *testing.T) {
				typeNode := testNode{size: checker.SizeEstimate{Min: 10, Max: 100}, typ: kt}
				est.EstimateCallCost("_==_", "", nil, []checker.AstNode{typeNode, typeNode})
				ex, ok := examples[kt.TypeName()]
				if !ok {
					t.Errorf("missing example for type: %s", kt.TypeName())
				}
				est.CallCost("_==_", "", []ref.Val{ex, ex}, nil)
			})
		}
	}
}

type testNode struct {
	size checker.SizeEstimate
	typ  *types.Type
}

var _ checker.AstNode = (*testNode)(nil)

func (t testNode) Path() []string {
	return nil // not needed
}

func (t testNode) Type() *types.Type {
	return t.typ // not needed
}

func (t testNode) Expr() ast.Expr {
	return nil // not needed
}

func (t testNode) ComputedSize() *checker.SizeEstimate {
	return &t.size
}

type testCostEstimator struct {
}

func (t *testCostEstimator) EstimateSize(element checker.AstNode) *checker.SizeEstimate {
	expr, err := cel.TypeToExprType(element.Type())
	if err != nil {
		return nil
	}
	switch expr.GetPrimitive() {
	case exprpb.Type_STRING:
		return &checker.SizeEstimate{Min: 0, Max: 12}
	case exprpb.Type_BYTES:
		return &checker.SizeEstimate{Min: 0, Max: 12}
	}
	return nil
}

func (t *testCostEstimator) EstimateCallCost(function, overloadId string, target *checker.AstNode, args []checker.AstNode) *checker.CallEstimate {
	return nil
}

type alwaysAllowAuthorizer struct{}

func (f alwaysAllowAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	return authorizer.DecisionAllow, "", nil
}
