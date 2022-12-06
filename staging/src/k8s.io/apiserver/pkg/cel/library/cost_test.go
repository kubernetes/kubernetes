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
	"fmt"
	"testing"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker"
	"github.com/google/cel-go/ext"
	expr "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
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
	}

	for _, tc := range cases {
		for _, op := range tc.ops {
			t.Run("url."+op, func(t *testing.T) {
				testCost(t, "url('https:://kubernetes.io/')"+op, tc.expectEsimatedCost, tc.expectRuntimeCost)
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
			name:               "upperAscii",
			expr:               "'ABCDEFGHIJ abcdefghij'.upperAscii()",
			expectEsimatedCost: checker.CostEstimate{Min: 3, Max: 3},
			expectRuntimeCost:  3,
		},
		{
			name:               "replace",
			expr:               "'abc 123 def 123'.replace('123', '456')",
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
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			testCost(t, tc.expr, tc.expectEsimatedCost, tc.expectRuntimeCost)
		})
	}
}

func testCost(t *testing.T, expr string, expectEsimatedCost checker.CostEstimate, expectRuntimeCost uint64) {
	est := &CostEstimator{SizeEstimator: &testCostEstimator{}}
	env, err := cel.NewEnv(append(k8sExtensionLibs, ext.Strings())...)
	if err != nil {
		t.Fatalf("%v", err)
	}
	compiled, issues := env.Compile(expr)
	if len(issues.Errors()) > 0 {
		t.Fatalf("%v", issues.Errors())
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
	_, details, err := prog.Eval(map[string]interface{}{})
	if err != nil {
		t.Fatalf("%v", err)
	}
	cost := details.ActualCost()
	if *cost != expectRuntimeCost {
		t.Errorf("Expected cost of %d but got %d", expectRuntimeCost, *cost)
	}
}

type testCostEstimator struct {
}

func (t *testCostEstimator) EstimateSize(element checker.AstNode) *checker.SizeEstimate {
	switch t := element.Type().TypeKind.(type) {
	case *expr.Type_Primitive:
		switch t.Primitive {
		case expr.Type_STRING:
			return &checker.SizeEstimate{Min: 0, Max: 12}
		case expr.Type_BYTES:
			return &checker.SizeEstimate{Min: 0, Max: 12}
		}
	}
	return nil
}

func (t *testCostEstimator) EstimateCallCost(function, overloadId string, target *checker.AstNode, args []checker.AstNode) *checker.CallEstimate {
	return nil
}
