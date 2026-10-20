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

package limitranger

import (
	"math"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// A quantity past the int64 range wraps through Value()/MilliValue(): 2^64
// projects to 0 and 2^63 to a negative number. The constraints used to compare
// those projections, so an oversized limit or request slipped past the range
// and the LimitRange stopped being enforced. The exact comparison holds.
func TestConstraintsAreOverflowSafe(t *testing.T) {
	huge := resource.MustParse("18446744073709551616") // 2^64
	normal := resource.MustParse("1000")
	small := resource.MustParse("1")

	if err := maxConstraint("Container", "memory", normal, api.ResourceList{}, api.ResourceList{api.ResourceMemory: huge}); err == nil {
		t.Error("maxConstraint admitted a limit above the max")
	}
	if err := maxRequestConstraint("Container", "memory", normal, api.ResourceList{api.ResourceMemory: huge}); err == nil {
		t.Error("maxRequestConstraint admitted a request above the max")
	}
	if err := minConstraint("Container", "memory", huge, api.ResourceList{api.ResourceMemory: small}, api.ResourceList{}); err == nil {
		t.Error("minConstraint admitted a request below the min")
	}
	if err := limitRequestRatioConstraint("Container", "memory", normal, api.ResourceList{api.ResourceMemory: small}, api.ResourceList{api.ResourceMemory: huge}); err == nil {
		t.Error("limitRequestRatioConstraint admitted a limit-to-request ratio above the max")
	}

	// The ratio path forms request*ratio, whose summed scale can fall past
	// inf.Scale; a 3:1 ratio at an extreme exponent must still be rejected.
	// These operands are capped when parsed, so we load them directly to keep
	// testing this.
	if err := limitRequestRatioConstraint("Container", "memory", resource.MustParse("2"),
		api.ResourceList{api.ResourceMemory: *resource.NewScaledQuantity(1, math.MaxInt32)},
		api.ResourceList{api.ResourceMemory: *resource.NewScaledQuantity(3, math.MaxInt32)}); err == nil {
		t.Error("limitRequestRatioConstraint admitted a 3:1 ratio above the 2:1 max at an extreme scale")
	}

	// A within-limit value must still pass, so the exact comparison did not
	// start rejecting ordinary requests.
	if err := maxConstraint("Container", "memory", normal, api.ResourceList{}, api.ResourceList{api.ResourceMemory: small}); err != nil {
		t.Errorf("maxConstraint rejected a within-limit value: %v", err)
	}
}

func TestLimitRangerRatioIsExact(t *testing.T) {
	// The ratio came from Value(), which wraps. A limit of 2^63+1 read as a
	// small negative, so a ratio of 9.2e18 looked like it was under the maximum
	// and was admitted. Two operands that both collapse to zero went the other
	// way, and a ratio of exactly 1 was rejected as "limit is 0".
	ratioOf := func(reqQuantity, limQuantity resource.Quantity, maxRatioStr string) error {
		enforced := resource.MustParse(maxRatioStr)
		req := api.ResourceList{api.ResourceMemory: reqQuantity}
		lim := api.ResourceList{api.ResourceMemory: limQuantity}
		return limitRequestRatioConstraint("Container", "memory", enforced, req, lim)
	}
	testCases := []struct {
		desc      string
		req       string
		reqVal    *resource.Quantity // if set, use this value instead of parsing the req string
		lim       string
		limVal    *resource.Quantity // if set, use this value instead of parsing the lim string
		maxRatio  string
		wantError bool
	}{
		{desc: "ratio at the maximum", req: "100", lim: "200", maxRatio: "2"},
		{desc: "ratio above the maximum", req: "100", lim: "300", maxRatio: "2", wantError: true},
		{desc: "limit whose projection wraps negative", req: "1", lim: "9223372036854775809", maxRatio: "2", wantError: true},
		{desc: "equal request and limit at 2^63", req: "9223372036854775808", lim: "9223372036854775808", maxRatio: "1"},
		{desc: "equal request and limit at 2^64", req: "18446744073709551616", lim: "18446744073709551616", maxRatio: "1"},
		{desc: "ratio at the maximum past 2^63", req: "9223372036854775808", lim: "18446744073709551616", maxRatio: "2"},
		{desc: "ratio above the maximum past 2^63", req: "9223372036854775808", lim: "27670116110564327424", maxRatio: "2", wantError: true},
		// float64 cannot separate 2^54 from 2^54+1, so a ratio computed that way
		// reads as exactly 2 on the row below the last one.
		{desc: "ratio one unit below the maximum", req: "9007199254740992", lim: "18014398509481983", maxRatio: "2"},
		{desc: "ratio exactly at the maximum", req: "9007199254740992", lim: "18014398509481984", maxRatio: "2"},
		{desc: "ratio one unit above the maximum", req: "9007199254740992", lim: "18014398509481985", maxRatio: "2", wantError: true},
		// Operands at the largest exponent are capped when parsed, so we load them directly to test how the ratio check handles them.
		{desc: "ratio at a suffixed maximum at the largest exponent", reqVal: resource.NewScaledQuantity(1, math.MaxInt32), limVal: resource.NewScaledQuantity(1000, math.MaxInt32), maxRatio: "1k"},
		// TODO(#141166): A 10000:1 ratio exceeds max 1000, so wantError must be true.
		{desc: "ratio above a suffixed maximum at the largest exponent", reqVal: resource.NewScaledQuantity(1, math.MaxInt32), limVal: resource.NewScaledQuantity(10000, math.MaxInt32), maxRatio: "1k"},
		// TODO(#141166): Ratio 3 exceeds max 2, so wantError must be true.
		{desc: "ratio above the maximum with both operands negative", req: "-1", lim: "-3", maxRatio: "2"},
	}
	// quantity returns val if it is set, and otherwise parses s.
	quantity := func(s string, val *resource.Quantity) resource.Quantity {
		if val != nil {
			return *val
		}
		return resource.MustParse(s)
	}
	for _, testCase := range testCases {
		req, lim := quantity(testCase.req, testCase.reqVal), quantity(testCase.lim, testCase.limVal)
		if err := ratioOf(req, lim, testCase.maxRatio); (err != nil) != testCase.wantError {
			t.Errorf("%s: got error %v, wantError %v", testCase.desc, err, testCase.wantError)
		}
	}
}

func TestLimitRequestRatioConstraint(t *testing.T) {
	testCases := []struct {
		desc     string
		req      string
		reqVal   *resource.Quantity // if set, use this value instead of parsing the req string
		lim      string
		limVal   *resource.Quantity // if set, use this value instead of parsing the lim string
		maxRatio string
		wantErr  string
	}{
		{desc: "ratio under the maximum", req: "1", lim: "1", maxRatio: "2"},
		{desc: "ratio at the maximum", req: "1", lim: "2", maxRatio: "2"},
		{desc: "ratio above the maximum", req: "1", lim: "3", maxRatio: "2", wantErr: "memory max limit to request ratio per Container is 2, but provided ratio is 3.000000"},
		// Operands past float64 are capped when parsed, so we load them directly to test how the ratio check handles them.
		{desc: "ratio at the maximum past float64", reqVal: resource.NewScaledQuantity(1, 400), limVal: resource.NewScaledQuantity(2, 400), maxRatio: "2"},
		// TODO(#141166): The message must show the exact ratio 3.
		{desc: "ratio above the maximum with the limit past float64", req: "1e308", limVal: resource.NewScaledQuantity(3, 308), maxRatio: "2", wantErr: "memory max limit to request ratio per Container is 2, but provided ratio is +Inf"},
		// TODO(#141166): The message must show the exact ratio 3.
		{desc: "ratio above the maximum with both operands past float64", reqVal: resource.NewScaledQuantity(1, 400), limVal: resource.NewScaledQuantity(3, 400), maxRatio: "2", wantErr: "memory max limit to request ratio per Container is 2, but provided ratio is NaN"},
		// TODO(#141166): A negative pair with ratio 1 must not fail the ratio check. Expect no error, or an error about negative values.
		{desc: "ratio under the maximum with both operands negative", req: "-1", lim: "-1", maxRatio: "2", wantErr: "memory max limit to request ratio per Container is 2, but provided ratio is 1.000000"},
	}
	// quantity returns val if it is set, and otherwise parses s.
	quantity := func(s string, val *resource.Quantity) resource.Quantity {
		if val != nil {
			return *val
		}
		return resource.MustParse(s)
	}
	for _, testCase := range testCases {
		err := limitRequestRatioConstraint("Container", "memory", resource.MustParse(testCase.maxRatio),
			api.ResourceList{api.ResourceMemory: quantity(testCase.req, testCase.reqVal)},
			api.ResourceList{api.ResourceMemory: quantity(testCase.lim, testCase.limVal)})
		gotErr := ""
		if err != nil {
			gotErr = err.Error()
		}
		if gotErr != testCase.wantErr {
			t.Errorf("%s: got error %q, wantErr %q", testCase.desc, gotErr, testCase.wantErr)
		}
	}
}

// Quantity.Cmp hands a decimal-backed operand to inf.Dec.Cmp, which aligns the
// two scales by writing their difference out in digits. The exponent that sets
// that difference comes from the request, so the constraints have to answer
// without reading it. Allocation count is a proxy for that, not a bound on time
// or bytes; it is here because it is deterministic enough to fail a regression.
func TestConstraintAllocationsDoNotGrowWithTheExponent(t *testing.T) {
	enforced := resource.MustParse("1Gi")
	memory := func(value string) api.ResourceList {
		return api.ResourceList{api.ResourceMemory: resource.MustParse(value)}
	}
	checks := map[string]func(api.ResourceList) error{
		"min request": func(l api.ResourceList) error {
			return minConstraint("Container", "memory", enforced, l, api.ResourceList{})
		},
		"min limit": func(l api.ResourceList) error {
			return minConstraint("Container", "memory", enforced, memory("1Gi"), l)
		},
		"max limit": func(l api.ResourceList) error {
			return maxConstraint("Container", "memory", enforced, api.ResourceList{}, l)
		},
		"max request": func(l api.ResourceList) error {
			return maxConstraint("Container", "memory", enforced, l, memory("1Gi"))
		},
		"max request only": func(l api.ResourceList) error {
			return maxRequestConstraint("Container", "memory", enforced, l)
		},
	}
	// The two counts are equal without -race and differ by at most one with it,
	// while reading the exponent adds thirty or more between these two values.
	const slack = 8
	for desc, check := range checks {
		// 1e10000000 is capped when parsed, so we load it directly to keep testing
		// the checks at a large exponent. 1e100 is loaded the same way, so any
		// difference between the two counts comes from the check.
		small := testing.AllocsPerRun(10, func() {
			_ = check(api.ResourceList{api.ResourceMemory: *resource.NewScaledQuantity(1, 100)})
		})
		large := testing.AllocsPerRun(10, func() {
			_ = check(api.ResourceList{api.ResourceMemory: *resource.NewScaledQuantity(1, 10000000)})
		})
		if large > small+slack {
			t.Errorf("%s: %v allocations at 1e100 and %v at 1e10000000, so the comparison is reading the exponent", desc, small, large)
		}
	}
}

// The old comparison projected request, limit and enforced together, taking
// milli units only when all three fit MaxMilliValue and whole units otherwise.
// That made a verdict depend on the other operands and on whether Value()
// happened to collapse one of them, so it is not a boundary this change can
// keep. These four inputs are ordinary decimals that the projection accepted
// and the exact comparison rejects.
func TestConstraintsRejectValuesTheProjectionRounded(t *testing.T) {
	memory := func(value string) api.ResourceList {
		return api.ResourceList{api.ResourceMemory: resource.MustParse(value)}
	}
	testCases := []struct {
		desc  string
		check func() error
	}{
		{"request a fraction below the minimum", func() error {
			return minConstraint("Container", "memory", resource.MustParse("1"), memory("0.9999"), api.ResourceList{})
		}},
		{"limit a fraction above the maximum", func() error {
			return maxConstraint("Container", "memory", resource.MustParse("1.0001"), api.ResourceList{}, memory("1.0009"))
		}},
		{"request a fraction above the maximum", func() error {
			return maxRequestConstraint("Container", "memory", resource.MustParse("1.0001"), memory("1.0009"))
		}},
		{"ratio a fraction above the maximum", func() error {
			return limitRequestRatioConstraint("Container", "memory", resource.MustParse("2"), memory("1.0001"), memory("2.001"))
		}},
	}
	for _, testCase := range testCases {
		if err := testCase.check(); err == nil {
			t.Errorf("%s: expected rejection", testCase.desc)
		}
	}
}
