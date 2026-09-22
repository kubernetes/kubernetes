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
	"math/big"
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
	if err := limitRequestRatioConstraint("Container", "memory", resource.MustParse("2"),
		api.ResourceList{api.ResourceMemory: resource.MustParse("1e2147483647")},
		api.ResourceList{api.ResourceMemory: resource.MustParse("3e2147483647")}); err == nil {
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
	ratioOf := func(reqStr, limStr, maxRatioStr string) error {
		enforced := resource.MustParse(maxRatioStr)
		req := api.ResourceList{api.ResourceMemory: resource.MustParse(reqStr)}
		lim := api.ResourceList{api.ResourceMemory: resource.MustParse(limStr)}
		return limitRequestRatioConstraint("Container", "memory", enforced, req, lim)
	}
	testCases := []struct {
		desc      string
		req       string
		lim       string
		maxRatio  string
		wantError bool
	}{
		{"ratio at the maximum", "100", "200", "2", false},
		{"ratio above the maximum", "100", "300", "2", true},
		{"limit whose projection wraps negative", "1", "9223372036854775809", "2", true},
		{"equal request and limit at 2^63", "9223372036854775808", "9223372036854775808", "1", false},
		{"equal request and limit at 2^64", "18446744073709551616", "18446744073709551616", "1", false},
		{"ratio at the maximum past 2^63", "9223372036854775808", "18446744073709551616", "2", false},
		{"ratio above the maximum past 2^63", "9223372036854775808", "27670116110564327424", "2", true},
		// float64 cannot separate 2^54 from 2^54+1, so a ratio computed that way
		// reads as exactly 2 on the row below the last one.
		{"ratio one unit below the maximum", "9007199254740992", "18014398509481983", "2", false},
		{"ratio exactly at the maximum", "9007199254740992", "18014398509481984", "2", false},
		{"ratio one unit above the maximum", "9007199254740992", "18014398509481985", "2", true},
		{"ratio well under a suffixed maximum at the largest exponent", "1e2147483647", "1e2147483647", "1k", false},
		{"ratio at a suffixed maximum at the largest exponent", "1e2147483647", "1000e2147483647", "1k", false},
		{"ratio above a suffixed maximum at the largest exponent", "1e2147483647", "10000e2147483647", "1k", true},
		{"ratio above the maximum with both operands negative", "-1", "-3", "2", true},
	}
	for _, testCase := range testCases {
		if err := ratioOf(testCase.req, testCase.lim, testCase.maxRatio); (err != nil) != testCase.wantError {
			t.Errorf("%s: got error %v, wantError %v", testCase.desc, err, testCase.wantError)
		}
	}
}

func TestLimitRequestRatioConstraint(t *testing.T) {
	testCases := []struct {
		desc     string
		req      string
		lim      string
		maxRatio string
		wantErr  string
	}{
		{"ratio under the maximum", "1", "1", "2", ""},
		{"ratio at the maximum", "1", "2", "2", ""},
		{"ratio above the maximum", "1", "3", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 3.000000"},
		{"ratio at the maximum past float64", "1e400", "2e400", "2", ""},
		{"ratio above the maximum with the limit past float64", "1e308", "3e308", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 3.000000"},
		{"ratio above the maximum with both operands past float64", "1e400", "3e400", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 3.000000"},
		{"ratio under the maximum with both operands negative", "-1", "-1", "2", ""},
		{"ratio above the maximum with both operands negative", "-1", "-3", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 3.000000"},
		{"ratio above the maximum with only the request negative", "-1", "3", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 3.000000"},
		{"ratio above the maximum with only the limit negative", "1", "-3", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 3.000000"},
		{"ratio with operands at different scales", "1", "1500m", "1", "memory max limit to request ratio per Container is 1, but provided ratio is 1.500000"},
		{"ratio too large to print in full", "1", "1e2147483647", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 1.000000e2147483647"},
		{"ratio whose coefficients differ in width", "10000000", "1e50", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 1.000000e43"},
		{"same ratio with the request written as an exponent", "1e7", "1e50", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 1.000000e43"},
		{"small ratio whose operands are at far apart scales", "10000000", "3e50", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 3.000000e43"},
		{"small ratio that needs no exponent once normalized", "100000000000000000000000000000000000000000000000000", "3e50", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 3.000000"},
		{"ratio just below the rounding boundary", "10000000", "99999994e50", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 9.999999e50"},
		{"ratio whose mantissa carries to ten", "10000000", "99999999e50", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 1.000000e51"},
		{"ratio at the rounding tie with an odd digit kept", "10000000", "99999995e50", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 1.000000e51"},
		{"ratio at the rounding tie with an even digit kept", "10000000", "99999985e50", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 9.999998e50"},
		{"ratio whose coefficient quotient is below one", "5", "1e50", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 2.000000e49"},
		{"ratio whose limit coefficient is the wider of the two", "1", "12345678e50", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 1.234568e57"},
		{"ratio at the last scale gap the plain form covers", "1", "1e40", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 10000000000000000000000000000000000000000.000000"},
		{"ratio one scale gap past the plain form", "1", "1e41", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 1.000000e41"},
		{"ratio at the last negative scale gap the plain form covers", "1e31", "10000000000000000000000000000000000000001", "2", "memory max limit to request ratio per Container is 2, but provided ratio is 1000000000.000000"},
	}
	for _, testCase := range testCases {
		err := limitRequestRatioConstraint("Container", "memory", resource.MustParse(testCase.maxRatio),
			api.ResourceList{api.ResourceMemory: resource.MustParse(testCase.req)},
			api.ResourceList{api.ResourceMemory: resource.MustParse(testCase.lim)})
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
		small := testing.AllocsPerRun(10, func() { _ = check(memory("1e100")) })
		large := testing.AllocsPerRun(10, func() { _ = check(memory("1e10000000")) })
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

// A Quantity whose coefficient passes int64 carries an inf.Dec, which Neg
// writes through, so the caller's own value would change with it.
func TestRatioHelpersLeaveTheirArgumentsAlone(t *testing.T) {
	testCases := []struct {
		desc     string
		req      string
		lim      string
		maxRatio string
	}{
		{"both operands negative", "-18446744073709551616", "-55340232221128654848", "2"},
		{"only the request negative", "-18446744073709551616", "55340232221128654848", "2"},
		{"neither operand negative", "18446744073709551616", "55340232221128654848", "2"},
	}
	for _, testCase := range testCases {
		req, lim := resource.MustParse(testCase.req), resource.MustParse(testCase.lim)
		maxRatio := resource.MustParse(testCase.maxRatio)
		reqWas, limWas, maxWas := req.DeepCopy(), lim.DeepCopy(), maxRatio.DeepCopy()
		exceedsAllowed(lim, req, maxRatio)
		ratioString(lim, req)
		if req.Cmp(reqWas) != 0 || lim.Cmp(limWas) != 0 || maxRatio.Cmp(maxWas) != 0 {
			t.Errorf("%s: arguments changed, request %v, limit %v, ratio %v", testCase.desc, &req, &lim, &maxRatio)
		}
	}
}

func TestCmpScaled(t *testing.T) {
	mustBig := func(s string) *big.Int {
		v, ok := new(big.Int).SetString(s, 10)
		if !ok {
			t.Fatalf("bad number %q", s)
		}
		return v
	}
	testCases := []struct {
		desc string
		a    string
		sa   int64
		b    string
		sb   int64
		want int
	}{
		{"zero against a value", "0", 0, "5", 0, -1},
		{"a value against zero", "5", 0, "0", 0, 1},
		{"both zero", "0", 0, "0", 0, 0},
		{"zero against a value at a wider scale", "0", 0, "5", 10, -1},
		{"a value at a wider scale against zero", "5", 10, "0", 0, 1},
		{"a value against a negative", "5", 0, "-5", 0, 1},
		{"a larger magnitude", "5", 0, "5", 1, 1},
		{"a smaller magnitude", "5", 1, "5", 0, -1},
		{"equal at the same scale", "5", 0, "5", 0, 0},
		{"equal magnitudes decided by the coefficients", "6", 0, "5", 0, 1},
		{"equal magnitudes with b at the wider scale", "5", 0, "500", 2, 0},
		{"equal magnitudes with a at the wider scale", "500", 2, "5", 0, 0},
		{"equal magnitudes past inf.Scale", "1000", -2147483647, "1", -2147483650, 0},
		{"a below b past inf.Scale", "1", -2147483647, "1", -2147483650, -1},
	}
	for _, testCase := range testCases {
		a, b := mustBig(testCase.a), mustBig(testCase.b)
		aCopy, bCopy := new(big.Int).Set(a), new(big.Int).Set(b)
		if got := cmpScaled(a, testCase.sa, b, testCase.sb); got != testCase.want {
			t.Errorf("%s: got %d, want %d", testCase.desc, got, testCase.want)
		}
		if a.Cmp(aCopy) != 0 || b.Cmp(bCopy) != 0 {
			t.Errorf("%s: cmpScaled modified its arguments", testCase.desc)
		}
	}
}
