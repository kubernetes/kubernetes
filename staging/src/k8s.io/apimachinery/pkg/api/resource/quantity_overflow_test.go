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

package resource

import (
	"math"
	"testing"
)

// Baseline for the Quantity int64 overflow burndown (#141166): each row pins
// today's value and names what its fix should produce, so a later fix flips the
// pin and clears the note. Accessor and parse rows use a TODO per wrong field;
// archSplitCases use a group TODO; mutation rows use fixedEqual. Done when
// nothing is left flagged. Rows are
// split by assertion shape, not one fix per row (a row may carry TODOs from
// several fixes):
//   - accessorCases: a quantity read through every accessor.
//   - parseErrorCases: input that parses today but should be rejected.
//   - archSplitCases: ToDec rows whose int64 accessors are arch-dependent
//     (golang/go#45588: MinInt64 amd64, MaxInt64 arm64), so they are guarded.

const (
	// Shared TODO text. #141166 settles the overflow value: cap at the
	// MinInt64/MaxInt64 rail (matching strconv). The checked-accessor API shape
	// that carries the overflow bool is still being finalized.
	saturatePos = "want math.MaxInt64 once positive overflow saturates instead of wrapping (#141166)"
	saturateNeg = "want math.MinInt64 once negative overflow saturates instead of wrapping or reading zero (#141166)"
	// String drops the DecimalSI suffix above 10^18.
)

type accessorCase struct {
	name string
	load func() Quantity

	wantSign int

	wantValue int64
	valueTODO string

	wantMilli int64
	milliTODO string

	wantScaledKilo int64
	scaledTODO     string

	wantAsInt64   int64
	wantAsInt64OK bool
	asInt64TODO   string

	wantFloat float64
	floatTODO string

	wantString string
	stringTODO string

	// Value already sits at MinInt64/MaxInt64 as its real value, so a saturating
	// fix changes nothing here. Keeps it from looking unfinished.
	atInt64Boundary bool
}

func quantityAccessorCases() []accessorCase {
	return []accessorCase{
		// --- In range or capped at parse time: only the smaller accessors overflow. ---
		{
			name: "int64-max-parsed", load: func() Quantity { return MustParse("9223372036854775807") },
			wantSign:  1,
			wantValue: math.MaxInt64,
			wantMilli: -1000, milliTODO: saturatePos,
			wantScaledKilo: 9223372036854776,
			wantAsInt64:    math.MaxInt64, wantAsInt64OK: false, asInt64TODO: "want (math.MaxInt64, true) once #138076 parses 19-digit values via the int64 fast path",
			wantFloat:       9.223372036854776e+18,
			wantString:      "9223372036854775807",
			atInt64Boundary: true,
		},
		{
			name: "int64-max-constructed", load: func() Quantity { return *NewQuantity(math.MaxInt64, DecimalSI) },
			wantSign:  1,
			wantValue: math.MaxInt64,
			wantMilli: -1000, milliTODO: saturatePos,
			wantScaledKilo: 9223372036854776,
			wantAsInt64:    math.MaxInt64, wantAsInt64OK: true,
			wantFloat:       9.223372036854776e+18,
			wantString:      "9223372036854775807",
			atInt64Boundary: true,
		},
		{
			name: "int64-min-constructed", load: func() Quantity { return *NewQuantity(math.MinInt64, DecimalSI) },
			wantSign:  -1,
			wantValue: math.MinInt64,
			wantMilli: 0, milliTODO: saturateNeg,
			wantScaledKilo: -9223372036854776,
			wantAsInt64:    math.MinInt64, wantAsInt64OK: true,
			wantFloat:       -9.223372036854776e+18,
			wantString:      "-9223372036854775808",
			atInt64Boundary: true,
		},
		{
			name: "ten-to-18-parsed", load: func() Quantity { return MustParse("1E") },
			wantSign:  1,
			wantValue: 1000000000000000000,
			wantMilli: 0, milliTODO: saturatePos,
			wantScaledKilo: 1000000000000000,
			wantAsInt64:    1000000000000000000, wantAsInt64OK: true,
			wantFloat:  1e18,
			wantString: "1E",
		},
		{
			name: "binary-8Ei-caps-at-max", load: func() Quantity { return MustParse("8Ei") },
			wantSign:  1,
			wantValue: math.MaxInt64,
			wantMilli: -1000, milliTODO: saturatePos,
			wantScaledKilo: 9223372036854776,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:       9.223372036854776e+18,
			wantString:      "9223372036854775807",
			atInt64Boundary: true,
		},
		{
			name: "binary-negative-8Ei-caps-at-negative-max", load: func() Quantity { return MustParse("-8Ei") },
			wantSign:  -1,
			wantValue: -math.MaxInt64,
			wantMilli: 1000, milliTODO: saturateNeg,
			wantScaledKilo: -9223372036854776,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:  -9.223372036854776e+18,
			wantString: "-9223372036854775807",
		},
		{
			name: "binary-20Ei-caps-at-max", load: func() Quantity { return MustParse("20Ei") },
			wantSign:  1,
			wantValue: math.MaxInt64,
			wantMilli: -1000, milliTODO: saturatePos,
			wantScaledKilo: 9223372036854776,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:       9.223372036854776e+18,
			wantString:      "9223372036854775807",
			atInt64Boundary: true,
		},
		{
			name: "binary-negative-20Ei-caps-at-negative-max", load: func() Quantity { return MustParse("-20Ei") },
			wantSign:  -1,
			wantValue: -math.MaxInt64,
			wantMilli: 1000, milliTODO: saturateNeg,
			wantScaledKilo: -9223372036854776,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:  -9.223372036854776e+18,
			wantString: "-9223372036854775807",
		},

		// --- Positive magnitude over int64: Value and MilliValue wrap or truncate. ---
		{
			name: "two-to-63-parsed", load: func() Quantity { return MustParse("9223372036854775808") },
			wantSign:  1,
			wantValue: math.MinInt64, valueTODO: saturatePos,
			wantMilli: 0, milliTODO: saturatePos,
			wantScaledKilo: 9223372036854776, // 2^63/1000 rounds back under int64
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:  9.223372036854776e+18,
			wantString: "9223372036854775808",
		},
		{
			name: "two-to-64-parsed", load: func() Quantity { return MustParse("18446744073709551616") },
			wantSign:  1,
			wantValue: 0, valueTODO: saturatePos,
			wantMilli: 0, milliTODO: saturatePos,
			wantScaledKilo: 18446744073709552, // 2^64/1000 fits, so ScaledValue is correct here
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:  1.8446744073709552e+19,
			wantString: "18446744073709551616",
		},
		{
			name: "ten-to-20-parsed", load: func() Quantity { return MustParse("100E") },
			wantSign:  1,
			wantValue: 0, valueTODO: saturatePos,
			wantMilli: 0, milliTODO: saturatePos,
			wantScaledKilo: 100000000000000000,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:  1e20,
			wantString: "100E",
		},
		{
			name: "ten-to-21-parsed", load: func() Quantity { return MustParse("1000E") },
			wantSign:  1,
			wantValue: 0, valueTODO: saturatePos,
			wantMilli: 0, milliTODO: saturatePos,
			wantScaledKilo: 1000000000000000000,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:  1e21,
			wantString: "1e21",
		},
		{
			name: "five-times-ten-to-21-parsed", load: func() Quantity { return MustParse("5000E") },
			wantSign:  1,
			wantValue: 0, valueTODO: saturatePos,
			wantMilli: 0, milliTODO: saturatePos,
			wantScaledKilo: 5000000000000000000,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:  5e21,
			wantString: "5e21",
		},
		{
			name: "ten-to-100-parsed", load: func() Quantity { return MustParse("1e100") },
			wantSign:  1,
			wantValue: 0, valueTODO: saturatePos,
			wantMilli: 0, milliTODO: saturatePos,
			wantScaledKilo: 0, scaledTODO: saturatePos, // 1e97 also overflows
			wantAsInt64: 0, wantAsInt64OK: false,
			wantFloat:  1e100,
			wantString: "10e99", // lossless: 10e99 == 1e100
		},
		{
			name: "scaled-one-times-ten-to-21", load: func() Quantity { return *NewScaledQuantity(1, 21) },
			wantSign:  1,
			wantValue: 0, valueTODO: saturatePos,
			wantMilli: 0, milliTODO: saturatePos,
			wantScaledKilo: 1000000000000000000,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:  1e21,
			wantString: "1e21",
		},
		{
			// NewScaledQuantity(MaxInt64, 1) is MaxInt64*10; Value wraps to -10,
			// the accidental negative that validateBasicResource rejects today.
			name: "scaled-max-times-ten", load: func() Quantity { return *NewScaledQuantity(math.MaxInt64, 1) },
			wantSign:  1,
			wantValue: -10, valueTODO: saturatePos,
			wantMilli: 0, milliTODO: saturatePos,
			wantScaledKilo: 92233720368547759, // MaxInt64*10/1000 fits
			wantAsInt64:    -10, wantAsInt64OK: false,
			wantFloat:  9.223372036854776e+19,
			wantString: "92233720368547758070", // lossless: the full unscaled value
		},

		// --- Negative 19-digit values routed through the Dec backend: MilliValue still overflows. ---
		{
			// MinInt64 from a string is Dec-backed, unlike NewQuantity(MinInt64), so the
			// int64 accessors go through scaledValue.
			name: "int64-min-parsed", load: func() Quantity { return MustParse("-9223372036854775808") },
			wantSign:  -1,
			wantValue: math.MinInt64,
			wantMilli: 0, milliTODO: saturateNeg,
			wantScaledKilo: -9223372036854776,
			wantAsInt64:    math.MinInt64, wantAsInt64OK: false, asInt64TODO: "want (math.MinInt64, true) once #138076 parses -2^63 via the int64 fast path",
			wantFloat:       -9.223372036854776e+18,
			wantString:      "-9223372036854775808",
			atInt64Boundary: true,
		},
		{
			name: "int64-min-plus-one-parsed", load: func() Quantity { return MustParse("-9223372036854775807") },
			wantSign:  -1,
			wantValue: math.MinInt64 + 1,
			wantMilli: 1000, milliTODO: saturateNeg,
			wantScaledKilo: -9223372036854776,
			wantAsInt64:    math.MinInt64 + 1, wantAsInt64OK: false, asInt64TODO: "want (math.MinInt64 + 1, true) once #138076 parses this via the int64 fast path",
			wantFloat:  -9.223372036854776e+18,
			wantString: "-9223372036854775807",
		},
		// --- Negative values kept in the int64 backend at a large scale; Value and Milli overflow to zero. ---
		{
			name: "negative-ten-to-30-parsed", load: func() Quantity { return MustParse("-1e30") },
			wantSign:  -1,
			wantValue: 0, valueTODO: saturateNeg,
			wantMilli: 0, milliTODO: saturateNeg,
			wantScaledKilo: 0, scaledTODO: saturateNeg,
			wantAsInt64: 0, wantAsInt64OK: false,
			wantFloat:  -1e30,
			wantString: "-1e30",
		},
		{
			name: "negative-ten-to-20-parsed", load: func() Quantity { return MustParse("-100E") },
			wantSign:  -1,
			wantValue: 0, valueTODO: saturateNeg,
			wantMilli: 0, milliTODO: saturateNeg,
			wantScaledKilo: -100000000000000000,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:  -1e20,
			wantString: "-100E",
		},
		{
			name: "negative-ten-to-21-parsed", load: func() Quantity { return MustParse("-1000E") },
			wantSign:  -1,
			wantValue: 0, valueTODO: saturateNeg,
			wantMilli: 0, milliTODO: saturateNeg,
			wantScaledKilo: -1000000000000000000,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:  -1e21,
			wantString: "-1e21",
		},

		// --- Negative rounding: magnitude fits int64 and rounds away from zero. ---
		{
			name: "negative-9_5Gi", load: func() Quantity { return MustParse("-9.5Gi") },
			wantSign:       -1,
			wantValue:      -10200547328,
			wantMilli:      -10200547328000,
			wantScaledKilo: -10200548,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:  -1.0200547328e+10,
			wantString: "-9728Mi",
		},
		{
			name: "negative-9_5000000001Gi", load: func() Quantity { return MustParse("-9.5000000001Gi") },
			wantSign:       -1,
			wantValue:      -10200547329,
			wantMilli:      -10200547328108,
			wantScaledKilo: -10200548,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:  -1.0200547328107376e+10,
			wantString: "-10200547328107374183n",
		},

		// --- NewMilliQuantity at the boundary: correct today, just documented. ---
		{
			name: "new-milli-max", load: func() Quantity { return *NewMilliQuantity(math.MaxInt64, DecimalSI) },
			wantSign:       1,
			wantValue:      9223372036854776, // ceil(MaxInt64/1000), correct
			wantMilli:      math.MaxInt64,
			wantScaledKilo: 9223372036855,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:  9.223372036854776e+15,
			wantString: "9223372036854775807m",
		},
		{
			name: "new-milli-min", load: func() Quantity { return *NewMilliQuantity(math.MinInt64, DecimalSI) },
			wantSign:       -1,
			wantValue:      -9223372036854776,
			wantMilli:      math.MinInt64,
			wantScaledKilo: -9223372036855,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:  -9.223372036854776e+15,
			wantString: "-9223372036854775808m",
		},
		{
			// Dec-backed sibling of new-milli-min: ToDec routes Value and Kilo
			// through the big.Int path, so this proves away-from-zero rounding
			// reaches the Dec backend too.
			name: "new-milli-min-via-dec", load: func() Quantity { q := NewMilliQuantity(math.MinInt64, DecimalSI); q.ToDec(); return *q },
			wantSign:       -1,
			wantValue:      -9223372036854776,
			wantMilli:      math.MinInt64,
			wantScaledKilo: -9223372036855,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:  -9.223372036854776e+15,
			wantString: "-9223372036854775808m",
		},

		// --- Zero at a large scale: AsApproximateFloat64 does 0*Inf = NaN (#139893). ---
		{
			name: "zero-at-large-scale", load: func() Quantity { return *NewScaledQuantity(0, 500) },
			wantSign:       0,
			wantValue:      0,
			wantMilli:      0,
			wantScaledKilo: 0,
			wantAsInt64:    0, wantAsInt64OK: true,
			wantFloat: math.NaN(), floatTODO: "want 0 once #139893 guards zero before the Pow10 multiply",
			wantString: "0",
		},
		{
			// Negative scale, no NaN; AsInt64 declines (ok=false) though in range.
			name: "zero-at-large-negative-scale", load: func() Quantity { return *NewScaledQuantity(0, -500) },
			wantSign:       0,
			wantValue:      0,
			wantMilli:      0,
			wantScaledKilo: 0,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:  0,
			wantString: "0",
		},
		{
			// Dec-backed sibling: #139893's 0*Inf NaN happens on the inf.Dec path too.
			name: "zero-at-large-scale-via-dec", load: func() Quantity { q := NewScaledQuantity(0, 500); q.ToDec(); return *q },
			wantSign:       0,
			wantValue:      0,
			wantMilli:      0,
			wantScaledKilo: 0,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat: math.NaN(), floatTODO: "want 0 once #139893 guards zero before the Pow10 multiply",
			wantString: "0",
		},
		{
			name: "zero-at-large-negative-scale-via-dec", load: func() Quantity { q := NewScaledQuantity(0, -500); q.ToDec(); return *q },
			wantSign:       0,
			wantValue:      0,
			wantMilli:      0,
			wantScaledKilo: 0,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:  0,
			wantString: "0",
		},

		// Add promotes to inf.Dec and keeps the exact value; only the int64
		// projections wrap. An accessor case, not a mutation bug.
		{
			name: "max-plus-max", load: func() Quantity {
				q := NewQuantity(math.MaxInt64, DecimalSI)
				q.Add(*NewQuantity(math.MaxInt64, DecimalSI))
				return *q
			},
			wantSign:  1,
			wantValue: -2, valueTODO: saturatePos,
			wantMilli: -2000, milliTODO: saturatePos,
			wantScaledKilo: 18446744073709552,
			wantAsInt64:    0, wantAsInt64OK: false,
			wantFloat:  1.8446744073709552e+19,
			wantString: "18446744073709551614",
		},
	}
}

// parseErrorCase covers input ParseQuantity accepts today but should reject.
type parseErrorCase struct {
	name  string
	input string

	// Today ParseQuantity accepts it with the value below; the fix should error.
	wantParseError    bool
	wantValueIfParsed int64
	parseTODO         string
}

func quantityParseErrorCases() []parseErrorCase {
	return []parseErrorCase{
		{
			// exponent reduced mod 2^32 (4294967297 -> 1), so this parses to 1e1.
			name:              "exponent-over-int32",
			input:             "1e4294967297",
			wantParseError:    false,
			wantValueIfParsed: 10,
			parseTODO:         "ParseQuantity should reject this; the int32 exponent overflow silently yields 1e1 (#141166)",
		},
	}
}

// floatMatches handles the NaN and infinity cases; finite values are exact.
func floatMatches(got, want float64) bool {
	switch {
	case math.IsNaN(want):
		return math.IsNaN(got)
	case math.IsInf(want, 0):
		return math.IsInf(got, int(math.Copysign(1, want)))
	default:
		return got == want
	}
}

func assertAccessors(t *testing.T, tc accessorCase) {
	t.Helper()
	t.Run(tc.name+"/Sign", func(t *testing.T) {
		q := tc.load()
		if got := q.Sign(); got != tc.wantSign {
			t.Errorf("Sign() = %d, want %d", got, tc.wantSign)
		}
	})
	t.Run(tc.name+"/Value", func(t *testing.T) {
		q := tc.load()
		if got := q.Value(); got != tc.wantValue {
			t.Errorf("Value() = %d, want %d%s", got, tc.wantValue, todoSuffix(tc.valueTODO))
		}
		if tc.atInt64Boundary && (tc.valueTODO != "" || (tc.wantValue != math.MaxInt64 && tc.wantValue != math.MinInt64)) {
			t.Errorf("atInt64Boundary row must pin Value to MinInt64/MaxInt64 with no TODO; a saturating fix is a no-op here")
		}
	})
	t.Run(tc.name+"/MilliValue", func(t *testing.T) {
		q := tc.load()
		if got := q.MilliValue(); got != tc.wantMilli {
			t.Errorf("MilliValue() = %d, want %d%s", got, tc.wantMilli, todoSuffix(tc.milliTODO))
		}
	})
	t.Run(tc.name+"/ScaledValueKilo", func(t *testing.T) {
		q := tc.load()
		if got := q.ScaledValue(Kilo); got != tc.wantScaledKilo {
			t.Errorf("ScaledValue(Kilo) = %d, want %d%s", got, tc.wantScaledKilo, todoSuffix(tc.scaledTODO))
		}
	})
	t.Run(tc.name+"/ScaledValue0", func(t *testing.T) {
		q := tc.load()
		if got := q.ScaledValue(0); got != tc.wantValue {
			t.Errorf("ScaledValue(0) = %d, want %d%s", got, tc.wantValue, todoSuffix(tc.valueTODO))
		}
	})
	t.Run(tc.name+"/ScaledValueMilli", func(t *testing.T) {
		q := tc.load()
		if got := q.ScaledValue(Milli); got != tc.wantMilli {
			t.Errorf("ScaledValue(Milli) = %d, want %d%s", got, tc.wantMilli, todoSuffix(tc.milliTODO))
		}
	})
	t.Run(tc.name+"/AsInt64", func(t *testing.T) {
		q := tc.load()
		got, ok := q.AsInt64()
		if ok != tc.wantAsInt64OK {
			t.Errorf("AsInt64() ok = %t, want %t%s", ok, tc.wantAsInt64OK, todoSuffix(tc.asInt64TODO))
		}
		// ok=false means the fast int64 path declined (Dec-backed or inexact), not
		// overflow, and leaves the returned int64 unspecified; pin it only when ok.
		if ok && got != tc.wantAsInt64 {
			t.Errorf("AsInt64() value = %d, want %d", got, tc.wantAsInt64)
		}
	})
	t.Run(tc.name+"/AsApproximateFloat64", func(t *testing.T) {
		q := tc.load()
		if got := q.AsApproximateFloat64(); !floatMatches(got, tc.wantFloat) {
			t.Errorf("AsApproximateFloat64() = %v, want %v%s", got, tc.wantFloat, todoSuffix(tc.floatTODO))
		}
	})
	t.Run(tc.name+"/String", func(t *testing.T) {
		q := tc.load()
		if got := q.String(); got != tc.wantString {
			t.Errorf("String() = %q, want %q%s", got, tc.wantString, todoSuffix(tc.stringTODO))
		}
	})
}

func TestQuantityAccessorBaseline(t *testing.T) {
	for _, tc := range quantityAccessorCases() {
		assertAccessors(t, tc)
	}
}

func TestQuantityParseErrorBaseline(t *testing.T) {
	for _, tc := range quantityParseErrorCases() {
		t.Run(tc.name, func(t *testing.T) {
			q, err := ParseQuantity(tc.input)
			if gotErr := err != nil; gotErr != tc.wantParseError {
				t.Errorf("ParseQuantity(%q) error = %v, want error = %t%s", tc.input, err, tc.wantParseError, todoSuffix(tc.parseTODO))
			}
			if err == nil {
				if got := q.Value(); got != tc.wantValueIfParsed {
					t.Errorf("ParseQuantity(%q).Value() = %d, want %d", tc.input, got, tc.wantValueIfParsed)
				}
			}
		})
	}
}

// archSplitCase holds a ToDec row whose int64 accessors are arch-dependent
// (golang/go#45588). Sign and String are pinned; each int64 column lists its
// amd64 and arm64 result, and the check accepts either. The two are equal for a
// portable column, so a wrapper or scale that quietly changed would still fail.
//
// TODO: these rows resolve when #141166's checked accessors cap the result at
// the MinInt64/MaxInt64 rail and report overflow. golang/go#76264 might later
// make Go's float-to-int step saturate on every arch, but that alone is not
// enough: it adds no overflow bool and does not fix the int64 multiply after it
// (ten-to-20-parsed-via-dec's MilliValue is 100 * MaxInt64, which still wraps to
// -100). Collapse each pair once #141166 lands; until then they track the split.
type archSplitCase struct {
	name       string
	load       func() Quantity
	wantSign   int
	wantString string

	valueAmd64, valueArm64   int64
	milliAmd64, milliArm64   int64
	scaledAmd64, scaledArm64 int64 // ScaledValue(Kilo)

	wantAsInt64OK bool    // Dec-backed here, so AsInt64 declines (ok=false)
	wantFloat     float64 // portable (big.Float path)
}

func quantityArchSplitCases() []archSplitCase {
	return []archSplitCase{
		{
			name: "ten-to-100-parsed-via-dec", load: func() Quantity { q := MustParse("1e100"); q.ToDec(); return q },
			wantSign: 1, wantString: "10e99",
			valueAmd64: math.MinInt64, valueArm64: math.MaxInt64,
			milliAmd64: math.MinInt64, milliArm64: math.MaxInt64,
			scaledAmd64: math.MinInt64, scaledArm64: math.MaxInt64,
			wantAsInt64OK: false, wantFloat: 1e100,
		},
		{
			name: "ten-to-19-parsed-via-dec", load: func() Quantity { q := MustParse("1e19"); q.ToDec(); return q },
			wantSign: 1, wantString: "10e18",
			valueAmd64: math.MinInt64, valueArm64: math.MaxInt64,
			milliAmd64: math.MinInt64, milliArm64: math.MaxInt64,
			scaledAmd64: 10000000000000000, scaledArm64: 10000000000000000, // Pow10(16) fits, portable
			wantAsInt64OK: false, wantFloat: 1e19,
		},
		{
			// Value uses Pow10(18) (fits), so its wrap is portable; MilliValue uses
			// Pow10(21) and lands on 0 (amd64) or -100 (arm64).
			name: "ten-to-20-parsed-via-dec", load: func() Quantity { q := MustParse("100E"); q.ToDec(); return q },
			wantSign: 1, wantString: "100E",
			valueAmd64: 7766279631452241920, valueArm64: 7766279631452241920,
			milliAmd64: 0, milliArm64: -100,
			scaledAmd64: 100000000000000000, scaledArm64: 100000000000000000, // 1e17
			wantAsInt64OK: false, wantFloat: 1e20,
		},
	}
}

// archOneOf accepts either platform's result (golang/go#45588); the two are
// equal for a portable column, so it still catches an unexpected value.
func archOneOf(t *testing.T, label string, got, amd64, arm64 int64) {
	t.Helper()
	if got != amd64 && got != arm64 {
		t.Errorf("%s = %d, want %d (amd64) or %d (arm64) per golang/go#45588", label, got, amd64, arm64)
	}
}

func TestQuantityArchSplitBaseline(t *testing.T) {
	for _, tc := range quantityArchSplitCases() {
		t.Run(tc.name+"/Sign", func(t *testing.T) {
			q := tc.load()
			if got := q.Sign(); got != tc.wantSign {
				t.Errorf("Sign() = %d, want %d", got, tc.wantSign)
			}
		})
		t.Run(tc.name+"/String", func(t *testing.T) {
			q := tc.load()
			if got := q.String(); got != tc.wantString {
				t.Errorf("String() = %q, want %q", got, tc.wantString)
			}
		})
		t.Run(tc.name+"/Value", func(t *testing.T) {
			q := tc.load()
			archOneOf(t, "Value()", q.Value(), tc.valueAmd64, tc.valueArm64)
		})
		t.Run(tc.name+"/ScaledValue0", func(t *testing.T) {
			q := tc.load()
			archOneOf(t, "ScaledValue(0)", q.ScaledValue(0), tc.valueAmd64, tc.valueArm64)
		})
		t.Run(tc.name+"/MilliValue", func(t *testing.T) {
			q := tc.load()
			archOneOf(t, "MilliValue()", q.MilliValue(), tc.milliAmd64, tc.milliArm64)
		})
		t.Run(tc.name+"/ScaledValueMilli", func(t *testing.T) {
			q := tc.load()
			archOneOf(t, "ScaledValue(Milli)", q.ScaledValue(Milli), tc.milliAmd64, tc.milliArm64)
		})
		t.Run(tc.name+"/ScaledValueKilo", func(t *testing.T) {
			q := tc.load()
			archOneOf(t, "ScaledValue(Kilo)", q.ScaledValue(Kilo), tc.scaledAmd64, tc.scaledArm64)
		})
		t.Run(tc.name+"/AsInt64", func(t *testing.T) {
			q := tc.load()
			// ok=false leaves the int64 unspecified (see assertAccessors); pin only ok.
			if _, ok := q.AsInt64(); ok != tc.wantAsInt64OK {
				t.Errorf("AsInt64() ok = %t, want %t", ok, tc.wantAsInt64OK)
			}
		})
		t.Run(tc.name+"/AsApproximateFloat64", func(t *testing.T) {
			q := tc.load()
			if got := q.AsApproximateFloat64(); !floatMatches(got, tc.wantFloat) {
				t.Errorf("AsApproximateFloat64() = %v, want %v", got, tc.wantFloat)
			}
		})
	}
}

// todoSuffix appends the pending correction to a failure message so a diff that
// closes a gap sees exactly what to change the expectation to.
func todoSuffix(todo string) string {
	if todo == "" {
		return ""
	}
	return " (pinned to today's value; " + todo + ")"
}
