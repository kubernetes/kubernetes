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
	"encoding/hex"
	"encoding/json"
	"math"
	"testing"
)

// Golden serialization corpus for the Quantity burndown (#141166).
//
// The round-trip tests in quantity_test.go (TestJSON, TestQuantityRoundtripCBOR)
// fuzz a Quantity, encode it, decode it and compare with Cmp. That asserts the
// value survives, not the form it travels in, so a change to canonicalization
// passes them untouched. The serialized form is API-visible: MarshalJSON is what
// writes a PVC's storage request into etcd. This file pins the bytes.
//
// Rows carry today's output. Where today's output is wrong, the row also names
// what the fix should produce in a TODO, so the fix flips the pin and clears the
// note (the convention from #141170). Done when no TODO is left.
//
// All three encodings share one canonicalization: MarshalCBOR is
// cbor.Marshal(q.String()), and MarshalJSON emits the same bytes inside quotes,
// so one table drives all three.

type serializationCase struct {
	name string
	load func() Quantity

	// wantString is what String() returns, and the payload both wantJSON and
	// wantCBOR carry.
	wantString string
	wantJSON   string
	// wantCBOR is the hex encoding of MarshalCBOR's output.
	wantCBOR string

	// wantCanonical is what the same value serializes to when the cached string
	// is not populated, i.e. through CanonicalizeBytes rather than the q.s fast
	// path in MarshalJSON. It differs from wantString only where parsing caches
	// a non-canonical form (#138165).
	wantCanonical string
	// canonicalTODO names the agreed correct value when wantString and
	// wantCanonical disagree.
	canonicalTODO string
}

// quantitySerializationCases covers the inventory tracked in #141166: the int64
// boundaries, the values above the largest SI suffix, the binary parse cap, the
// negative rounding class, and the fractional-digit class from #138165.
//
// Serialization is architecture-independent even for the inputs whose int64
// accessors are not (golang/go#45588, see quantity_overflow_test.go), because
// String() formats from the decimal representation rather than through
// int64(math.Pow10(n)). Rows here are therefore pinned exactly, not arch-split.
func quantitySerializationCases() []serializationCase {
	return []serializationCase{
		// --- ordinary values, all three formats ---
		{
			name: "zero-value", load: func() Quantity { return Quantity{} },
			wantString: "0", wantJSON: `"0"`, wantCBOR: "4130", wantCanonical: "0",
		},
		{
			name: "one", load: func() Quantity { return MustParse("1") },
			wantString: "1", wantJSON: `"1"`, wantCBOR: "4131", wantCanonical: "1",
		},
		{
			name: "negative-one", load: func() Quantity { return MustParse("-1") },
			wantString: "-1", wantJSON: `"-1"`, wantCBOR: "422d31", wantCanonical: "-1",
		},
		{
			name: "binary-1Ki", load: func() Quantity { return MustParse("1Ki") },
			wantString: "1Ki", wantJSON: `"1Ki"`, wantCBOR: "43314b69", wantCanonical: "1Ki",
		},
		{
			name: "binary-1024Mi-normalizes-to-1Gi", load: func() Quantity { return MustParse("1024Mi") },
			wantString: "1Gi", wantJSON: `"1Gi"`, wantCBOR: "43314769", wantCanonical: "1Gi",
		},
		{
			name: "decimal-1000M-normalizes-to-1G", load: func() Quantity { return MustParse("1000M") },
			wantString: "1G", wantJSON: `"1G"`, wantCBOR: "423147", wantCanonical: "1G",
		},
		{
			name: "decimal-exponent-format-preserved", load: func() Quantity { return MustParse("1e3") },
			wantString: "1e3", wantJSON: `"1e3"`, wantCBOR: "43316533", wantCanonical: "1e3",
		},
		{
			name: "milli-from-fraction", load: func() Quantity { return MustParse("0.1") },
			wantString: "100m", wantJSON: `"100m"`, wantCBOR: "443130306d", wantCanonical: "100m",
		},
		{
			name: "milli-constructed", load: func() Quantity { return *NewMilliQuantity(1441, DecimalSI) },
			wantString: "1441m", wantJSON: `"1441m"`, wantCBOR: "45313434316d", wantCanonical: "1441m",
		},
		{
			name: "nano-from-sub-milli", load: func() Quantity { return MustParse("0.000000000000001") },
			wantString: "1n", wantJSON: `"1n"`, wantCBOR: "42316e", wantCanonical: "1n",
		},

		// --- the fractional-digit class: #138165, fixed by #138166 ---
		//
		// ParseQuantity's fast path treats scale%3 == 0 as already canonical and
		// caches the input verbatim, so a decimal with exactly 3n fractional
		// digits keeps its input form. The same value reached any other way
		// canonicalizes to the suffixed form, so one value has two wire forms.
		{
			name: "three-fractional-digits", load: func() Quantity { return MustParse("1.441") },
			wantString: "1.441", wantJSON: `"1.441"`, wantCBOR: "45312e343431",
			wantCanonical: "1441m",
			canonicalTODO: "#138166: parse should canonicalize to 1441m, matching NewMilliQuantity(1441)",
		},
		{
			name: "six-fractional-digits", load: func() Quantity { return MustParse("1.000001") },
			wantString: "1.000001", wantJSON: `"1.000001"`, wantCBOR: "48312e303030303031",
			wantCanonical: "1000001u",
			canonicalTODO: "#138166: parse should canonicalize to 1000001u",
		},
		{
			name: "nine-fractional-digits", load: func() Quantity { return MustParse("1.000000001") },
			wantString: "1.000000001", wantJSON: `"1.000000001"`, wantCBOR: "4b312e303030303030303031",
			wantCanonical: "1000000001n",
			canonicalTODO: "#138166: parse should canonicalize to 1000000001n",
		},
		{
			// A non-3n fraction takes the slow path today, so it is already
			// canonical. Kept as the control for the three rows above.
			name: "one-fractional-digit-already-canonical", load: func() Quantity { return MustParse("1.5") },
			wantString: "1500m", wantJSON: `"1500m"`, wantCBOR: "45313530306d", wantCanonical: "1500m",
		},

		// --- int64 boundaries ---
		{
			name: "int64-max-parsed", load: func() Quantity { return MustParse("9223372036854775807") },
			wantString: "9223372036854775807", wantJSON: `"9223372036854775807"`,
			wantCBOR:      "5339323233333732303336383534373735383037",
			wantCanonical: "9223372036854775807",
		},
		{
			name: "int64-min-constructed", load: func() Quantity { return *NewQuantity(math.MinInt64, DecimalSI) },
			wantString: "-9223372036854775808", wantJSON: `"-9223372036854775808"`,
			wantCBOR:      "542d39323233333732303336383534373735383038",
			wantCanonical: "-9223372036854775808",
		},
		{
			name: "two-to-63-exceeds-int64", load: func() Quantity { return MustParse("9223372036854775808") },
			wantString: "9223372036854775808", wantJSON: `"9223372036854775808"`,
			wantCBOR:      "5339323233333732303336383534373735383038",
			wantCanonical: "9223372036854775808",
		},
		{
			name: "two-to-64-exceeds-int64", load: func() Quantity { return MustParse("18446744073709551616") },
			wantString: "18446744073709551616", wantJSON: `"18446744073709551616"`,
			wantCBOR:      "543138343436373434303733373039353531363136",
			wantCanonical: "18446744073709551616",
		},
		{
			name: "scaled-max-times-ten", load: func() Quantity { return *NewScaledQuantity(math.MaxInt64, 1) },
			wantString: "92233720368547758070", wantJSON: `"92233720368547758070"`,
			wantCBOR:      "543932323333373230333638353437373538303730",
			wantCanonical: "92233720368547758070",
		},

		// --- above the largest SI suffix (#140459 and its follow-up #141817) ---
		{
			name: "ten-to-18-has-suffix", load: func() Quantity { return MustParse("1E") },
			wantString: "1E", wantJSON: `"1E"`, wantCBOR: "423145", wantCanonical: "1E",
		},
		{
			name: "ten-to-20-has-suffix", load: func() Quantity { return MustParse("100E") },
			wantString: "100E", wantJSON: `"100E"`, wantCBOR: "4431303045", wantCanonical: "100E",
		},
		{
			name: "ten-to-21-falls-back-to-exponent", load: func() Quantity { return MustParse("1000E") },
			wantString: "1e21", wantJSON: `"1e21"`, wantCBOR: "4431653231", wantCanonical: "1e21",
		},
		{
			name: "ten-to-100-falls-back-to-exponent", load: func() Quantity { return MustParse("1e100") },
			wantString: "10e99", wantJSON: `"10e99"`, wantCBOR: "453130653939", wantCanonical: "10e99",
		},
		{
			name: "scaled-ten-to-21", load: func() Quantity { return *NewScaledQuantity(1, 21) },
			wantString: "1e21", wantJSON: `"1e21"`, wantCBOR: "4431653231", wantCanonical: "1e21",
		},

		// --- binary values capped at parse time ---
		{
			name: "binary-8Ei-caps-at-int64-max", load: func() Quantity { return MustParse("8Ei") },
			wantString: "9223372036854775807", wantJSON: `"9223372036854775807"`,
			wantCBOR:      "5339323233333732303336383534373735383037",
			wantCanonical: "9223372036854775807",
		},
		{
			name: "binary-negative-20Ei-caps-at-int64-min", load: func() Quantity { return MustParse("-20Ei") },
			wantString: "-9223372036854775807", wantJSON: `"-9223372036854775807"`,
			wantCBOR:      "542d39323233333732303336383534373735383037",
			wantCanonical: "-9223372036854775807",
		},

		// --- negative rounding class (#138510) ---
		{
			name: "negative-9_5Gi", load: func() Quantity { return MustParse("-9.5Gi") },
			wantString: "-9728Mi", wantJSON: `"-9728Mi"`, wantCBOR: "472d393732384d69",
			wantCanonical: "-9728Mi",
		},
		{
			name: "negative-9_5000000001Gi", load: func() Quantity { return MustParse("-9.5000000001Gi") },
			wantString: "-10200547328107374183n", wantJSON: `"-10200547328107374183n"`,
			wantCBOR:      "562d31303230303534373332383130373337343138336e",
			wantCanonical: "-10200547328107374183n",
		},
	}
}

// cleared returns the case's quantity with the cached string dropped, so
// serialization goes through CanonicalizeBytes instead of the q.s fast path.
func (tc serializationCase) cleared() Quantity {
	q := tc.load()
	q.s = ""
	return q
}

func TestQuantitySerializationGolden(t *testing.T) {
	for _, tc := range quantitySerializationCases() {
		t.Run(tc.name+"/String", func(t *testing.T) {
			q := tc.load()
			if got := q.String(); got != tc.wantString {
				t.Errorf("String() = %q, want %q", got, tc.wantString)
			}
		})
		t.Run(tc.name+"/MarshalJSON", func(t *testing.T) {
			q := tc.load()
			got, err := json.Marshal(q)
			if err != nil {
				t.Fatalf("json.Marshal() error = %v", err)
			}
			if string(got) != tc.wantJSON {
				t.Errorf("json.Marshal() = %s, want %s", got, tc.wantJSON)
			}
		})
		t.Run(tc.name+"/MarshalCBOR", func(t *testing.T) {
			q := tc.load()
			got, err := q.MarshalCBOR()
			if err != nil {
				t.Fatalf("MarshalCBOR() error = %v", err)
			}
			if hex.EncodeToString(got) != tc.wantCBOR {
				t.Errorf("MarshalCBOR() = %x, want %s", got, tc.wantCBOR)
			}
		})
	}
}

// TestQuantitySerializationCacheAgreement pins the two paths through
// MarshalJSON against each other. A Quantity carrying a cached string emits it
// verbatim; one without it canonicalizes. Any input where those disagree has two
// valid wire forms for a single value, which is what #138166 resolves.
func TestQuantitySerializationCacheAgreement(t *testing.T) {
	for _, tc := range quantitySerializationCases() {
		t.Run(tc.name, func(t *testing.T) {
			uncached := tc.cleared()
			if got := uncached.String(); got != tc.wantCanonical {
				t.Errorf("String() without cached form = %q, want %q%s", got, tc.wantCanonical, todoSuffix(tc.canonicalTODO))
			}
			gotJSON, err := json.Marshal(tc.cleared())
			if err != nil {
				t.Fatalf("json.Marshal() error = %v", err)
			}
			if want := `"` + tc.wantCanonical + `"`; string(gotJSON) != want {
				t.Errorf("json.Marshal() without cached form = %s, want %s%s", gotJSON, want, todoSuffix(tc.canonicalTODO))
			}
			// The row is consistent only when both forms agree; a disagreement
			// must be explained by a TODO naming the fix that removes it.
			disagrees := tc.wantCanonical != tc.wantString
			if disagrees != (tc.canonicalTODO != "") {
				t.Errorf("row pins wantString %q and wantCanonical %q but canonicalTODO = %q; a disagreement needs a TODO and agreement must not carry one",
					tc.wantString, tc.wantCanonical, tc.canonicalTODO)
			}
		})
	}
}

// TestQuantitySerializationSurvivesNoOpArithmetic checks that arithmetic which
// does not change the value does not change the bytes either. Add clears the
// cached string, so for any input whose cached form is not canonical this is
// where the wire form silently changes: adding zero to a stored PVC request
// rewrites "1.441" as "1441m".
func TestQuantitySerializationSurvivesNoOpArithmetic(t *testing.T) {
	zero := *NewQuantity(0, DecimalSI)
	for _, tc := range quantitySerializationCases() {
		t.Run(tc.name, func(t *testing.T) {
			q := tc.load()
			before, err := json.Marshal(q)
			if err != nil {
				t.Fatalf("json.Marshal() error = %v", err)
			}
			q.Add(zero)
			after, err := json.Marshal(q)
			if err != nil {
				t.Fatalf("json.Marshal() after Add(0) error = %v", err)
			}
			want := `"` + tc.wantCanonical + `"`
			if string(after) != want {
				t.Errorf("json.Marshal() after Add(0) = %s, want %s%s", after, want, todoSuffix(tc.canonicalTODO))
			}
			if string(before) != string(after) && tc.canonicalTODO == "" {
				t.Errorf("Add(0) changed the serialized form from %s to %s with no TODO explaining it", before, after)
			}
		})
	}
}

// TestQuantitySerializationRoundTripsByteStable decodes each encoding and
// re-encodes it, which is what an apiserver does on every read-modify-write. The
// second encoding must be byte-identical to the first, otherwise a no-op update
// rewrites the stored value.
func TestQuantitySerializationRoundTripsByteStable(t *testing.T) {
	for _, tc := range quantitySerializationCases() {
		t.Run(tc.name+"/JSON", func(t *testing.T) {
			first, err := json.Marshal(tc.load())
			if err != nil {
				t.Fatalf("json.Marshal() error = %v", err)
			}
			var decoded Quantity
			if err := json.Unmarshal(first, &decoded); err != nil {
				t.Fatalf("json.Unmarshal(%s) error = %v", first, err)
			}
			second, err := json.Marshal(decoded)
			if err != nil {
				t.Fatalf("json.Marshal() of decoded value error = %v", err)
			}
			if string(second) != string(first) {
				t.Errorf("re-encoded %s, want %s", second, first)
			}
		})
		t.Run(tc.name+"/CBOR", func(t *testing.T) {
			q := tc.load()
			first, err := q.MarshalCBOR()
			if err != nil {
				t.Fatalf("MarshalCBOR() error = %v", err)
			}
			var decoded Quantity
			if err := decoded.UnmarshalCBOR(first); err != nil {
				t.Fatalf("UnmarshalCBOR(%x) error = %v", first, err)
			}
			second, err := decoded.MarshalCBOR()
			if err != nil {
				t.Fatalf("MarshalCBOR() of decoded value error = %v", err)
			}
			if string(second) != string(first) {
				t.Errorf("re-encoded %x, want %x", second, first)
			}
		})
	}
}
