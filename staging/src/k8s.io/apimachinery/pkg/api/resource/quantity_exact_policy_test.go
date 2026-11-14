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

// This file holds the test policy: which deviations from the exact value are
// tolerated today, and why.

import (
	"math/big"
	"regexp"
)

// knownDeviation records a currently-wrong behaviour the laws find, keyed by
// a stable case id and pointing at the tracker line or PR.
type knownDeviation struct {
	kind string // "hang", "panic" or the law: "wrong-value", "canonical", "roundtrip", "misc", ...
	ref  string
	// observed is the detail the case reports today; a different wrong result
	// fails. Hang entries have none.
	observed string
}

// parserLeniencyRef is the reason for the digit-less-mantissa entries of
// parseFrozenExamples: the hand-rolled scanner in parseQuantityString accepts
// input with no digits at all as zero, which the documented grammar (which
// requires at least one digit) rejects. The splitREString regexp quoted by
// ErrFormatWrong is never actually matched against the input; note it would
// still accept the bare "." (since "." is in its [0-9.] class).
const parserLeniencyRef = "parseQuantityString accepts a digit-less mantissa as 0 (bare sign, dot, or exponent marker); the documented grammar requires at least one digit"

// nonCanonicalRef is the shared reason for the round-trip law's non-canonical
// spellings: the parse fast path caches the input spelling verbatim, so
// String() emits it instead of the canonical form. Recorded as TODO(#141166) in
// quantity_golden_test.go ("fast path caches a non-canonical spelling; not
// covered by #138166").
const nonCanonicalRef = "#141166: String() fast path caches a non-canonical spelling verbatim (quantity_golden_test.go; not covered by #138166)"

// binarySICapRef: parsing caps a BinarySI spelling at MaxInt64, so a BinarySI
// value above it, which only arithmetic produces, is stored in a spelling that
// decodes to MaxInt64.
const binarySICapRef = "#141166: a BinarySI value above MaxInt64 is stored as a spelling that parses capped at MaxInt64 (quantity_test.go: \"TODO(#141166): once the binarySI parse cap is removed these keep their value ... and round-trip as written\")"

// minInt32AsDecRef: a scale of MinInt32 has no inf.Dec form, since infScale
// cannot negate it (#142817 makes infScale report that), but
// int64Amount.AsDec still flips the exponent's sign. ToDec, Cmp against a
// value at another scale, and the inf.Dec fallback of Add and Sub see the
// flipped value.
const minInt32AsDecRef = "#141166 \"TODO: resolve extreme source-scale behavior in conversion, comparison and Neg/Sub\": int64Amount.AsDec flips the sign of a MinInt32 scale (amount.go \"TODO: handle !canInfScale case\"; quantity_test.go TODO(#141166) Cmp rows)"

// minInt32StringRef: String() keeps an exponent of MinInt32 as it is, since no
// lower multiple of three fits in int32 (#142820), and ParseQuantity rejects
// that exponent (#142395), so a value at that scale does not survive a round
// trip through any encoding.
const minInt32StringRef = "#141166 TODO on extreme source scales: String() writes the MinInt32 scale as 7e-2147483648 (#142820), which ParseQuantity rejects (#142395)"

// knownDeviations is the explicit list of deviations the test tolerates on
// master. Everything not listed here must match the exact value, and every
// listed case must still deviate: if a fix lands, the case starts matching and
// the test fails, forcing the entry to be removed.
var knownDeviations = map[string]knownDeviation{
	"misc/todec/ctor/NewScaledQuantity/7/MinInt32":       {kind: "misc", ref: minInt32AsDecRef, observed: "ToDec value 7e2147483648 != 7e-2147483648"},
	"cmp/7e-2147483648|1":                                {kind: "wrong-value", ref: minInt32AsDecRef, observed: "Cmp = 1, want -1"},
	"cmp/1|7e-2147483648":                                {kind: "wrong-value", ref: minInt32AsDecRef, observed: "Cmp = -1, want 1"},
	"add/7e-2147483648+5":                                {kind: "wrong-value", ref: minInt32AsDecRef, observed: "Add = 700000000000000000000000000001e2147483619, want 500000000000000000000000000001e-29"},
	"sub/7e-2147483648-5":                                {kind: "wrong-value", ref: minInt32AsDecRef, observed: "Sub = 699999999999999999999999999999e2147483619, want -499999999999999999999999999999e-29"},
	"roundtrip/string/ctor/NewScaledQuantity/7/MinInt32": {kind: "roundtrip", ref: minInt32StringRef, observed: "String() = \"7e-2147483648\" does not reparse: unable to parse quantity's suffix"},
	"roundtrip/json/ctor/NewScaledQuantity/7/MinInt32":   {kind: "roundtrip", ref: minInt32StringRef, observed: "json.Unmarshal(\"7e-2147483648\"): unable to parse quantity's suffix"},
	"roundtrip/cbor/ctor/NewScaledQuantity/7/MinInt32":   {kind: "roundtrip", ref: minInt32StringRef, observed: "UnmarshalCBOR: unable to parse quantity's suffix"},
	"roundtrip/proto/ctor/NewScaledQuantity/7/MinInt32":  {kind: "roundtrip", ref: minInt32StringRef, observed: "proto Unmarshal: unable to parse quantity's suffix"},
	"roundtrip/string/mul/capped-8Ei*1024":               {kind: "roundtrip", ref: binarySICapRef, observed: "String \"9223372036854775807Ki\" round trip value 9223372036854775807e0 != 9444732965739290426368e0"},
	"roundtrip/json/mul/capped-8Ei*1024":                 {kind: "roundtrip", ref: binarySICapRef, observed: "json \"9223372036854775807Ki\" round trip value 9223372036854775807e0 != 9444732965739290426368e0"},
	"roundtrip/cbor/mul/capped-8Ei*1024":                 {kind: "roundtrip", ref: binarySICapRef, observed: "cbor round trip value 9223372036854775807e0 != 9444732965739290426368e0"},
	"roundtrip/proto/mul/capped-8Ei*1024":                {kind: "roundtrip", ref: binarySICapRef, observed: "proto round trip value 9223372036854775807e0 != 9444732965739290426368e0"},
	"roundtrip/string/mul/capped--8Ei*1024":              {kind: "roundtrip", ref: binarySICapRef, observed: "String \"-9223372036854775807Ki\" round trip value -9223372036854775807e0 != -9444732965739290426368e0"},
	"roundtrip/json/mul/capped--8Ei*1024":                {kind: "roundtrip", ref: binarySICapRef, observed: "json \"-9223372036854775807Ki\" round trip value -9223372036854775807e0 != -9444732965739290426368e0"},
	"roundtrip/cbor/mul/capped--8Ei*1024":                {kind: "roundtrip", ref: binarySICapRef, observed: "cbor round trip value -9223372036854775807e0 != -9444732965739290426368e0"},
	"roundtrip/proto/mul/capped--8Ei*1024":               {kind: "roundtrip", ref: binarySICapRef, observed: "proto round trip value -9223372036854775807e0 != -9444732965739290426368e0"},

	// The shortest spelling of the carry at the saturated scale that
	// parseRegressions lists for the grid's 100-digit mantissas.
	"sigdigits/999999999999999999999999999999e2147483647":  {kind: "parse", ref: carrySaturationRef, observed: "decode 1e2147483676 (DecimalExponent), exact 1e2147483676 (DecimalExponent), v1.37.0 (not judged), stored as \"1000000000000000000000000000000e2147483646\", which decodes to 1e2147483675 (DecimalExponent)"},
	"sigdigits/-999999999999999999999999999999e2147483647": {kind: "parse", ref: carrySaturationRef, observed: "decode -1e2147483676 (DecimalExponent), exact -1e2147483676 (DecimalExponent), v1.37.0 (not judged), stored as \"-1000000000000000000000000000000e2147483646\", which decodes to -1e2147483675 (DecimalExponent)"},

	// The scale-alignment hang class (#141166 Known Gaps): AsFloat64Slow
	// still builds 10^2147483647. Add, Sub and RoundUp no longer do.
	"float64slow/1e2147483647": {kind: "hang", ref: "#141166 Known Gaps: inf.Dec scale alignment writes 10^delta digits; AsFloat64Slow is still unbounded"},

	// Non-canonical spellings the parse fast path caches verbatim; the
	// canonical-form check of the String law flags each of them. See
	// nonCanonicalRef. ".5" is canonicalized correctly and is therefore not
	// listed here. The exp grid's zero-exponent cases ("1e0", "123e0") are the
	// same class.
	"canonical/noncanon/5.":    {kind: "canonical", ref: nonCanonicalRef, observed: "String() = \"5.\", canonical form \"5\""},
	"canonical/noncanon/1.":    {kind: "canonical", ref: nonCanonicalRef, observed: "String() = \"1.\", canonical form \"1\""},
	"canonical/noncanon/+1":    {kind: "canonical", ref: nonCanonicalRef, observed: "String() = \"+1\", canonical form \"1\""},
	"canonical/noncanon/01":    {kind: "canonical", ref: nonCanonicalRef, observed: "String() = \"01\", canonical form \"1\""},
	"canonical/noncanon/1E3":   {kind: "canonical", ref: nonCanonicalRef, observed: "String() = \"1E3\", canonical form \"1e3\""},
	"canonical/noncanon/1e+21": {kind: "canonical", ref: nonCanonicalRef, observed: "String() = \"1e+21\", canonical form \"1e21\""},
	"canonical/exp/1e0":        {kind: "canonical", ref: nonCanonicalRef, observed: "String() = \"1e0\", canonical form \"1\""},
	"canonical/exp/123e0":      {kind: "canonical", ref: nonCanonicalRef, observed: "String() = \"123e0\", canonical form \"123\""},
}

// The parse law puts every grid input in one of these quadrants, from two
// independent verdicts: correct (the decode matches exactDecode and the JSON
// spelling it is stored as decodes back to the same value) and compatible (the
// decode equals the decode of the last release, decodeCorpusRelease, recorded
// in testdata/quantity_decode_corpus.json, or differs from it only by the
// rounding to 29 significant digits #143245 added, see roundedFromRelease; those
// are counted apart, as "correct/rounded" and "frozen/rounded").
//
//   - correct and compatible: passes.
//   - wrong and compatible ("frozen"): the release decoded the input the same wrong
//     way and may have persisted the spelling, so it must not change. It passes
//     when frozenMechanism names why the decode is wrong, so that a fault in
//     exactDecode cannot pass as frozen; a change in its decode fails as
//     incompatible.
//   - correct and incompatible: allowed only if listed in parseApprovedChanges.
//   - wrong and incompatible: allowed only if listed in parseRegressions.
//   - a stored spelling that does not decode back to the decoded value is
//     never covered: "wrong/stored-spelling" for a correct decode and
//     "frozen/stored-spelling" for a frozen one must be listed in
//     parseRegressions too.
//
// An input on which the release hung has no recorded decode to be compatible with:
// nothing can have been stored through it. It passes when correct and must be
// listed in parseRegressions when wrong. Every list entry
// must be in the grid and in the quadrant its list names, so a fix or a change
// makes a stale entry fail.

// parseQuadrantCounts records how many grid inputs fall in each quadrant, so
// that a decode moving between quadrants shows up even where no list names it
// (an input the release recorded as hanging, say, or a frozen one turning
// unjudged).
var parseQuadrantCounts = map[string]int{
	"correct/compatible":     5556,
	"correct/no-1.37-decode": 819,
	"correct/rounded":        178,
	"frozen":                 1132,
	"frozen/rounded":         36,
	"unjudged/compatible":    4,
	"wrong/incompatible":     6,
	"wrong/no-1.37-decode":   66,
}

// parseApprovedChanges lists inputs whose decode differs from the release's and
// is now correct. None today.
var parseApprovedChanges = map[string]string{}

const (
	zeroMinInt32Ref    = "#141166 #142452: a zero whose decimal exponent narrows to MinInt32 decoded to 0 on 1.37 and is rejected now (ErrSuffix)"
	int32WrapRef       = "#141166 TODO on exponents outside the int32 range, counting digits after the decimal point, which still wrap for stored-data compatibility; 1.37 hung on these"
	carrySaturationRef = "not on #141166 yet: rounding to 29 digits carries into a 30th at the saturated MaxInt32 scale, String() writes that as 31 digits at MaxInt32-1, and reparsing rounds and saturates again, to a tenth of the value; 1.37 hung on these"
)

// parseRegression is a parseRegressions entry: the reason, and the decode
// observed today (see parseVerdict.observed); a different wrong decode fails.
type parseRegression struct {
	ref      string
	observed string
}

// parseRegressions lists the inputs that decode wrongly and differently from
// the release (either it decoded them correctly, or it hung on them), and the
// inputs whose decode is right but whose stored spelling does not decode back. A decode
// is wrong when it differs from exactDecode or when the spelling it is stored
// as does not decode back to the same value. They were found by the cross
// product of the int64-rail mantissas with the int32-edge exponents.
var parseRegressions = map[string]parseRegression{
	"0e-2147483648":                                                  {zeroMinInt32Ref, "err"},
	"0e2147483648":                                                   {zeroMinInt32Ref, "err"},
	"1000000000000000000e-2147483649":                                {int32WrapRef, "1e2147483665 (DecimalExponent)"},
	"10000000000000000000e-2147483649":                               {int32WrapRef, "1e2147483666 (DecimalExponent)"},
	"100000000000000000000e-2147483649":                              {int32WrapRef, "1e2147483667 (DecimalExponent)"},
	"9223372036854775806e-2147483649":                                {int32WrapRef, "9223372036854775806e2147483647 (DecimalExponent)"},
	"9223372036854775807e-2147483649":                                {int32WrapRef, "9223372036854775807e2147483647 (DecimalExponent)"},
	"9223372036854775808e-2147483649":                                {int32WrapRef, "9223372036854775808e2147483647 (DecimalExponent)"},
	"18446744073709551615e-2147483649":                               {int32WrapRef, "18446744073709551615e2147483647 (DecimalExponent)"},
	"18446744073709551616e-2147483649":                               {int32WrapRef, "18446744073709551616e2147483647 (DecimalExponent)"},
	"123456789012345678901e-2147483649":                              {int32WrapRef, "123456789012345678901e2147483647 (DecimalExponent)"},
	"1234567890123456789012345678901234567890e-2147483649":           {int32WrapRef, "1234567890123456789012345679e2147483648 (DecimalExponent)"},
	"10000000000000000000000000000000000000000000000000e-2147483649": {int32WrapRef, "1e2147483675 (DecimalExponent)"},
	"9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e-2147483649": {int32WrapRef, "1e2147483676 (DecimalExponent), stored as \"1000000000000000000000000000000e2147483646\""},
	"9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e2147483627":  {carrySaturationRef, "1e2147483676 (DecimalExponent), stored as \"1000000000000000000000000000000e2147483646\""},
	"9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e2147483628":  {carrySaturationRef, "1e2147483676 (DecimalExponent), stored as \"1000000000000000000000000000000e2147483646\""},
	"9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e2147483646":  {carrySaturationRef, "1e2147483676 (DecimalExponent), stored as \"1000000000000000000000000000000e2147483646\""},
	"9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e2147483647":  {carrySaturationRef, "1e2147483676 (DecimalExponent), stored as \"1000000000000000000000000000000e2147483646\""},
	"9223372036854775.807e-2147483649":                                {int32WrapRef, "9223372036854775807e2147483644 (DecimalExponent)"},
	"9223372036854775.807e-2147483648":                                {int32WrapRef, "9223372036854775807e2147483645 (DecimalExponent)"},
	"9223372036854775.807e-2147483647":                                {int32WrapRef, "9223372036854775807e2147483646 (DecimalExponent)"},
	"9223372036854775.807e-2147483646":                                {int32WrapRef, "9223372036854775807e2147483647 (DecimalExponent)"},
	"9223372036854775.808e-2147483649":                                {int32WrapRef, "9223372036854775808e2147483644 (DecimalExponent)"},
	"-0e-2147483648":                                                  {zeroMinInt32Ref, "err"},
	"-0e2147483648":                                                   {zeroMinInt32Ref, "err"},
	"-1000000000000000000e-2147483649":                                {int32WrapRef, "-1e2147483665 (DecimalExponent)"},
	"-10000000000000000000e-2147483649":                               {int32WrapRef, "-1e2147483666 (DecimalExponent)"},
	"-100000000000000000000e-2147483649":                              {int32WrapRef, "-1e2147483667 (DecimalExponent)"},
	"-9223372036854775806e-2147483649":                                {int32WrapRef, "-9223372036854775806e2147483647 (DecimalExponent)"},
	"-9223372036854775807e-2147483649":                                {int32WrapRef, "-9223372036854775807e2147483647 (DecimalExponent)"},
	"-9223372036854775808e-2147483649":                                {int32WrapRef, "-9223372036854775808e2147483647 (DecimalExponent)"},
	"-18446744073709551615e-2147483649":                               {int32WrapRef, "-18446744073709551615e2147483647 (DecimalExponent)"},
	"-18446744073709551616e-2147483649":                               {int32WrapRef, "-18446744073709551616e2147483647 (DecimalExponent)"},
	"-123456789012345678901e-2147483649":                              {int32WrapRef, "-123456789012345678901e2147483647 (DecimalExponent)"},
	"-1234567890123456789012345678901234567890e-2147483649":           {int32WrapRef, "-1234567890123456789012345679e2147483648 (DecimalExponent)"},
	"-10000000000000000000000000000000000000000000000000e-2147483649": {int32WrapRef, "-1e2147483675 (DecimalExponent)"},
	"-9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e-2147483649": {int32WrapRef, "-1e2147483676 (DecimalExponent), stored as \"-1000000000000000000000000000000e2147483646\""},
	"-9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e2147483627":  {carrySaturationRef, "-1e2147483676 (DecimalExponent), stored as \"-1000000000000000000000000000000e2147483646\""},
	"-9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e2147483628":  {carrySaturationRef, "-1e2147483676 (DecimalExponent), stored as \"-1000000000000000000000000000000e2147483646\""},
	"-9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e2147483646":  {carrySaturationRef, "-1e2147483676 (DecimalExponent), stored as \"-1000000000000000000000000000000e2147483646\""},
	"-9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e2147483647":  {carrySaturationRef, "-1e2147483676 (DecimalExponent), stored as \"-1000000000000000000000000000000e2147483646\""},
	"-9223372036854775.807e-2147483649":                               {int32WrapRef, "-9223372036854775807e2147483644 (DecimalExponent)"},
	"-9223372036854775.807e-2147483648":                               {int32WrapRef, "-9223372036854775807e2147483645 (DecimalExponent)"},
	"-9223372036854775.807e-2147483647":                               {int32WrapRef, "-9223372036854775807e2147483646 (DecimalExponent)"},
	"-9223372036854775.807e-2147483646":                               {int32WrapRef, "-9223372036854775807e2147483647 (DecimalExponent)"},
	"-9223372036854775.808e-2147483649":                               {int32WrapRef, "-9223372036854775808e2147483644 (DecimalExponent)"},
	"-9223372036854775.808e-2147483648":                               {int32WrapRef, "-9223372036854775808e2147483645 (DecimalExponent)"},
	"-9223372036854775.808e-2147483647":                               {int32WrapRef, "-9223372036854775808e2147483646 (DecimalExponent)"},
	"-9223372036854775.808e-2147483646":                               {int32WrapRef, "-9223372036854775808e2147483647 (DecimalExponent)"},
	"+0e-2147483648":                                                  {zeroMinInt32Ref, "err"},
	"+0e2147483648":                                                   {zeroMinInt32Ref, "err"},
	"+1000000000000000000e-2147483649":                                {int32WrapRef, "1e2147483665 (DecimalExponent)"},
	"+10000000000000000000e-2147483649":                               {int32WrapRef, "1e2147483666 (DecimalExponent)"},
	"+100000000000000000000e-2147483649":                              {int32WrapRef, "1e2147483667 (DecimalExponent)"},
	"+9223372036854775806e-2147483649":                                {int32WrapRef, "9223372036854775806e2147483647 (DecimalExponent)"},
	"+9223372036854775807e-2147483649":                                {int32WrapRef, "9223372036854775807e2147483647 (DecimalExponent)"},
	"+9223372036854775808e-2147483649":                                {int32WrapRef, "9223372036854775808e2147483647 (DecimalExponent)"},
	"+18446744073709551615e-2147483649":                               {int32WrapRef, "18446744073709551615e2147483647 (DecimalExponent)"},
	"+18446744073709551616e-2147483649":                               {int32WrapRef, "18446744073709551616e2147483647 (DecimalExponent)"},
	"+123456789012345678901e-2147483649":                              {int32WrapRef, "123456789012345678901e2147483647 (DecimalExponent)"},
	"+1234567890123456789012345678901234567890e-2147483649":           {int32WrapRef, "1234567890123456789012345679e2147483648 (DecimalExponent)"},
	"+10000000000000000000000000000000000000000000000000e-2147483649": {int32WrapRef, "1e2147483675 (DecimalExponent)"},
	"+9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e-2147483649": {int32WrapRef, "1e2147483676 (DecimalExponent), stored as \"1000000000000000000000000000000e2147483646\""},
	"+9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e2147483627":  {carrySaturationRef, "1e2147483676 (DecimalExponent), stored as \"1000000000000000000000000000000e2147483646\""},
	"+9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e2147483628":  {carrySaturationRef, "1e2147483676 (DecimalExponent), stored as \"1000000000000000000000000000000e2147483646\""},
	"+9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e2147483646":  {carrySaturationRef, "1e2147483676 (DecimalExponent), stored as \"1000000000000000000000000000000e2147483646\""},
	"+9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e2147483647":  {carrySaturationRef, "1e2147483676 (DecimalExponent), stored as \"1000000000000000000000000000000e2147483646\""},
	"+9223372036854775.807e-2147483649": {int32WrapRef, "9223372036854775807e2147483644 (DecimalExponent)"},
	"+9223372036854775.807e-2147483648": {int32WrapRef, "9223372036854775807e2147483645 (DecimalExponent)"},
	"+9223372036854775.807e-2147483647": {int32WrapRef, "9223372036854775807e2147483646 (DecimalExponent)"},
	"+9223372036854775.807e-2147483646": {int32WrapRef, "9223372036854775807e2147483647 (DecimalExponent)"},
	"+9223372036854775.808e-2147483649": {int32WrapRef, "9223372036854775808e2147483644 (DecimalExponent)"},
}

// parseFrozenExamples names representatives of the frozen quadrant with the
// reason 1.37 decodes them the way it does. Each must stay frozen: a change to
// its decode changes what an already-stored spelling means.
var parseFrozenExamples = map[string]string{
	"-":            parserLeniencyRef,
	"+":            parserLeniencyRef,
	".":            parserLeniencyRef,
	"e3":           parserLeniencyRef,
	"-e3":          parserLeniencyRef,
	"1e4294967297": "#142395 exponent truncated to 32 bits (2^32+1 -> 1)",
	"1e4294967296": "#142395 exponent truncated to 32 bits (2^32 -> 0)",
}

// int32NarrowingRef is the shared reason for the frozen decodes of decimal
// exponents: the parser keeps the exponent, and the scale it becomes after the
// fraction digits are taken off, in int32, so one outside int32 is truncated
// or wraps, and one outside int64 is rejected.
const int32NarrowingRef = "#142395/#141166: a decimal exponent outside int32, before or after the fraction digits are taken off, is truncated or wraps; outside int64 it is rejected"

// frozenMechanism returns why ParseQuantity's decode of input may be wrong
// and still frozen, or "" when no known mechanism makes it wrong. A frozen
// decode is one that matches the release and disagrees with exactDecode, so
// without this check a fault in exactDecode on any input the library decodes
// correctly would read as frozen, not as a failure.
func frozenMechanism(input string) string {
	if digitlessInput.MatchString(input) {
		return parserLeniencyRef
	}
	m := decimalExponentInput.FindStringSubmatch(input)
	if m == nil {
		return ""
	}
	exp, ok := new(big.Int).SetString(m[4], 10)
	if !ok {
		return ""
	}
	scale := new(big.Int).Sub(exp, big.NewInt(int64(len(m[3]))))
	for _, e := range []*big.Int{exp, scale} {
		if !e.IsInt64() || e.Int64() != int64(int32(e.Int64())) {
			return int32NarrowingRef
		}
	}
	return ""
}

// digitlessInput matches an input whose number has no digits: a bare sign or
// dot, optionally followed by a suffix.
var digitlessInput = regexp.MustCompile(`^[+-]?\.?(?:[eEinumkKMGTP].*)?$`)

// decimalExponentInput matches an input of the grammar with a decimal
// exponent suffix: sign, whole digits, fraction digits, exponent literal.
var decimalExponentInput = regexp.MustCompile(`^([+-]?)([0-9]*)(?:\.([0-9]*))?[eE]([+-]?[0-9]+)$`)
