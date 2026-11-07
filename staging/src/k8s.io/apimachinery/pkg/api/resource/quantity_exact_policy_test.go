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
	"math"
	"math/big"
	"regexp"
	"strconv"
	"strings"
)

// isKnownHang reports whether knownDeviations records the case as a hang.
func isKnownHang(id string) bool {
	return knownDeviations[id].kind == "hang"
}

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
const binarySICapRef = "#141166 (parse-time cap, issuecomment-5183321192): a BinarySI value above MaxInt64 is stored as a spelling that parses capped at MaxInt64"

// knownDeviations is the explicit list of deviations the test tolerates on
// master. Everything not listed here must match the exact value, and every
// listed case must still deviate: if a fix lands, the case starts matching and
// the test fails, forcing the entry to be removed.
var knownDeviations = map[string]knownDeviation{
	// Parse hang representatives. Every other input parseHangs reports is
	// skipped; these two run in the child process so the day the hang is fixed
	// the entry trips. The first is on the nano-round path of a negative
	// exponent, the second on the long-mantissa path of a positive one.
	"parse/1e-2147483630": {kind: "hang", ref: "#141166 hang class: nano-round aligns scales at an int32-edge negative exponent"},
	"parse/1234567890123456789012345678901234567890e2147483000": {kind: "hang", ref: "#141166 hang class: nano-round aligns scales at an int32-edge positive exponent (inf.Dec path of a long mantissa)"},

	// A scale of MinInt32 cannot be represented in inf.Dec (infScale negates it
	// back to MinInt32), so ToDec changes the value, RoundUp/AsScale wraps the
	// scale delta, and String() serializes the wrapped value. This is the same
	// class pinned as TODO(#141166) in quantity_test.go's MinInt32-scale
	// Cmp/CmpInt64 and String() rows (#142224); #142156 tracks the same class.
	"misc/roundup/ctor/NewScaledQuantity/7/MinInt32":     {kind: "misc", ref: "#141166 MinInt32 scale is unrepresentable in inf.Dec (quantity_test.go TODO(#141166) Cmp/CmpInt64 rows, #142156)", observed: "RoundUp(0) = 7e0, true; exact 1e0, false"},
	"misc/asscale/ctor/NewScaledQuantity/7/MinInt32":     {kind: "misc", ref: "#141166 MinInt32 scale is unrepresentable in inf.Dec (quantity_test.go TODO(#141166) Cmp/CmpInt64 rows, #142156)", observed: "AsScale(0) = 7e0, true; exact 1e0, false"},
	"misc/todec/ctor/NewScaledQuantity/7/MinInt32":       {kind: "misc", ref: "#141166 MinInt32 scale is unrepresentable in inf.Dec (quantity_test.go TODO(#141166) Cmp/CmpInt64 rows, #142156)", observed: "ToDec value 7e2147483648 != 7e-2147483648"},
	"canonical/ctor/NewScaledQuantity/7/MinInt32":        {kind: "canonical", ref: "#141166 MinInt32-scale String() wraps the exponent (7e-2147483648 -> 70e2147483647; quantity_test.go TODO(#141166) String() rows, #142156)", observed: "String() = \"70e2147483647\", canonical form \"70e-2147483649\""},
	"roundtrip/string/ctor/NewScaledQuantity/7/MinInt32": {kind: "roundtrip", ref: "#141166 the wrapped String() reparses to a different value (#142156)", observed: "String \"70e2147483647\" round trip value 7e2147483648 != original 7e-2147483648"},
	"roundtrip/json/ctor/NewScaledQuantity/7/MinInt32":   {kind: "roundtrip", ref: "#141166 the wrapped String() reparses to a different value (#142156)", observed: "json \"70e2147483647\" round trip value 7e2147483648 != original 7e-2147483648"},
	"roundtrip/cbor/ctor/NewScaledQuantity/7/MinInt32":   {kind: "roundtrip", ref: "#141166 the wrapped String() reparses to a different value (#142156)", observed: "cbor round trip value 7e2147483648 != original 7e-2147483648"},
	"roundtrip/proto/ctor/NewScaledQuantity/7/MinInt32":  {kind: "roundtrip", ref: "#141166 the wrapped String() reparses to a different value (#142156)", observed: "proto round trip value 7e2147483648 != original 7e-2147483648"},
	"roundtrip/string/mul/capped-8Ei*1024":               {kind: "roundtrip", ref: binarySICapRef, observed: "String \"9223372036854775807Ki\" round trip value 9223372036854775807e0 != original 9444732965739290426368e0"},
	"roundtrip/json/mul/capped-8Ei*1024":                 {kind: "roundtrip", ref: binarySICapRef, observed: "json \"9223372036854775807Ki\" round trip value 9223372036854775807e0 != original 9444732965739290426368e0"},
	"roundtrip/cbor/mul/capped-8Ei*1024":                 {kind: "roundtrip", ref: binarySICapRef, observed: "cbor round trip value 9223372036854775807e0 != original 9444732965739290426368e0"},
	"roundtrip/proto/mul/capped-8Ei*1024":                {kind: "roundtrip", ref: binarySICapRef, observed: "proto round trip value 9223372036854775807e0 != original 9444732965739290426368e0"},
	"roundtrip/string/mul/capped--8Ei*1024":              {kind: "roundtrip", ref: binarySICapRef, observed: "String \"-9223372036854775807Ki\" round trip value -9223372036854775807e0 != original -9444732965739290426368e0"},
	"roundtrip/json/mul/capped--8Ei*1024":                {kind: "roundtrip", ref: binarySICapRef, observed: "json \"-9223372036854775807Ki\" round trip value -9223372036854775807e0 != original -9444732965739290426368e0"},
	"roundtrip/cbor/mul/capped--8Ei*1024":                {kind: "roundtrip", ref: binarySICapRef, observed: "cbor round trip value -9223372036854775807e0 != original -9444732965739290426368e0"},
	"roundtrip/proto/mul/capped--8Ei*1024":               {kind: "roundtrip", ref: binarySICapRef, observed: "proto round trip value -9223372036854775807e0 != original -9444732965739290426368e0"},
	// The same wrap at the top: 1e2147483648 prints as 10e2147483647, the right
	// value with an exponent that is not a multiple of 3.
	"canonical/ctor/NewScaledQuantity/100/2147483646": {kind: "canonical", ref: "#141166 String() wraps the exponent past MaxInt32 (1e2147483648 -> 10e2147483647)", observed: "String() = \"10e2147483647\", canonical form \"100e2147483646\""},
	// Sub-nano values constructed via NewScaledQuantity serialize to a string
	// that reparses to 1n, so the round trip is not value-preserving (#141306).
	"roundtrip/string/ctor/NewScaledQuantity/7/-18":   {kind: "roundtrip", ref: "#141306 sub-nano quantities do not round-trip (reparse rounds to 1n)", observed: "String \"7e-18\" round trip value 1e-9 != original 7e-18"},
	"roundtrip/json/ctor/NewScaledQuantity/7/-18":     {kind: "roundtrip", ref: "#141306 sub-nano quantities do not round-trip (reparse rounds to 1n)", observed: "json \"7e-18\" round trip value 1e-9 != original 7e-18"},
	"roundtrip/cbor/ctor/NewScaledQuantity/7/-18":     {kind: "roundtrip", ref: "#141306 sub-nano quantities do not round-trip (reparse rounds to 1n)", observed: "cbor round trip value 1e-9 != original 7e-18"},
	"roundtrip/proto/ctor/NewScaledQuantity/7/-18":    {kind: "roundtrip", ref: "#141306 sub-nano quantities do not round-trip (reparse rounds to 1n)", observed: "proto round trip value 1e-9 != original 7e-18"},
	"roundtrip/string/ctor/NewScaledQuantity/-7/-18":  {kind: "roundtrip", ref: "#141306 sub-nano quantities do not round-trip (reparse rounds to 1n)", observed: "String \"-7e-18\" round trip value -1e-9 != original -7e-18"},
	"roundtrip/json/ctor/NewScaledQuantity/-7/-18":    {kind: "roundtrip", ref: "#141306 sub-nano quantities do not round-trip (reparse rounds to 1n)", observed: "json \"-7e-18\" round trip value -1e-9 != original -7e-18"},
	"roundtrip/cbor/ctor/NewScaledQuantity/-7/-18":    {kind: "roundtrip", ref: "#141306 sub-nano quantities do not round-trip (reparse rounds to 1n)", observed: "cbor round trip value -1e-9 != original -7e-18"},
	"roundtrip/proto/ctor/NewScaledQuantity/-7/-18":   {kind: "roundtrip", ref: "#141306 sub-nano quantities do not round-trip (reparse rounds to 1n)", observed: "proto round trip value -1e-9 != original -7e-18"},
	"roundtrip/string/ctor/NewScaledQuantity/-7/-400": {kind: "roundtrip", ref: "#141306 sub-nano quantities do not round-trip (reparse rounds to 1n)", observed: "String \"-700e-402\" round trip value -1e-9 != original -7e-400"},
	"roundtrip/json/ctor/NewScaledQuantity/-7/-400":   {kind: "roundtrip", ref: "#141306 sub-nano quantities do not round-trip (reparse rounds to 1n)", observed: "json \"-700e-402\" round trip value -1e-9 != original -7e-400"},
	"roundtrip/cbor/ctor/NewScaledQuantity/-7/-400":   {kind: "roundtrip", ref: "#141306 sub-nano quantities do not round-trip (reparse rounds to 1n)", observed: "cbor round trip value -1e-9 != original -7e-400"},
	"roundtrip/proto/ctor/NewScaledQuantity/-7/-400":  {kind: "roundtrip", ref: "#141306 sub-nano quantities do not round-trip (reparse rounds to 1n)", observed: "proto round trip value -1e-9 != original -7e-400"},

	// The canonical hang class: inf.Dec aligns scales by writing 10^delta
	// digits. Reachable through Add/Sub, AsScale/RoundUp, and AsFloat64Slow.
	"add/1e2147483647+1":       {kind: "hang", ref: "#141166 hang class: inf.Dec scale alignment writes 10^2147483647 digits"},
	"float64slow/1e2147483647": {kind: "hang", ref: "#141166 hang class: AsFloat64Slow materializes 10^2147483647"},
	"roundup/1e2147483647-dec": {kind: "hang", ref: "#141166 hang class: AsScale/RoundUp aligns scales at a huge exponent"},
	// int64Amount.Add/Sub compute the scale delta in int32 and wrap instead of
	// falling back to inf.Dec (Cmp got this fix in #142013, Add/Sub did not).
	"add/1e2147483647+1n": {kind: "wrong-value", ref: "int64Amount.Add computes the scale delta in int32 and wraps (1e2147483647+1n -> 2e-9)", observed: "Add = 2e-9, want 1e2147483647 (int32 scale delta wraps)"},
	"sub/1e2147483647-1n": {kind: "wrong-value", ref: "int64Amount.Sub computes the scale delta in int32 and wraps (1e2147483647-1n -> 0)", observed: "Sub = 0, want 1e2147483647 (int32 scale delta wraps)"},
	"add/7e-2147483648+5": {kind: "wrong-value", ref: "int64Amount.Add computes the scale delta in int32 and wraps (7e-2147483648+5 -> 12e-2147483648)", observed: "Add = 12e-2147483648, want 5e0 (int32 scale delta wraps)"},
	"sub/7e-2147483648-5": {kind: "wrong-value", ref: "int64Amount.Sub computes the scale delta in int32 and wraps (7e-2147483648-5 -> 2e-2147483648)", observed: "Sub = 2e-2147483648, want -5e0 (int32 scale delta wraps)"},
	"cmp/7e-2147483648|1": {kind: "wrong-value", ref: "#141166 MinInt32 scale: Cmp against a scale-0 value goes through AsDec, which cannot hold scale 2^31 and flips the exponent sign, so 7e-2147483648 compares greater than 1", observed: "Cmp = 1, want -1"},
	"cmp/1|7e-2147483648": {kind: "wrong-value", ref: "#141166 MinInt32 scale: the same AsDec flip seen from the other operand, so 1 compares less than 7e-2147483648", observed: "Cmp = -1, want 1"},

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
// independent verdicts: correct (the decode matches exact.Parse and the JSON
// spelling it is stored as decodes back to the same value) and compatible (the
// decode equals the decode of the last release, decodeCorpusRelease, recorded
// in testdata/quantity_decode_corpus.json).
//
//   - correct and compatible: passes.
//   - wrong and compatible ("frozen"): the release decoded the input the same wrong
//     way and may have persisted the spelling, so it must not change. It passes
//     when frozenMechanism names why the decode is wrong, so that a fault in
//     exact.Parse cannot pass as frozen; a change in its decode fails as
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
	"correct/compatible":     5722,
	"correct/no-1.37-decode": 31,
	"frozen":                 1162,
	"frozen/stored-spelling": 6,
	"hang":                   619,
	"unjudged/compatible":    4,
	"wrong/incompatible":     6,
	"wrong/no-1.37-decode":   235,
	"wrong/stored-spelling":  12,
}

// parseApprovedChanges lists inputs whose decode differs from the release's and
// is now correct. None today.
var parseApprovedChanges = map[string]string{}

const (
	zeroMinInt32Ref   = "TODO(before 1.38): a zero whose decimal exponent truncates to MinInt32 decoded to 0 on 1.37 and is rejected now; ParseQuantity checks exponent == MinInt32 before it checks for zero"
	negMinInt32Ref    = "TODO(before 1.38): a decimal exponent of -2147483648 is rejected (ErrSuffix); the grammar accepts it and the value rounds up to 1n (1.37 hung on these)"
	posMinInt32Ref    = "TODO(before 1.38): a decimal exponent of 2147483648 truncates to MinInt32 and is rejected (ErrSuffix); the grammar accepts it (1.37 hung on these)"
	wrappedRef        = "TODO(before 1.38): the int32 scale arithmetic wraps and the decode flips the exponent's sign (1.37 hung on these)"
	storedHangsRef    = "TODO(before 1.38): decodes exactly, but the stored spelling's exponent wraps in int32 and the spelling hangs on decode (1.37 hung on these)"
	storedSpellingRef = "TODO(before 1.38): decodes exactly and as on 1.37, but String() wraps the exponent or pads the mantissa and the stored spelling hangs on decode (#142156, #141306)"
	frozenStoredRef   = "TODO(before 1.38): decodes as 1.37 did (wrongly, the exponent wraps), and the spelling it is stored as hangs on decode; the decode must stay, the stored spelling must not hang"
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
// is wrong when it differs from exact.Parse or when the spelling it is stored
// as does not decode back to the same value. They were found by the cross
// product of the int64-rail mantissas with the int32-edge exponents.
var parseRegressions = map[string]parseRegression{
	"1000e-2147483649":                                     {frozenStoredRef, "1e2147483650 (DecimalExponent), stored as \"1e-2147483646\""},
	"999999999999999999e-2147483649":                       {frozenStoredRef, "999999999999999999e2147483647 (DecimalExponent), stored as \"9999999999999999990e2147483646\""},
	"-1000e-2147483649":                                    {frozenStoredRef, "-1e2147483650 (DecimalExponent), stored as \"-1e-2147483646\""},
	"-999999999999999999e-2147483649":                      {frozenStoredRef, "-999999999999999999e2147483647 (DecimalExponent), stored as \"-9999999999999999990e2147483646\""},
	"+1000e-2147483649":                                    {frozenStoredRef, "1e2147483650 (DecimalExponent), stored as \"1e-2147483646\""},
	"+999999999999999999e-2147483649":                      {frozenStoredRef, "999999999999999999e2147483647 (DecimalExponent), stored as \"9999999999999999990e2147483646\""},
	"1000e2147483647":                                      {storedSpellingRef, "1e2147483650 (DecimalExponent), stored as \"1e-2147483646\""},
	"999999999999999999e2147483000":                        {storedSpellingRef, "999999999999999999e2147483000 (DecimalExponent), stored as \"99999999999999999900e2147482998\""},
	"999999999999999999e2147483627":                        {storedSpellingRef, "999999999999999999e2147483627 (DecimalExponent), stored as \"99999999999999999900e2147483625\""},
	"999999999999999999e2147483647":                        {storedSpellingRef, "999999999999999999e2147483647 (DecimalExponent), stored as \"9999999999999999990e2147483646\""},
	"-1000e2147483647":                                     {storedSpellingRef, "-1e2147483650 (DecimalExponent), stored as \"-1e-2147483646\""},
	"-999999999999999999e2147483000":                       {storedSpellingRef, "-999999999999999999e2147483000 (DecimalExponent), stored as \"-99999999999999999900e2147482998\""},
	"-999999999999999999e2147483627":                       {storedSpellingRef, "-999999999999999999e2147483627 (DecimalExponent), stored as \"-99999999999999999900e2147483625\""},
	"-999999999999999999e2147483647":                       {storedSpellingRef, "-999999999999999999e2147483647 (DecimalExponent), stored as \"-9999999999999999990e2147483646\""},
	"+1000e2147483647":                                     {storedSpellingRef, "1e2147483650 (DecimalExponent), stored as \"1e-2147483646\""},
	"+999999999999999999e2147483000":                       {storedSpellingRef, "999999999999999999e2147483000 (DecimalExponent), stored as \"99999999999999999900e2147482998\""},
	"+999999999999999999e2147483627":                       {storedSpellingRef, "999999999999999999e2147483627 (DecimalExponent), stored as \"99999999999999999900e2147483625\""},
	"+999999999999999999e2147483647":                       {storedSpellingRef, "999999999999999999e2147483647 (DecimalExponent), stored as \"9999999999999999990e2147483646\""},
	"0e-2147483648":                                        {zeroMinInt32Ref, "err"},
	"0e2147483648":                                         {zeroMinInt32Ref, "err"},
	"-0e-2147483648":                                       {zeroMinInt32Ref, "err"},
	"-0e2147483648":                                        {zeroMinInt32Ref, "err"},
	"+0e-2147483648":                                       {zeroMinInt32Ref, "err"},
	"+0e2147483648":                                        {zeroMinInt32Ref, "err"},
	"01e-2147483648":                                       {negMinInt32Ref, "err"},
	"1e-2147483648":                                        {negMinInt32Ref, "err"},
	"5e-2147483648":                                        {negMinInt32Ref, "err"},
	"7e-2147483648":                                        {negMinInt32Ref, "err"},
	"8e-2147483648":                                        {negMinInt32Ref, "err"},
	"12e-2147483648":                                       {negMinInt32Ref, "err"},
	"123e-2147483648":                                      {negMinInt32Ref, "err"},
	"999e-2147483648":                                      {negMinInt32Ref, "err"},
	"1000e-2147483648":                                     {negMinInt32Ref, "err"},
	"1024e-2147483648":                                     {negMinInt32Ref, "err"},
	"123456789e-2147483648":                                {negMinInt32Ref, "err"},
	"999999999999999999e-2147483648":                       {negMinInt32Ref, "err"},
	"1000000000000000000e-2147483648":                      {negMinInt32Ref, "err"},
	"10000000000000000000e-2147483648":                     {negMinInt32Ref, "err"},
	"100000000000000000000e-2147483648":                    {negMinInt32Ref, "err"},
	"9223372036854775e-2147483648":                         {negMinInt32Ref, "err"},
	"9223372036854776e-2147483648":                         {negMinInt32Ref, "err"},
	"9223372036854775806e-2147483648":                      {negMinInt32Ref, "err"},
	"9223372036854775807e-2147483648":                      {negMinInt32Ref, "err"},
	"9223372036854775808e-2147483648":                      {negMinInt32Ref, "err"},
	"18446744073709551615e-2147483648":                     {negMinInt32Ref, "err"},
	"18446744073709551616e-2147483648":                     {negMinInt32Ref, "err"},
	"123456789012345678901e-2147483648":                    {negMinInt32Ref, "err"},
	"1234567890123456789012345678901234567890e-2147483648": {negMinInt32Ref, "err"},
	"10000000000000000000000000000000000000000000000000e-2147483648":                                                   {negMinInt32Ref, "err"},
	"9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e-2147483648": {negMinInt32Ref, "err"},
	"1.e-2147483648":                                                  {negMinInt32Ref, "err"},
	"9223372036854775.808e-2147483648":                                {negMinInt32Ref, "err"},
	"-01e-2147483648":                                                 {negMinInt32Ref, "err"},
	"-1e-2147483648":                                                  {negMinInt32Ref, "err"},
	"-5e-2147483648":                                                  {negMinInt32Ref, "err"},
	"-7e-2147483648":                                                  {negMinInt32Ref, "err"},
	"-8e-2147483648":                                                  {negMinInt32Ref, "err"},
	"-12e-2147483648":                                                 {negMinInt32Ref, "err"},
	"-123e-2147483648":                                                {negMinInt32Ref, "err"},
	"-999e-2147483648":                                                {negMinInt32Ref, "err"},
	"-1000e-2147483648":                                               {negMinInt32Ref, "err"},
	"-1024e-2147483648":                                               {negMinInt32Ref, "err"},
	"-123456789e-2147483648":                                          {negMinInt32Ref, "err"},
	"-999999999999999999e-2147483648":                                 {negMinInt32Ref, "err"},
	"-1000000000000000000e-2147483648":                                {negMinInt32Ref, "err"},
	"-10000000000000000000e-2147483648":                               {negMinInt32Ref, "err"},
	"-100000000000000000000e-2147483648":                              {negMinInt32Ref, "err"},
	"-9223372036854775e-2147483648":                                   {negMinInt32Ref, "err"},
	"-9223372036854776e-2147483648":                                   {negMinInt32Ref, "err"},
	"-9223372036854775806e-2147483648":                                {negMinInt32Ref, "err"},
	"-9223372036854775807e-2147483648":                                {negMinInt32Ref, "err"},
	"-9223372036854775808e-2147483648":                                {negMinInt32Ref, "err"},
	"-18446744073709551615e-2147483648":                               {negMinInt32Ref, "err"},
	"-18446744073709551616e-2147483648":                               {negMinInt32Ref, "err"},
	"-123456789012345678901e-2147483648":                              {negMinInt32Ref, "err"},
	"-1234567890123456789012345678901234567890e-2147483648":           {negMinInt32Ref, "err"},
	"-10000000000000000000000000000000000000000000000000e-2147483648": {negMinInt32Ref, "err"},
	"-9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e-2147483648": {negMinInt32Ref, "err"},
	"-1.e-2147483648":                                                 {negMinInt32Ref, "err"},
	"+01e-2147483648":                                                 {negMinInt32Ref, "err"},
	"+1e-2147483648":                                                  {negMinInt32Ref, "err"},
	"+5e-2147483648":                                                  {negMinInt32Ref, "err"},
	"+7e-2147483648":                                                  {negMinInt32Ref, "err"},
	"+8e-2147483648":                                                  {negMinInt32Ref, "err"},
	"+12e-2147483648":                                                 {negMinInt32Ref, "err"},
	"+123e-2147483648":                                                {negMinInt32Ref, "err"},
	"+999e-2147483648":                                                {negMinInt32Ref, "err"},
	"+1000e-2147483648":                                               {negMinInt32Ref, "err"},
	"+1024e-2147483648":                                               {negMinInt32Ref, "err"},
	"+123456789e-2147483648":                                          {negMinInt32Ref, "err"},
	"+999999999999999999e-2147483648":                                 {negMinInt32Ref, "err"},
	"+1000000000000000000e-2147483648":                                {negMinInt32Ref, "err"},
	"+10000000000000000000e-2147483648":                               {negMinInt32Ref, "err"},
	"+100000000000000000000e-2147483648":                              {negMinInt32Ref, "err"},
	"+9223372036854775e-2147483648":                                   {negMinInt32Ref, "err"},
	"+9223372036854776e-2147483648":                                   {negMinInt32Ref, "err"},
	"+9223372036854775806e-2147483648":                                {negMinInt32Ref, "err"},
	"+9223372036854775807e-2147483648":                                {negMinInt32Ref, "err"},
	"+9223372036854775808e-2147483648":                                {negMinInt32Ref, "err"},
	"+18446744073709551615e-2147483648":                               {negMinInt32Ref, "err"},
	"+18446744073709551616e-2147483648":                               {negMinInt32Ref, "err"},
	"+123456789012345678901e-2147483648":                              {negMinInt32Ref, "err"},
	"+1234567890123456789012345678901234567890e-2147483648":           {negMinInt32Ref, "err"},
	"+10000000000000000000000000000000000000000000000000e-2147483648": {negMinInt32Ref, "err"},
	"+9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e-2147483648": {negMinInt32Ref, "err"},
	"+1.e-2147483648":                                     {negMinInt32Ref, "err"},
	"+9223372036854775.808e-2147483648":                   {negMinInt32Ref, "err"},
	"01e2147483648":                                       {posMinInt32Ref, "err"},
	"1e2147483648":                                        {posMinInt32Ref, "err"},
	"5e2147483648":                                        {posMinInt32Ref, "err"},
	"7e2147483648":                                        {posMinInt32Ref, "err"},
	"8e2147483648":                                        {posMinInt32Ref, "err"},
	"12e2147483648":                                       {posMinInt32Ref, "err"},
	"123e2147483648":                                      {posMinInt32Ref, "err"},
	"999e2147483648":                                      {posMinInt32Ref, "err"},
	"1000e2147483648":                                     {posMinInt32Ref, "err"},
	"1024e2147483648":                                     {posMinInt32Ref, "err"},
	"123456789e2147483648":                                {posMinInt32Ref, "err"},
	"999999999999999999e2147483648":                       {posMinInt32Ref, "err"},
	"1000000000000000000e2147483648":                      {posMinInt32Ref, "err"},
	"10000000000000000000e2147483648":                     {posMinInt32Ref, "err"},
	"100000000000000000000e2147483648":                    {posMinInt32Ref, "err"},
	"9223372036854775e2147483648":                         {posMinInt32Ref, "err"},
	"9223372036854776e2147483648":                         {posMinInt32Ref, "err"},
	"9223372036854775806e2147483648":                      {posMinInt32Ref, "err"},
	"9223372036854775807e2147483648":                      {posMinInt32Ref, "err"},
	"9223372036854775808e2147483648":                      {posMinInt32Ref, "err"},
	"18446744073709551615e2147483648":                     {posMinInt32Ref, "err"},
	"18446744073709551616e2147483648":                     {posMinInt32Ref, "err"},
	"123456789012345678901e2147483648":                    {posMinInt32Ref, "err"},
	"1234567890123456789012345678901234567890e2147483648": {posMinInt32Ref, "err"},
	"10000000000000000000000000000000000000000000000000e2147483648":                                                   {posMinInt32Ref, "err"},
	"9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e2147483648": {posMinInt32Ref, "err"},
	"1.e2147483648":                                        {posMinInt32Ref, "err"},
	"9223372036854775.808e2147483648":                      {posMinInt32Ref, "err"},
	"-01e2147483648":                                       {posMinInt32Ref, "err"},
	"-1e2147483648":                                        {posMinInt32Ref, "err"},
	"-5e2147483648":                                        {posMinInt32Ref, "err"},
	"-7e2147483648":                                        {posMinInt32Ref, "err"},
	"-8e2147483648":                                        {posMinInt32Ref, "err"},
	"-12e2147483648":                                       {posMinInt32Ref, "err"},
	"-123e2147483648":                                      {posMinInt32Ref, "err"},
	"-999e2147483648":                                      {posMinInt32Ref, "err"},
	"-1000e2147483648":                                     {posMinInt32Ref, "err"},
	"-1024e2147483648":                                     {posMinInt32Ref, "err"},
	"-123456789e2147483648":                                {posMinInt32Ref, "err"},
	"-999999999999999999e2147483648":                       {posMinInt32Ref, "err"},
	"-1000000000000000000e2147483648":                      {posMinInt32Ref, "err"},
	"-10000000000000000000e2147483648":                     {posMinInt32Ref, "err"},
	"-100000000000000000000e2147483648":                    {posMinInt32Ref, "err"},
	"-9223372036854775e2147483648":                         {posMinInt32Ref, "err"},
	"-9223372036854776e2147483648":                         {posMinInt32Ref, "err"},
	"-9223372036854775806e2147483648":                      {posMinInt32Ref, "err"},
	"-9223372036854775807e2147483648":                      {posMinInt32Ref, "err"},
	"-9223372036854775808e2147483648":                      {posMinInt32Ref, "err"},
	"-18446744073709551615e2147483648":                     {posMinInt32Ref, "err"},
	"-18446744073709551616e2147483648":                     {posMinInt32Ref, "err"},
	"-123456789012345678901e2147483648":                    {posMinInt32Ref, "err"},
	"-1234567890123456789012345678901234567890e2147483648": {posMinInt32Ref, "err"},
	"-10000000000000000000000000000000000000000000000000e2147483648":                                                   {posMinInt32Ref, "err"},
	"-9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e2147483648": {posMinInt32Ref, "err"},
	"-1.e2147483648":                                       {posMinInt32Ref, "err"},
	"+01e2147483648":                                       {posMinInt32Ref, "err"},
	"+1e2147483648":                                        {posMinInt32Ref, "err"},
	"+5e2147483648":                                        {posMinInt32Ref, "err"},
	"+7e2147483648":                                        {posMinInt32Ref, "err"},
	"+8e2147483648":                                        {posMinInt32Ref, "err"},
	"+12e2147483648":                                       {posMinInt32Ref, "err"},
	"+123e2147483648":                                      {posMinInt32Ref, "err"},
	"+999e2147483648":                                      {posMinInt32Ref, "err"},
	"+1000e2147483648":                                     {posMinInt32Ref, "err"},
	"+1024e2147483648":                                     {posMinInt32Ref, "err"},
	"+123456789e2147483648":                                {posMinInt32Ref, "err"},
	"+999999999999999999e2147483648":                       {posMinInt32Ref, "err"},
	"+1000000000000000000e2147483648":                      {posMinInt32Ref, "err"},
	"+10000000000000000000e2147483648":                     {posMinInt32Ref, "err"},
	"+100000000000000000000e2147483648":                    {posMinInt32Ref, "err"},
	"+9223372036854775e2147483648":                         {posMinInt32Ref, "err"},
	"+9223372036854776e2147483648":                         {posMinInt32Ref, "err"},
	"+9223372036854775806e2147483648":                      {posMinInt32Ref, "err"},
	"+9223372036854775807e2147483648":                      {posMinInt32Ref, "err"},
	"+9223372036854775808e2147483648":                      {posMinInt32Ref, "err"},
	"+18446744073709551615e2147483648":                     {posMinInt32Ref, "err"},
	"+18446744073709551616e2147483648":                     {posMinInt32Ref, "err"},
	"+123456789012345678901e2147483648":                    {posMinInt32Ref, "err"},
	"+1234567890123456789012345678901234567890e2147483648": {posMinInt32Ref, "err"},
	"+10000000000000000000000000000000000000000000000000e2147483648":                                                   {posMinInt32Ref, "err"},
	"+9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999e2147483648": {posMinInt32Ref, "err"},
	"+1.e2147483648":                    {posMinInt32Ref, "err"},
	"+9223372036854775.808e2147483648":  {posMinInt32Ref, "err"},
	"1000000000000000000e-2147483649":   {wrappedRef, "1e2147483665 (DecimalExponent), stored as \"1e-2147483631\""},
	"9223372036854775806e-2147483649":   {wrappedRef, "9223372036854775806e2147483647 (DecimalExponent), stored as \"92233720368547758060e2147483646\""},
	"9223372036854775807e-2147483649":   {wrappedRef, "9223372036854775807e2147483647 (DecimalExponent), stored as \"92233720368547758070e2147483646\""},
	"9223372036854775.807e-2147483649":  {wrappedRef, "9223372036854775807e2147483644 (DecimalExponent), stored as \"92233720368547758070e2147483643\""},
	"9223372036854775.807e-2147483648":  {wrappedRef, "9223372036854775807e2147483645 (DecimalExponent), stored as \"922337203685477580700e2147483643\""},
	"9223372036854775.807e-2147483647":  {wrappedRef, "9223372036854775807e2147483646 (DecimalExponent)"},
	"9223372036854775.807e-2147483646":  {wrappedRef, "9223372036854775807e2147483647 (DecimalExponent), stored as \"92233720368547758070e2147483646\""},
	"-1000000000000000000e-2147483649":  {wrappedRef, "-1e2147483665 (DecimalExponent), stored as \"-1e-2147483631\""},
	"-9223372036854775806e-2147483649":  {wrappedRef, "-9223372036854775806e2147483647 (DecimalExponent), stored as \"-92233720368547758060e2147483646\""},
	"-9223372036854775807e-2147483649":  {wrappedRef, "-9223372036854775807e2147483647 (DecimalExponent), stored as \"-92233720368547758070e2147483646\""},
	"-9223372036854775808e-2147483649":  {wrappedRef, "-9223372036854775808e2147483647 (DecimalExponent), stored as \"-92233720368547758080e2147483646\""},
	"-9223372036854775.807e-2147483649": {wrappedRef, "-9223372036854775807e2147483644 (DecimalExponent), stored as \"-92233720368547758070e2147483643\""},
	"-9223372036854775.807e-2147483648": {wrappedRef, "-9223372036854775807e2147483645 (DecimalExponent), stored as \"-922337203685477580700e2147483643\""},
	"-9223372036854775.807e-2147483647": {wrappedRef, "-9223372036854775807e2147483646 (DecimalExponent)"},
	"-9223372036854775.807e-2147483646": {wrappedRef, "-9223372036854775807e2147483647 (DecimalExponent), stored as \"-92233720368547758070e2147483646\""},
	"-9223372036854775.808e-2147483649": {wrappedRef, "-9223372036854775808e2147483644 (DecimalExponent), stored as \"-92233720368547758080e2147483643\""},
	"-9223372036854775.808e-2147483648": {wrappedRef, "-9223372036854775808e2147483645 (DecimalExponent), stored as \"-922337203685477580800e2147483643\""},
	"-9223372036854775.808e-2147483647": {wrappedRef, "-9223372036854775808e2147483646 (DecimalExponent)"},
	"-9223372036854775.808e-2147483646": {wrappedRef, "-9223372036854775808e2147483647 (DecimalExponent), stored as \"-92233720368547758080e2147483646\""},
	"+1000000000000000000e-2147483649":  {wrappedRef, "1e2147483665 (DecimalExponent), stored as \"1e-2147483631\""},
	"+9223372036854775806e-2147483649":  {wrappedRef, "9223372036854775806e2147483647 (DecimalExponent), stored as \"92233720368547758060e2147483646\""},
	"+9223372036854775807e-2147483649":  {wrappedRef, "9223372036854775807e2147483647 (DecimalExponent), stored as \"92233720368547758070e2147483646\""},
	"+9223372036854775.807e-2147483649": {wrappedRef, "9223372036854775807e2147483644 (DecimalExponent), stored as \"92233720368547758070e2147483643\""},
	"+9223372036854775.807e-2147483648": {wrappedRef, "9223372036854775807e2147483645 (DecimalExponent), stored as \"922337203685477580700e2147483643\""},
	"+9223372036854775.807e-2147483647": {wrappedRef, "9223372036854775807e2147483646 (DecimalExponent)"},
	"+9223372036854775.807e-2147483646": {wrappedRef, "9223372036854775807e2147483647 (DecimalExponent), stored as \"92233720368547758070e2147483646\""},
	"1000000000000000000e2147483646":    {storedHangsRef, "1e2147483664 (DecimalExponent), stored as \"100e-2147483634\""},
	"1000000000000000000e2147483647":    {storedHangsRef, "1e2147483665 (DecimalExponent), stored as \"1e-2147483631\""},
	"9223372036854775806e2147483000":    {storedHangsRef, "9223372036854775806e2147483000 (DecimalExponent), stored as \"922337203685477580600e2147482998\""},
	"9223372036854775806e2147483627":    {storedHangsRef, "9223372036854775806e2147483627 (DecimalExponent), stored as \"922337203685477580600e2147483625\""},
	"9223372036854775806e2147483647":    {storedHangsRef, "9223372036854775806e2147483647 (DecimalExponent), stored as \"92233720368547758060e2147483646\""},
	"9223372036854775807e2147483000":    {storedHangsRef, "9223372036854775807e2147483000 (DecimalExponent), stored as \"922337203685477580700e2147482998\""},
	"9223372036854775807e2147483627":    {storedHangsRef, "9223372036854775807e2147483627 (DecimalExponent), stored as \"922337203685477580700e2147483625\""},
	"9223372036854775807e2147483647":    {storedHangsRef, "9223372036854775807e2147483647 (DecimalExponent), stored as \"92233720368547758070e2147483646\""},
	"9223372036854775.807e2147483000":   {storedHangsRef, "9223372036854775807e2147482997 (DecimalExponent), stored as \"922337203685477580700e2147482995\""},
	"9223372036854775.807e2147483627":   {storedHangsRef, "9223372036854775807e2147483624 (DecimalExponent), stored as \"922337203685477580700e2147483622\""},
	"9223372036854775.807e2147483647":   {storedHangsRef, "9223372036854775807e2147483644 (DecimalExponent), stored as \"92233720368547758070e2147483643\""},
	"9223372036854775.807e2147483648":   {storedHangsRef, "9223372036854775807e2147483645 (DecimalExponent), stored as \"922337203685477580700e2147483643\""},
	"-1000000000000000000e2147483646":   {storedHangsRef, "-1e2147483664 (DecimalExponent), stored as \"-100e-2147483634\""},
	"-1000000000000000000e2147483647":   {storedHangsRef, "-1e2147483665 (DecimalExponent), stored as \"-1e-2147483631\""},
	"-9223372036854775806e2147483000":   {storedHangsRef, "-9223372036854775806e2147483000 (DecimalExponent), stored as \"-922337203685477580600e2147482998\""},
	"-9223372036854775806e2147483627":   {storedHangsRef, "-9223372036854775806e2147483627 (DecimalExponent), stored as \"-922337203685477580600e2147483625\""},
	"-9223372036854775806e2147483647":   {storedHangsRef, "-9223372036854775806e2147483647 (DecimalExponent), stored as \"-92233720368547758060e2147483646\""},
	"-9223372036854775807e2147483000":   {storedHangsRef, "-9223372036854775807e2147483000 (DecimalExponent), stored as \"-922337203685477580700e2147482998\""},
	"-9223372036854775807e2147483627":   {storedHangsRef, "-9223372036854775807e2147483627 (DecimalExponent), stored as \"-922337203685477580700e2147483625\""},
	"-9223372036854775807e2147483647":   {storedHangsRef, "-9223372036854775807e2147483647 (DecimalExponent), stored as \"-92233720368547758070e2147483646\""},
	"-9223372036854775808e2147483000":   {storedHangsRef, "-9223372036854775808e2147483000 (DecimalExponent), stored as \"-922337203685477580800e2147482998\""},
	"-9223372036854775808e2147483627":   {storedHangsRef, "-9223372036854775808e2147483627 (DecimalExponent), stored as \"-922337203685477580800e2147483625\""},
	"-9223372036854775808e2147483647":   {storedHangsRef, "-9223372036854775808e2147483647 (DecimalExponent), stored as \"-92233720368547758080e2147483646\""},
	"-9223372036854775.807e2147483000":  {storedHangsRef, "-9223372036854775807e2147482997 (DecimalExponent), stored as \"-922337203685477580700e2147482995\""},
	"-9223372036854775.807e2147483627":  {storedHangsRef, "-9223372036854775807e2147483624 (DecimalExponent), stored as \"-922337203685477580700e2147483622\""},
	"-9223372036854775.807e2147483647":  {storedHangsRef, "-9223372036854775807e2147483644 (DecimalExponent), stored as \"-92233720368547758070e2147483643\""},
	"-9223372036854775.807e2147483648":  {storedHangsRef, "-9223372036854775807e2147483645 (DecimalExponent), stored as \"-922337203685477580700e2147483643\""},
	"-9223372036854775.808e2147483000":  {storedHangsRef, "-9223372036854775808e2147482997 (DecimalExponent), stored as \"-922337203685477580800e2147482995\""},
	"-9223372036854775.808e2147483627":  {storedHangsRef, "-9223372036854775808e2147483624 (DecimalExponent), stored as \"-922337203685477580800e2147483622\""},
	"-9223372036854775.808e2147483647":  {storedHangsRef, "-9223372036854775808e2147483644 (DecimalExponent), stored as \"-92233720368547758080e2147483643\""},
	"-9223372036854775.808e2147483648":  {storedHangsRef, "-9223372036854775808e2147483645 (DecimalExponent), stored as \"-922337203685477580800e2147483643\""},
	"+1000000000000000000e2147483646":   {storedHangsRef, "1e2147483664 (DecimalExponent), stored as \"100e-2147483634\""},
	"+1000000000000000000e2147483647":   {storedHangsRef, "1e2147483665 (DecimalExponent), stored as \"1e-2147483631\""},
	"+9223372036854775806e2147483000":   {storedHangsRef, "9223372036854775806e2147483000 (DecimalExponent), stored as \"922337203685477580600e2147482998\""},
	"+9223372036854775806e2147483627":   {storedHangsRef, "9223372036854775806e2147483627 (DecimalExponent), stored as \"922337203685477580600e2147483625\""},
	"+9223372036854775806e2147483647":   {storedHangsRef, "9223372036854775806e2147483647 (DecimalExponent), stored as \"92233720368547758060e2147483646\""},
	"+9223372036854775807e2147483000":   {storedHangsRef, "9223372036854775807e2147483000 (DecimalExponent), stored as \"922337203685477580700e2147482998\""},
	"+9223372036854775807e2147483627":   {storedHangsRef, "9223372036854775807e2147483627 (DecimalExponent), stored as \"922337203685477580700e2147483625\""},
	"+9223372036854775807e2147483647":   {storedHangsRef, "9223372036854775807e2147483647 (DecimalExponent), stored as \"92233720368547758070e2147483646\""},
	"+9223372036854775.807e2147483000":  {storedHangsRef, "9223372036854775807e2147482997 (DecimalExponent), stored as \"922337203685477580700e2147482995\""},
	"+9223372036854775.807e2147483627":  {storedHangsRef, "9223372036854775807e2147483624 (DecimalExponent), stored as \"922337203685477580700e2147483622\""},
	"+9223372036854775.807e2147483647":  {storedHangsRef, "9223372036854775807e2147483644 (DecimalExponent), stored as \"92233720368547758070e2147483643\""},
	"+9223372036854775.807e2147483648":  {storedHangsRef, "9223372036854775807e2147483645 (DecimalExponent), stored as \"922337203685477580700e2147483643\""},
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

// parseHangs reports whether ParseQuantity hangs on input in this tree. It
// mirrors the mechanism: an input with a decimal exponent that ParseQuantity
// cannot keep in an int64Amount goes through inf.Dec, whose round to the nano
// scale shifts by 10^(9 - scale) with the shift computed in int32; a shift of
// more than 2^20 digits does not return. On every run TestQuantityLawParse
// checks it against the release: an input it reports must have hung on
// v1.37.0 too. Whether it reports exactly the inputs that hang in this tree is
// checked by TestQuantityLawParseHangPredicate, which decodes every grid input
// and stored spelling in child processes and runs only with
// QUANTITY_LAW_VERIFY_HANGS=1 (about four minutes).
func parseHangs(input string) bool {
	m := decimalExponentInput.FindStringSubmatch(input)
	if m == nil {
		return false
	}
	sign, whole, fraction, expLiteral := m[1], m[2], m[3], m[4]
	exp64, err := strconv.ParseInt(expLiteral, 10, 64)
	if err != nil {
		return false // ErrSuffix
	}
	exp := int32(exp64)
	if strings.Trim(whole+fraction, "0") == "" {
		return false // zero is not rounded
	}
	num := strings.TrimLeft(whole, "0")
	if num == "" {
		num = "0"
	}
	if maxInt64Factors-int32(len(num)+len(fraction)) >= 0 && exp-int32(len(fraction)) >= int32(Nano) {
		if u, err := strconv.ParseUint(num+fraction, 10, 64); err == nil && (u <= math.MaxInt64 || (sign == "-" && u == 1<<63)) {
			return false // int64Amount fast path
		}
	}
	if exp == math.MinInt32 {
		return false // ErrSuffix before rounding
	}
	scale := int32(len(fraction)) + -exp
	shift := int32(Nano.infScale()) - scale
	if shift == math.MinInt32 {
		return false // negating the shift wraps to itself; inf.Dec treats it as a no-op
	}
	if shift < 0 {
		shift = -shift
	}
	return shift > 1<<20
}

// int32NarrowingRef is the shared reason for the frozen decodes of decimal
// exponents: the parser keeps the exponent, and the scale it becomes after the
// fraction digits are taken off, in int32, so one outside int32 is truncated
// or wraps, and one outside int64 is rejected.
const int32NarrowingRef = "#142395/#141166: a decimal exponent outside int32, before or after the fraction digits are taken off, is truncated or wraps; outside int64 it is rejected"

// frozenMechanism returns why ParseQuantity's decode of input may be wrong
// and still frozen, or "" when no known mechanism makes it wrong. A frozen
// decode is one that matches the release and disagrees with exact.Parse, so
// without this check a fault in exact.Parse on any input the library decodes
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
