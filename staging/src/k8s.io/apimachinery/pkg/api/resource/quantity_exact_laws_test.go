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
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"math/big"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource/internal/exact"
)

// TestQuantityLawParse checks ParseQuantity on every input of the parse grid
// against two independent references: exact.Parse (is the decode correct?)
// and the recorded decode of the last release (is it compatible with what an
// apiserver of that release may have stored?). The quadrant each input falls
// in decides what the input needs; see the comment above parseApprovedChanges.
// Inputs that parseHangs reports are not decoded, apart from the listed
// representatives, which run in a child process.
func TestQuantityLawParse(t *testing.T) {
	recorded := loadDecodeCorpus(t)
	visited := map[string]string{}
	counts := map[string]int{}
	for _, input := range parseGridInputs() {
		id := "parse/" + input
		t.Run(id, func(t *testing.T) {
			old, ok := recorded[input]
			if !ok {
				t.Fatalf("no %s decode recorded for %q; regenerate %s", decodeCorpusRelease, input, decodeCorpusFile)
			}
			if parseHangs(input) {
				counts["hang"]++
				visited[input] = "hang"
				if old.Outcome != "hang" {
					t.Errorf("%q hangs in this tree but %s decoded it (%s %s): a hang regression", input, decodeCorpusRelease, old.Outcome, old.Value)
				}
				if isKnownHang(id) {
					runHangLaw(t, id, "parse")
				}
				return
			}
			v, err := classifyParse(input, old)
			if err != nil {
				t.Fatal(err)
			}
			quadrant := v.quadrant
			counts[quadrant]++
			visited[input] = quadrant
			detail := v.String()
			if strings.HasPrefix(quadrant, "frozen") && frozenMechanism(input) == "" {
				t.Errorf("%q decodes as %s did and differs from exact.Parse (%s), but no known mechanism makes this decode wrong: is exact.Parse right?", input, decodeCorpusRelease, detail)
			}
			switch quadrant {
			case "unjudged/incompatible":
				t.Errorf("%q decodes differently from %s (%s), and exact.Parse cannot judge it", input, decodeCorpusRelease, detail)
			case "unjudged/stored-spelling":
				t.Errorf("%q: the stored spelling does not decode back (%s); exact.Parse cannot judge the decode itself", input, detail)
			case "correct/incompatible":
				if _, ok := parseApprovedChanges[input]; !ok {
					t.Errorf("%q decodes differently from %s, which may have stored this spelling (%s); list it in parseApprovedChanges only if the change is intended", input, decodeCorpusRelease, detail)
				}
			case "wrong/incompatible", "wrong/no-1.37-decode", "wrong/stored-spelling", "frozen/stored-spelling":
				switch entry, listed := parseRegressions[input]; {
				case listed && v.observed() != entry.observed:
					t.Errorf("%q: decode is now %s, listed in parseRegressions with %s", input, v.observed(), entry.observed)
				case listed:
					t.Logf("%s: expected deviation, still present: %s", id, detail)
				case quadrant == "wrong/incompatible":
					t.Errorf("%q decodes differently from %s, which may have stored this spelling, and wrongly (%s); not listed in parseRegressions", input, decodeCorpusRelease, detail)
				default:
					t.Errorf("%q decodes wrongly (%s); not listed in parseRegressions", input, detail)
				}
			}
		})
	}
	checkParseLists(t, visited)
	t.Logf("parse grid quadrants: %v", counts)
	if allSubtestsRan() && !maps.Equal(counts, parseQuadrantCounts) {
		t.Errorf("parse grid quadrants %v, recorded %v: a decode moved between quadrants; update parseQuadrantCounts once the lists explain it", counts, parseQuadrantCounts)
	}
	checkVisited(t, hasPrefix("parse/"))
}

// parseVerdict is what TestQuantityLawParse found for one input.
type parseVerdict struct {
	cur, want, old decodeRecord
	// judged is false when the input is outside exact.Parse's domain.
	judged bool
	// reencoded is the JSON spelling of the decoded quantity, the form an
	// apiserver stores; roundTrip says why it does not decode back to the same
	// value, or is empty when it does.
	reencoded, roundTrip string
	quadrant             string
}

// observed is what parseRegressions records for an entry: the decode, and the
// spelling it is stored as when that does not decode back.
func (v parseVerdict) observed() string {
	s := describeDecode(v.cur)
	if v.roundTrip != "" {
		s += ", stored as " + strconv.Quote(v.reencoded)
	}
	return s
}

func (v parseVerdict) String() string {
	s := fmt.Sprintf("decode %s, exact %s, %s %s", describeDecode(v.cur), describeDecode(v.want), decodeCorpusRelease, describeDecode(v.old))
	if v.roundTrip != "" {
		s += ", " + v.roundTrip
	}
	return s
}

// classifyParse decodes input, re-encodes the result the way it is stored, and
// decodes that spelling again. A decode is correct when it matches exact.Parse
// and its stored spelling decodes back to the same value.
func classifyParse(input string, old decodeRecord) (parseVerdict, error) {
	v := parseVerdict{old: old}
	var q Quantity
	_, timedOut, panicked, _ := runLawCase(func() string {
		var err error
		q, err = ParseQuantity(input)
		if err != nil {
			v.cur = decodeRecord{Input: input, Outcome: "err"}
			return ""
		}
		v.cur = decodeRecord{Input: input, Outcome: "ok", Value: decodeValue(&q), Format: q.Format}
		encoded, err := json.Marshal(q)
		if err != nil {
			v.roundTrip = fmt.Sprintf("json.Marshal: %v", err)
			return ""
		}
		v.reencoded = strings.Trim(string(encoded), `"`)
		return ""
	})
	switch {
	case timedOut:
		return v, fmt.Errorf("decoding or re-encoding %q did not return within %s; parseHangs does not cover it", input, caseTimeout)
	case panicked != "":
		return v, fmt.Errorf("decoding or re-encoding %q: %s", input, panicked)
	}
	if v.cur.Outcome == "ok" && v.roundTrip == "" {
		switch again := v.reencoded; {
		case parseHangs(again):
			// Predicted, not observed: TestQuantityLawParseHangPredicate checks the
			// predicate against real decodes.
			v.roundTrip = fmt.Sprintf("stored as %q, which parseHangs predicts hangs on decode", again)
		default:
			// The stored spelling is a second decode, so it runs under the
			// timeout like the first.
			var back decodeRecord
			_, timedOut, panicked, _ := runLawCase(func() string {
				back = decodeOne(again)
				return ""
			})
			switch {
			case timedOut:
				return v, fmt.Errorf("decoding the stored spelling %q of %q did not return within %s; parseHangs does not cover it", again, input, caseTimeout)
			case panicked != "":
				return v, fmt.Errorf("decoding the stored spelling %q of %q: %s", again, input, panicked)
			// Only the value has to come back: a zero decodes as DecimalSI
			// whatever format it was written in.
			case back.Outcome != "ok" || back.Value != v.cur.Value:
				v.roundTrip = fmt.Sprintf("stored as %q, which decodes to %s", again, describeDecode(back))
			}
		}
	}
	// exact.Parse runs under the timeout too: its cost grows with the input.
	_, timedOut, panicked, unjudged := runLawCase(func() string {
		v.want, v.judged = exactDecode(input)
		return ""
	})
	switch {
	case timedOut:
		return v, fmt.Errorf("exact.Parse(%q) did not return within %s", input, caseTimeout)
	case panicked != "" || unjudged != "":
		return v, fmt.Errorf("exact.Parse(%q): %s%s", input, panicked, unjudged)
	}
	v.quadrant = parseQuadrant(v.judged, v.judged && v.cur == v.want, v.roundTrip == "", old, v.cur)
	return v, nil
}

// parseQuadrant names the quadrant of a decode; see parseApprovedChanges. A
// decode is correct when it matches exact.Parse and its stored spelling decodes
// back to it. "frozen" is only for a decode that is itself wrong in the same way
// as in the release: a stored spelling that stops decoding back is a regression
// even when the decode still matches the release. For an input outside
// exact.Parse's domain the decode itself cannot be judged, but its stored
// spelling must still decode back to it, which needs no reference: one that
// does not is "unjudged/stored-spelling".
func parseQuadrant(judged, decodeCorrect, roundTrips bool, old, cur decodeRecord) string {
	correct := decodeCorrect && roundTrips
	switch {
	case !judged && !roundTrips:
		return "unjudged/stored-spelling"
	case !judged && old == cur:
		return "unjudged/compatible"
	case !judged:
		return "unjudged/incompatible"
	case old.Outcome == "hang" && correct:
		return "correct/no-1.37-decode"
	case old.Outcome == "hang":
		return "wrong/no-1.37-decode"
	case old == cur && correct:
		return "correct/compatible"
	case old == cur && decodeCorrect:
		return "wrong/stored-spelling"
	case old == cur && !roundTrips:
		return "frozen/stored-spelling"
	case old == cur:
		return "frozen"
	case correct:
		return "correct/incompatible"
	default:
		return "wrong/incompatible"
	}
}

// checkParseLists fails every list entry that is not in the grid or not in
// the quadrant its list names.
func checkParseLists(t *testing.T, visited map[string]string) {
	t.Helper()
	if !allSubtestsRan() {
		return
	}
	check := func(list, input string, want ...string) {
		got, ok := visited[input]
		switch {
		case !ok:
			t.Errorf("%s entry %q is not a parse grid input", list, input)
		case !slices.Contains(want, got):
			t.Errorf("%s entry %q is stale: its decode is now %s", list, input, got)
		}
	}
	for input := range parseApprovedChanges {
		check("parseApprovedChanges", input, "correct/incompatible")
	}
	for input := range parseRegressions {
		check("parseRegressions", input, "wrong/incompatible", "wrong/no-1.37-decode", "wrong/stored-spelling", "frozen/stored-spelling")
	}
	for input := range parseFrozenExamples {
		check("parseFrozenExamples", input, "frozen")
	}
}

// exactDecode returns exact.Parse's verdict on input in the decodeRecord form.
// judged is false when the input's exponent is outside exact.Parse's domain.
func exactDecode(input string) (want decodeRecord, judged bool) {
	e, err := exact.Parse(input)
	switch {
	case errors.Is(err, exact.ErrOutOfDomain):
		return decodeRecord{}, false
	case err != nil:
		return decodeRecord{Input: input, Outcome: "err"}, true
	}
	value := "0"
	if !e.IsZero() {
		value = e.Mantissa().String() + "e" + strconv.FormatInt(e.Exponent(), 10)
	}
	return decodeRecord{Input: input, Outcome: "ok", Value: value, Format: Format(e.Format())}, true
}

func describeDecode(r decodeRecord) string {
	switch r.Outcome {
	case "ok":
		return fmt.Sprintf("%s (%s)", r.Value, r.Format)
	case "":
		return "(not judged)"
	default:
		return r.Outcome
	}
}

// TestQuantityLawParseHangPredicate decodes, in child processes, every grid
// input and every stored spelling TestQuantityLawParse asks parseHangs about,
// and checks that parseHangs reports exactly the ones that hang. It takes a
// few minutes and runs only when QUANTITY_LAW_VERIFY_HANGS=1.
func TestQuantityLawParseHangPredicate(t *testing.T) {
	if os.Getenv("QUANTITY_LAW_VERIFY_HANGS") != "1" {
		t.Skip("set QUANTITY_LAW_VERIFY_HANGS=1 to run")
	}
	recorded := loadDecodeCorpus(t)
	inputs := parseGridInputs()
	for _, input := range parseGridInputs() {
		if parseHangs(input) {
			continue
		}
		if v, err := classifyParse(input, recorded[input]); err == nil && v.reencoded != "" && v.reencoded != input {
			inputs = append(inputs, v.reencoded)
		}
	}
	path := filepath.Join(t.TempDir(), "inputs.json")
	raw, err := json.Marshal(inputs)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	env := []string{decodeCorpusInputsEnv + "=" + path}
	records := make([]decodeRecord, len(inputs))
	workers := runtime.NumCPU()
	done := make(chan error, workers)
	for w := range workers {
		go func() { done <- runDecodeWorker(inputs, workers, w, records, env) }()
	}
	for range workers {
		if err := <-done; err != nil {
			t.Fatal(err)
		}
	}
	hangs := 0
	for i, r := range records {
		hung := r.Outcome == "hang"
		if hung {
			hangs++
		}
		if hung != parseHangs(inputs[i]) {
			t.Errorf("%q: hang observed %t, parseHangs %t", inputs[i], hung, parseHangs(inputs[i]))
		}
	}
	t.Logf("%d of %d inputs hang (%d grid inputs, %d stored spellings)", hangs, len(inputs), len(parseGridInputs()), len(inputs)-len(parseGridInputs()))
}

// buildValueCases builds each value case once. Laws that mutate rebuild from
// the make func instead.
func buildValueCases(t *testing.T) []valueCase {
	t.Helper()
	return valueCases()
}

// pairCases drops quantities that cannot be paired against arbitrary others:
// a MinInt32 scale cannot be represented in inf.Dec, so AsDec (used by Cmp and
// the Add/Sub fallback) changes its value. It is exercised on its own in
// TestQuantityLawMisc and TestQuantityLawKnownGaps instead.
func pairCases(t *testing.T, cases []valueCase) []valueCase {
	t.Helper()
	const unpairable = "ctor/NewScaledQuantity/7/MinInt32"
	out := make([]valueCase, 0, len(cases))
	for _, c := range cases {
		if c.id != unpairable {
			out = append(out, c)
		}
	}
	if len(out) != len(cases)-1 {
		t.Fatalf("value case %q not found", unpairable)
	}
	return out
}

// TestQuantityLawSign checks Sign and IsZero against the exact value, once per
// value case.
func TestQuantityLawSign(t *testing.T) {
	for _, c := range buildValueCases(t) {
		q := c.make()
		o := exactValue(q)
		runLaw(t, "sign/"+c.id, "sign", func() string {
			if got := q.Sign(); got != o.Sign() {
				return fmt.Sprintf("Sign = %d, exact %d", got, o.Sign())
			}
			if got := q.IsZero(); got != o.IsZero() {
				return fmt.Sprintf("IsZero = %t, exact %t", got, o.IsZero())
			}
			return ""
		})
	}
	checkVisited(t, hasPrefix("sign/"))
}

// TestQuantityLawCompare checks Cmp, CmpInt64 and Equal against the exact
// value, on a deterministic diagonal, a seeded random sample, and the
// explicit seam pairs (huge-vs-tiny scales, both backends).
func TestQuantityLawCompare(t *testing.T) {
	cases := pairCases(t, buildValueCases(t))
	// A diagonal, a seeded random sample, and the seam pairs below.
	pairs := make([][2]int, 0, len(cases)*3)
	for i := range cases {
		for j := 0; j < len(cases); j += 7 {
			pairs = append(pairs, [2]int{i, j})
		}
		pairs = append(pairs, [2]int{i, i})
	}
	rng := rand.New(rand.NewSource(141166))
	for range 2000 {
		pairs = append(pairs, [2]int{rng.Intn(len(cases)), rng.Intn(len(cases))})
	}
	// Deterministic seam pairs: every pairing of huge and tiny scales and the
	// int64 rails, so the int32-scale-difference comparison paths are always
	// exercised rather than left to random chance.
	seamIDs := []string{
		"exp/1e2147483647", "exp/1e2147483646", "exp/1e2147483628",
		"exp/1e-19", "exp/1e-9", "exp/1e9", "exp/1e19",
		"suffix/1n", "suffix/1", "suffix/1E", "suffix/8Ei", "suffix/-1.5Ki",
		"digits/1/lo", "digits/40/hi", "ctor/NewScaledQuantity/7/9",
	}
	idx := make(map[string]int, len(cases))
	for i, c := range cases {
		idx[c.id] = i
	}
	for _, id := range seamIDs {
		if _, ok := idx[id]; !ok {
			t.Fatalf("seam value case %q not found", id)
		}
	}
	for _, ai := range seamIDs {
		for _, bi := range seamIDs {
			pairs = append(pairs, [2]int{idx[ai], idx[bi]})
		}
	}
	cmpInt64s := []int64{0, 1, -1, math.MaxInt64, math.MinInt64}
	for _, p := range pairs {
		a := cases[p[0]].make()
		b := cases[p[1]].make()
		ao, bo := exactValue(a), exactValue(b)
		runLaw(t, "cmp/"+cases[p[0]].id+"|"+cases[p[1]].id, "compare", func() string {
			if got := a.Cmp(*b); got != ao.Cmp(bo) {
				return fmt.Sprintf("Cmp = %d, exact %d (a=%s b=%s)", got, ao.Cmp(bo), ao.ExactString(), bo.ExactString())
			}
			if got := a.Equal(*b); got != ao.Equal(bo) {
				return fmt.Sprintf("Equal = %t, exact %t", got, ao.Equal(bo))
			}
			for _, y := range cmpInt64s {
				if got := a.CmpInt64(y); got != ao.CmpInt64(y) {
					return fmt.Sprintf("CmpInt64(%d) = %d, exact %d", y, got, ao.CmpInt64(y))
				}
			}
			return ""
		})
	}
	checkVisited(t, hasPrefix("cmp/"))
}

// arithUnjudgedPairs is how many of the sampled pairs TestQuantityLawArithmetic
// cannot check Add and Sub on.
const arithUnjudgedPairs = 38

// mulFactors are the int64 factors the arithmetic law multiplies by. 1000
// moves a value's exponent by three, which within three of ±2^40 would make
// package exact give up (exact.Unjudgeable); the value cases stay near the
// int32 edges, far from that.
var mulFactors = []int64{0, 1, -1, 7, 1000, math.MaxInt64, math.MinInt64}

// TestQuantityLawArithmetic checks Add, Sub, Neg and Mul against the exact value for
// pairs whose alignment delta package exact can materialize. Extreme alignments
// (the int32 edges) are exercised separately in TestQuantityLawKnownGaps under the
// timeout.
func TestQuantityLawArithmetic(t *testing.T) {
	cases := pairCases(t, buildValueCases(t))
	rng := rand.New(rand.NewSource(141167))
	unjudged := 0
	for range 2000 {
		i, j := rng.Intn(len(cases)), rng.Intn(len(cases))
		a, b := cases[i].make(), cases[j].make()
		ao, bo := exactValue(a), exactValue(b)
		sum, sumOK := ao.Add(bo)
		diff, diffOK := ao.Sub(bo)
		if !sumOK || !diffOK {
			unjudged++
		}
		runLaw(t, "arith/"+cases[i].id+"|"+cases[j].id, "arith", func() string {
			if sumOK {
				r := a.DeepCopy()
				r.Add(*b)
				if got := exactValue(&r); !got.Equal(sum) {
					return fmt.Sprintf("Add: %s != exact %s", got.ExactString(), sum.ExactString())
				}
			}
			if diffOK {
				r := a.DeepCopy()
				r.Sub(*b)
				if got := exactValue(&r); !got.Equal(diff) {
					return fmt.Sprintf("Sub: %s != exact %s", got.ExactString(), diff.ExactString())
				}
			}
			r := a.DeepCopy()
			r.Neg()
			if got := exactValue(&r); !got.Equal(ao.Neg()) {
				return fmt.Sprintf("Neg: %s != exact %s", got.ExactString(), ao.Neg().ExactString())
			}
			// Mul's bool reports whether the result kept the int64 backend, which
			// is not a property of the value, so only the value is checked.
			for _, y := range mulFactors {
				r := a.DeepCopy()
				r.Mul(y)
				if got, want := exactValue(&r), ao.Mul(y); !got.Equal(want) {
					return fmt.Sprintf("Mul(%d): %s != exact %s", y, got.ExactString(), want.ExactString())
				}
			}
			return ""
		})
	}
	// Add and Sub go unchecked on pairs whose alignment package exact cannot
	// build (an int32-edge value against a far scale; TestQuantityLawKnownGaps
	// covers that class). Their number is pinned so that it cannot grow unseen.
	if allSubtestsRan() && unjudged != arithUnjudgedPairs {
		t.Errorf("Add/Sub unchecked on %d of 2000 pairs, recorded %d: the value cases or package exact changed; update arithUnjudgedPairs once that is understood", unjudged, arithUnjudgedPairs)
	}
	checkVisited(t, hasPrefix("arith/"))
}

// accessorScales are the scales the accessor and rounding laws ask for: both
// sides of the int64 decimal range (19 digits). The int32 edges are left out on
// purpose: rounding an ordinary value to them is the inf.Dec alignment hang
// class, which in process cannot be stopped; package exact's own tests cover
// those scales.
var accessorScales = []Scale{0, -3, 3, -9, 9, 12, -18, 18, -19, 19}

// TestQuantityLawAccessors checks the checked accessors AsScaledInt64/AsMilliInt64
// and their wrappers Value/MilliValue/ScaledValue against the exact value, and
// AsInt64's ok==true direction (ok==false is allowed for a value that fits:
// AsInt64 only reports a fast conversion). AsApproximateFloat64 is checked
// against the nearest float64 of the exact value; AsFloat64Slow is checked the
// same way when its exponent is at most 1000 in magnitude, beyond which it is
// the inf.Dec alignment hang class (float64slow/1e2147483647).
func TestQuantityLawAccessors(t *testing.T) {
	cases := buildValueCases(t)
	slowChecked := 0
	for _, c := range cases {
		t.Run(c.id, func(t *testing.T) {
			q := c.make()
			eo := exactValue(q)
			slow := eo.Exponent() <= 1000 && eo.Exponent() >= -1000
			if slow {
				slowChecked++
			}
			runLaw(t, "accessor/"+c.id, "accessor", func() string {
				for _, s := range accessorScales {
					want, wantOK := eo.AsScaledInt64(int32(s))
					if got, ok := q.AsScaledInt64(s); got != want || ok != wantOK {
						return fmt.Sprintf("AsScaledInt64(%d) = (%d,%t), exact (%d,%t)", s, got, ok, want, wantOK)
					}
				}
				want, wantOK := eo.AsScaledInt64(int32(Milli))
				if got, ok := q.AsMilliInt64(); got != want || ok != wantOK {
					return fmt.Sprintf("AsMilliInt64() = (%d,%t), exact (%d,%t)", got, ok, want, wantOK)
				}
				if want, wantOK := eo.AsScaledInt64(0); q.Value() != want {
					return fmt.Sprintf("Value() = %d, exact %d (ok=%t)", q.Value(), want, wantOK)
				}
				if want, _ := eo.AsScaledInt64(int32(Milli)); q.MilliValue() != want {
					return fmt.Sprintf("MilliValue() = %d, exact %d", q.MilliValue(), want)
				}
				if want, _ := eo.AsScaledInt64(int32(Kilo)); q.ScaledValue(Kilo) != want {
					return fmt.Sprintf("ScaledValue(Kilo) = %d, exact %d", q.ScaledValue(Kilo), want)
				}
				if v, ok := q.AsInt64(); ok {
					if want, wantOK := eo.AsInt64(); !wantOK || v != want {
						return fmt.Sprintf("AsInt64() ok but = (%d,true), exact (%d,%t)", v, want, wantOK)
					}
				}
				if got := q.AsApproximateFloat64(); !floatNear(got, eo.AsApproximateFloat64()) {
					return fmt.Sprintf("AsApproximateFloat64() = %v, exact %v", got, eo.AsApproximateFloat64())
				}
				if slow {
					if got := q.AsFloat64Slow(); !floatNear(got, eo.AsApproximateFloat64()) {
						return fmt.Sprintf("AsFloat64Slow() = %v, exact %v", got, eo.AsApproximateFloat64())
					}
				}
				return ""
			})
		})
	}
	if allSubtestsRan() && slowChecked < len(cases)*9/10 {
		t.Errorf("AsFloat64Slow checked on %d of %d value cases; the exponent bound skips too many", slowChecked, len(cases))
	}
	checkVisited(t, hasPrefix("accessor/"))
}

// TestQuantityLawString checks String() against the canonical form computed
// from the value, and the round trip of each encoding on its own: String()
// reparses to the same value and is a fixed point, and JSON, CBOR and proto
// decode back to the same value. Each has its own id, so that a listed
// deviation in one does not hide another.
func TestQuantityLawString(t *testing.T) {
	cases := buildValueCases(t)
	for _, c := range cases {
		t.Run(c.id, func(t *testing.T) {
			q := c.make()
			eo := exactValue(q)
			back := func(codec string, r Quantity) string {
				if got := exactValue(&r); !got.Equal(eo) {
					return fmt.Sprintf("%s round trip value %s != original %s", codec, got.ExactString(), eo.ExactString())
				}
				return ""
			}
			runLaw(t, "canonical/"+c.id, "canonical", func() string {
				if s, want := q.String(), canonicalString(q); s != want {
					return fmt.Sprintf("String() = %q, canonical form %q", s, want)
				}
				return ""
			})
			runLaw(t, "roundtrip/string/"+c.id, "roundtrip", func() string {
				s := q.String()
				r, err := ParseQuantity(s)
				if err != nil {
					return fmt.Sprintf("String() = %q does not reparse: %v", s, err)
				}
				if d := back("String "+strconv.Quote(s), r); d != "" {
					return d
				}
				if r.String() != s {
					return fmt.Sprintf("String() not a fixed point: %q -> %q", s, r.String())
				}
				return ""
			})
			runLaw(t, "roundtrip/json/"+c.id, "roundtrip", func() string {
				j, err := json.Marshal(*q)
				if err != nil {
					return fmt.Sprintf("json.Marshal: %v", err)
				}
				var r Quantity
				if err := json.Unmarshal(j, &r); err != nil {
					return fmt.Sprintf("json.Unmarshal(%s): %v", j, err)
				}
				return back("json "+string(j), r)
			})
			runLaw(t, "roundtrip/cbor/"+c.id, "roundtrip", func() string {
				b, err := q.MarshalCBOR()
				if err != nil {
					return fmt.Sprintf("MarshalCBOR: %v", err)
				}
				var r Quantity
				if err := r.UnmarshalCBOR(b); err != nil {
					return fmt.Sprintf("UnmarshalCBOR: %v", err)
				}
				return back("cbor", r)
			})
			runLaw(t, "roundtrip/proto/"+c.id, "roundtrip", func() string {
				b, err := q.Marshal()
				if err != nil {
					return fmt.Sprintf("proto Marshal: %v", err)
				}
				var r Quantity
				if err := r.Unmarshal(b); err != nil {
					return fmt.Sprintf("proto Unmarshal: %v", err)
				}
				return back("proto", r)
			})
		})
	}
	checkVisited(t, hasPrefix("canonical/", "roundtrip/"))
}

// canonicalValue reads the value an AsScale result denotes.
func canonicalValue(v CanonicalValue) exact.Quantity {
	digits, exp := v.AsCanonicalBytes(nil)
	m, ok := new(big.Int).SetString(string(digits), 10)
	if !ok {
		panic(fmt.Sprintf("AsCanonicalBytes returned %q", digits))
	}
	return exact.New(m, int64(exp), exact.DecimalSI)
}

// TestQuantityLawMisc checks AsScale, RoundUp, DeepCopy, ToDec and the
// constructors against the exact value. Each check has its own id.
func TestQuantityLawMisc(t *testing.T) {
	cases := buildValueCases(t)
	for _, c := range cases {
		t.Run(c.id, func(t *testing.T) {
			q := c.make()
			eo := exactValue(q)
			runLaw(t, "misc/asscale/"+c.id, "misc", func() string {
				for _, s := range accessorScales {
					want, wantExact := eo.RoundToScale(int32(s))
					v, wasExact := q.AsScale(s)
					if got := canonicalValue(v); !got.Equal(want) || wasExact != wantExact {
						return fmt.Sprintf("AsScale(%d) = %s, %t; exact %s, %t", s, got.ExactString(), wasExact, want.ExactString(), wantExact)
					}
				}
				return ""
			})
			runLaw(t, "misc/roundup/"+c.id, "misc", func() string {
				for _, s := range accessorScales {
					want, wantExact := eo.RoundToScale(int32(s))
					r := q.DeepCopy()
					wasExact := r.RoundUp(s)
					if got := exactValue(&r); !got.Equal(want) || wasExact != wantExact {
						return fmt.Sprintf("RoundUp(%d) = %s, %t; exact %s, %t", s, got.ExactString(), wasExact, want.ExactString(), wantExact)
					}
				}
				return ""
			})
			runLaw(t, "misc/deepcopy/"+c.id, "misc", func() string {
				cp := q.DeepCopy()
				if got := exactValue(&cp); !got.Equal(eo) {
					return fmt.Sprintf("DeepCopy value %s != %s", got.ExactString(), eo.ExactString())
				}
				cp.Neg()
				if got := exactValue(q); !got.Equal(eo) {
					return "DeepCopy shares state with the original"
				}
				return ""
			})
			runLaw(t, "misc/todec/"+c.id, "misc", func() string {
				td := q.DeepCopy()
				td.ToDec()
				if got := exactValue(&td); !got.Equal(eo) {
					return fmt.Sprintf("ToDec value %s != %s", got.ExactString(), eo.ExactString())
				}
				return ""
			})
		})
	}

	// Constructors and setters are exact by definition; check a sample.
	t.Run("constructors", func(t *testing.T) {
		checks := []struct {
			id   string
			want exact.Quantity
			make func() *Quantity
		}{
			{"NewQuantity/5", exact.NewInt64(5, 0), func() *Quantity { return NewQuantity(5, DecimalSI) }},
			{"NewMilliQuantity/5", exact.NewInt64(5, -3), func() *Quantity { return NewMilliQuantity(5, DecimalSI) }},
			{"NewScaledQuantity/5/-9", exact.NewInt64(5, -9), func() *Quantity { return NewScaledQuantity(5, -9) }},
			{"Set/5", exact.NewInt64(5, 0), func() *Quantity { q := &Quantity{}; q.Set(5); return q }},
			{"SetMilli/5", exact.NewInt64(5, -3), func() *Quantity { q := &Quantity{}; q.SetMilli(5); return q }},
			{"SetScaled/5/-9", exact.NewInt64(5, -9), func() *Quantity { q := &Quantity{}; q.SetScaled(5, -9); return q }},
		}
		for _, c := range checks {
			t.Run(c.id, func(t *testing.T) {
				q := c.make()
				if got := exactValue(q); !got.Equal(c.want) {
					t.Errorf("%s: value %s != %s", c.id, got.ExactString(), c.want.ExactString())
				}
			})
		}
	})
	checkVisited(t, hasPrefix("misc/"))
}

// TestQuantityLawKnownGaps exercises the currently-wrong behaviours that package
// exact cannot express as a cheap exact sum (extreme scale alignments): the known
// hangs and the int64Amount.Add/Sub int32 wrap. The hang cases run in child
// processes (runHangCase) so a killed child cannot leak goroutines or CPU, and
// they run one at a time so at most one stuck child exists at any moment. Each
// case is in knownDeviations, so a fix that changes the behaviour makes the entry fail
// and forces its removal.
func TestQuantityLawKnownGaps(t *testing.T) {
	// The canonical hang class: inf.Dec aligns scales by writing 10^2147483647
	// digits, reachable through Add, AsFloat64Slow, and RoundUp.
	for _, id := range []string{
		"add/1e2147483647+1",
		"float64slow/1e2147483647",
		"roundup/1e2147483647-dec",
	} {
		t.Run(id, func(t *testing.T) { runHangLaw(t, id, "wrong-value") })
	}

	// int64Amount.Add/Sub compute the scale delta in int32 and wrap instead of
	// falling back: 1e2147483647 + 1n is about 1e2147483647, not 2e-9. These
	// cases return immediately, so they stay in-process. The comparisons are
	// on the representations, since Cmp would have to align the same scales.
	wrap := func(id string, a, b *Quantity, sub bool, want exact.Quantity) {
		t.Run(id, func(t *testing.T) {
			runLaw(t, id, "wrong-value", func() string {
				op := "Add"
				if sub {
					op = "Sub"
					a.Sub(*b)
				} else {
					a.Add(*b)
				}
				if got := exactValue(a); !got.Equal(want) {
					return fmt.Sprintf("%s = %s, want %s (int32 scale delta wraps)", op, got.ExactString(), want.ExactString())
				}
				return ""
			})
		})
	}
	huge := func() *Quantity { q := MustParse("1e2147483647"); return &q }
	tiny := func() *Quantity { return NewScaledQuantity(7, math.MinInt32) }
	nano := MustParse("1n")
	five := NewQuantity(5, DecimalSI)
	// The exact sums are within 10^-2147483656 of these, below what the int32
	// scale can tell apart, so the library is held to them.
	wrap("add/1e2147483647+1n", huge(), &nano, false, exact.NewInt64(1, math.MaxInt32))
	wrap("sub/1e2147483647-1n", huge(), &nano, true, exact.NewInt64(1, math.MaxInt32))
	wrap("add/7e-2147483648+5", tiny(), five, false, exact.NewInt64(5, 0))
	wrap("sub/7e-2147483648-5", tiny(), five, true, exact.NewInt64(-5, 0))

	// The MinInt32 scale seen from Cmp: against a scale-0 value the widened
	// scale difference sends Cmp through AsDec, where the scale does not fit,
	// so 7e-2147483648 compares greater than 1. Both orders, since Cmp must be
	// antisymmetric.
	for _, tc := range []struct {
		id   string
		a, b *Quantity
		want int
	}{
		{"cmp/7e-2147483648|1", tiny(), NewQuantity(1, DecimalSI), -1},
		{"cmp/1|7e-2147483648", NewQuantity(1, DecimalSI), tiny(), 1},
	} {
		t.Run(tc.id, func(t *testing.T) {
			runLaw(t, tc.id, "wrong-value", func() string {
				if got := tc.a.Cmp(*tc.b); got != tc.want {
					return fmt.Sprintf("Cmp = %d, want %d", got, tc.want)
				}
				return ""
			})
		})
	}
	// Everything the law tests do not own belongs here, knownGapCmpIDs included
	// (hasPrefix leaves them out of cmp/).
	checkVisited(t, func(id string) bool { return !hasPrefix(lawPrefixes...)(id) })
}

// TestQuantityLawGrammar keeps the regular expression Quantity publishes as its
// grammar (splitREString, quoted by ErrFormatWrong) and the one exact.Parse
// splits its input with as a single definition. ParseQuantity itself runs a
// hand-written scanner, not this expression; the parse law is what checks the
// scanner against exact.Parse, input by input.
func TestQuantityLawGrammar(t *testing.T) {
	if splitREString != exact.Grammar {
		t.Errorf("splitREString = %q, exact.Grammar = %q; keep them identical", splitREString, exact.Grammar)
	}
}

// TestQuantityLawLists checks the shape of every list entry: a reason, a kind
// the laws use, and the observed deviation a non-hang entry is held to.
func TestQuantityLawLists(t *testing.T) {
	kinds := []string{"hang", "panic", "wrong-value", "sign", "compare", "arith", "accessor", "canonical", "roundtrip", "misc", "parse"}
	for id, e := range knownDeviations {
		switch {
		case e.ref == "":
			t.Errorf("knownDeviations %q has no ref", id)
		case !slices.Contains(kinds, e.kind):
			t.Errorf("knownDeviations %q has kind %q, not one of %v", id, e.kind, kinds)
		case (e.kind == "hang") != (e.observed == ""):
			t.Errorf("knownDeviations %q: a hang records no observed detail, every other kind does", id)
		}
	}
	for input, e := range parseRegressions {
		if e.ref == "" || e.observed == "" {
			t.Errorf("parseRegressions %q needs a ref and the observed decode", input)
		}
	}
	for list, m := range map[string]map[string]string{"parseApprovedChanges": parseApprovedChanges, "parseFrozenExamples": parseFrozenExamples} {
		for input, reason := range m {
			if reason == "" {
				t.Errorf("%s %q has no reason", list, input)
			}
		}
	}
}

// TestQuantityLawSuffixes keeps exact.Parse's suffix table in step with the
// library's: every suffix of up to two characters the grammar allows is
// accepted by both or by neither (decimal exponents are a separate rule in
// both). A suffix added to one table alone fails here rather than as a wrong
// decode on every input that uses it.
func TestQuantityLawSuffixes(t *testing.T) {
	const chars = "eEinumkKMGTP"
	var suffixes []string
	for _, a := range chars {
		suffixes = append(suffixes, string(a))
		for _, b := range chars {
			suffixes = append(suffixes, string(a)+string(b))
		}
	}
	for _, sfx := range suffixes {
		_, _, _, lib := quantitySuffixer.interpret(suffix(sfx))
		_, err := exact.Parse("1" + sfx)
		if lib != (err == nil) {
			t.Errorf("suffix %q: library accepts %t, exact.Parse error %v", sfx, lib, err)
		}
	}
}

// TestQuantityLawParseRatSeam checks ParseQuantity against exact.Parse around
// ±exact.RatLimit, where exact.Quantity changes from a big.Rat to a mantissa
// and exponent and fraction digits move the exponent across the boundary.
// The parse grid has no exponents there. The mantissas are short: a long one
// goes through inf.Dec, whose nano round at these exponents takes minutes.
func TestQuantityLawParseRatSeam(t *testing.T) {
	total := 0
	for _, sign := range []string{"", "-"} {
		for _, mantissa := range []string{"1", "1.5", "0.1", "12.34", "1000"} {
			for _, e := range []int64{exact.RatLimit - 1, exact.RatLimit, exact.RatLimit + 1, exact.RatLimit + 3} {
				for _, exp := range []int64{e, -e} {
					input := sign + mantissa + "e" + strconv.FormatInt(exp, 10)
					total++
					if parseHangs(input) {
						t.Errorf("%q: parseHangs reports a hang; the seam law needs inputs that decode", input)
						continue
					}
					runLaw(t, "seam/"+input, "parse", func() string {
						// Correct as in the parse law: the exact decode, stored in a
						// spelling that decodes back to it.
						v, err := classifyParse(input, decodeRecord{})
						switch {
						case err != nil:
							return err.Error()
						case !v.judged:
							panic(exact.Unjudgeable(fmt.Sprintf("exact.Parse cannot judge %q", input)))
						case v.cur != v.want || v.roundTrip != "":
							return v.String()
						}
						return ""
					})
				}
			}
		}
	}
	t.Logf("%d seam inputs", total)
	checkVisited(t, hasPrefix("seam/"))
}
