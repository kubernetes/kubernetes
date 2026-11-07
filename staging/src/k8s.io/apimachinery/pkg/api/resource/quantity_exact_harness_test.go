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

// This file holds the harness the Quantity laws run under: the conversion of a
// Quantity to its exact value (package internal/exact), the per-case timeout,
// the child process that runs a case which may hang, and checkLaw, which
// records each result against knownDeviations in quantity_exact_policy_test.go.
//
// A Quantity denotes an exact rational number:
//
//   - int64Amount{value, scale} denotes value * 10^scale.
//   - infDecAmount (an inf.Dec) denotes unscaled * 10^-Scale().
//
// The laws compare the library with that value, computed by package exact
// from math/big only: never with the library's own arithmetic, and never
// through float64 or int64(float) conversions.

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"io"
	"math"
	"math/big"
	"os"
	"os/exec"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/resource/internal/exact"
)

// caseTimeout bounds each law case so that a hang, such as the known inf.Dec
// scale-alignment hang for 1e2147483647 + 1, fails that case with its
// description instead of stalling the suite.
const caseTimeout = 2 * time.Second

// hangTimeout bounds a hang case run in a child process (runHangCase). Such a
// case either never returns or, once fixed, returns in microseconds, so one
// second separates the two with a wide margin while bounding the memory the
// stuck child touches before it is killed.
const hangTimeout = time.Second

// helperStartTimeout bounds how long the child process may take to start and
// report that it is ready, before hangTimeout starts.
const helperStartTimeout = time.Minute

// lawHelperEnv and lawHelperCase name the environment variables through
// which the parent (runHangCase) tells the re-execed helper process which case
// to run (the standard TestHelperProcess pattern from the Go standard library).
const (
	lawHelperEnv   = "QUANTITY_LAW_HELPER_PROCESS"
	lawHelperCase  = "QUANTITY_LAW_HELPER_CASE"
	lawHelperReady = "QUANTITY_LAW_HELPER_READY"
)

// exactValue returns the exact value q denotes, read from its representation.
func exactValue(q *Quantity) exact.Quantity {
	if q.d.Dec != nil {
		return exact.New(q.d.Dec.UnscaledBig(), -int64(q.d.Dec.Scale()), string(q.Format))
	}
	return exact.New(big.NewInt(q.i.value), int64(q.i.scale), string(q.Format))
}

// canonicalString returns q's canonical serialization, computed by package
// exact from q's value rather than by CanonicalizeBytes.
func canonicalString(q *Quantity) string {
	return exactValue(q).Canonical(string(q.Format))
}

// floatNear reports whether got matches want within a small relative error, or
// the matching infinity or signed zero when want is out of the finite range.
func floatNear(got, want float64) bool {
	if math.IsInf(want, 0) {
		return math.IsInf(got, int(math.Copysign(1, want)))
	}
	if want == 0 {
		return got == 0 && math.Signbit(got) == math.Signbit(want)
	}
	return math.Abs(got-want)/math.Abs(want) <= 1e-12
}

// runLawCase runs fn (which returns a non-empty string describing a deviation,
// or "" if the library matched the exact value) under the per-case timeout. A
// goroutine that exceeds the timeout is abandoned: Go cannot kill a goroutine,
// so a genuinely stuck operation (the inf.Dec scale-alignment hang in #141166)
// keeps running and may keep growing the process until the test binary exits.
// The hang cases are therefore few and run sequentially, never in parallel.
//
// A panic is reported in panicked, unless package exact raised it with
// exact.Unjudgeable: that is the reference giving up, not the library
// failing, and is reported in unjudged instead.
func runLawCase(fn func() string) (detail string, timedOut bool, panicked, unjudged string) {
	type result struct{ detail, panicked, unjudged string }
	done := make(chan result, 1)
	go func() {
		var r result
		func() {
			defer func() {
				switch p := recover().(type) {
				case nil:
				case exact.Unjudgeable:
					r.unjudged = p.Error()
				default:
					r.panicked = fmt.Sprintf("panic: %v", p)
				}
			}()
			r.detail = fn()
		}()
		done <- r
	}()
	select {
	case r := <-done:
		if r.panicked != "" {
			return r.panicked, false, r.panicked, ""
		}
		return r.detail, false, "", r.unjudged
	case <-time.After(caseTimeout):
		return "", true, "", ""
	}
}

// runHangCaseBody runs one known hang case inside the helper child process
// (see TestQuantityLawHelperProcess). It returns "" when the library now
// matches the exact value and a description when it deviates, and panics with
// exact.Unjudgeable when package exact cannot judge the result. Each body
// checks the result against the exact value, as the law the case belongs to
// does; where package exact cannot build that value (Add at the int32 edge),
// the case fails as unjudged until it gets a check of its own. While the bug
// is present the case never returns, and the parent kills the child
// hangTimeout after the child reports that it is ready.
func runHangCaseBody(id string) string {
	if input, ok := strings.CutPrefix(id, "parse/"); ok {
		// The release hung on every listed input, so it recorded no decode.
		v, err := classifyParse(input, decodeRecord{Input: input, Outcome: "hang"})
		switch {
		case err != nil:
			return err.Error()
		case !v.judged:
			panic(exact.Unjudgeable(fmt.Sprintf("exact.Parse cannot judge %q (%s)", input, v)))
		case v.quadrant != "correct/no-1.37-decode":
			return v.String()
		}
		return ""
	}
	huge := MustParse("1e2147483647")
	eo := exactValue(&huge)
	switch id {
	case "add/1e2147483647+1":
		one := MustParse("1")
		huge.Add(one)
		want, ok := eo.Add(exactValue(&one))
		if !ok {
			panic(exact.Unjudgeable("1e2147483647 + 1 needs more digits than package exact materializes"))
		}
		if got := exactValue(&huge); !got.Equal(want) {
			return fmt.Sprintf("Add: %s != exact %s", got.ExactString(), want.ExactString())
		}
		return ""
	case "float64slow/1e2147483647":
		if got, want := huge.AsFloat64Slow(), eo.AsApproximateFloat64(); !floatNear(got, want) {
			return fmt.Sprintf("AsFloat64Slow() = %v, exact %v", got, want)
		}
		return ""
	case "roundup/1e2147483647-dec":
		huge.ToDec()
		want, wantExact := eo.RoundToScale(0)
		wasExact := huge.RoundUp(0)
		if got := exactValue(&huge); !got.Equal(want) || wasExact != wantExact {
			return fmt.Sprintf("RoundUp(0) = %s, %t; exact %s, %t", got.ExactString(), wasExact, want.ExactString(), wantExact)
		}
		return ""
	default:
		return fmt.Sprintf("unknown law helper case %q", id)
	}
}

// Child results. The helper prints one of these and its quoted detail on the
// line after lawHelperReady, so the parent can record a returned case under
// the same kinds runLaw uses.
const (
	lawHelperMatched  = "matched"
	lawHelperDeviated = "deviated"
	lawHelperPanicked = "panicked"
	lawHelperUnjudged = "unjudged"
)

// TestQuantityLawHelperProcess is the re-exec target for the known hang cases.
// runHangCase re-runs this test binary with -test.run selecting only this test
// and the case id in the environment. The child runs the case under
// runLawCase, prints its result, and exits 0; while the bug is present it
// never gets that far and is killed by the parent.
func TestQuantityLawHelperProcess(t *testing.T) {
	if os.Getenv(lawHelperEnv) != "1" {
		t.Skip("re-exec target of runHangCase, not a test on its own")
	}
	// Starting the binary is not part of the case: the parent's timer starts
	// at this marker, so a slow start cannot read as a hang.
	fmt.Println(lawHelperReady)
	id := os.Getenv(lawHelperCase)
	detail, timedOut, panicked, unjudged := runLawCase(func() string { return runHangCaseBody(id) })
	switch {
	case timedOut:
		// The parent's hangTimeout is shorter than caseTimeout, so it has
		// already killed this process; nothing to report.
		select {}
	case panicked != "":
		fmt.Println(lawHelperPanicked, strconv.Quote(panicked))
	case unjudged != "":
		fmt.Println(lawHelperUnjudged, strconv.Quote(unjudged))
	case detail != "":
		fmt.Println(lawHelperDeviated, strconv.Quote(detail))
	default:
		fmt.Println(lawHelperMatched, strconv.Quote(""))
	}
	os.Exit(0)
}

// runHangCase runs one known hang case in a child process and waits up to
// hangTimeout for it. It returns what runLawCase returns for an in-process
// case: a child killed on timeout is timedOut, and a child that returned (the
// hang is fixed) reports its result, so a wrong value, a panic or an
// unjudgeable result is recorded as such rather than as a pass.
func runHangCase(t *testing.T, id string) (detail string, timedOut bool, panicked, unjudged string) {
	t.Helper()
	// The child gets its own deadline, so it stops even if this test binary
	// times out first and never kills it.
	cmd := exec.Command(os.Args[0], "-test.run=^TestQuantityLawHelperProcess$", "-test.count=1",
		"-test.timeout="+(helperStartTimeout+caseTimeout).String())
	// A race-enabled binary sleeps for a second before a clean exit
	// (atexit_sleep_ms), which would make a fixed case look like a hang.
	cmd.Env = append(os.Environ(), lawHelperEnv+"=1", lawHelperCase+"="+id,
		"GORACE="+strings.TrimSpace(os.Getenv("GORACE")+" atexit_sleep_ms=0"))
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("law helper for %s: %v", id, err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("start law helper for %s: %v", id, err)
	}
	ready := make(chan struct{})
	type exit struct {
		out string
		err error
	}
	done := make(chan exit, 1)
	go func() {
		r := bufio.NewReader(stdout)
		line, _ := r.ReadString('\n')
		if strings.TrimSpace(line) == lawHelperReady {
			close(ready)
		}
		rest, _ := io.ReadAll(r)
		// Wait closes the pipe, so it runs only after every read is done.
		done <- exit{string(rest), cmd.Wait()}
	}()
	select {
	case <-ready:
	case e := <-done:
		t.Fatalf("law helper for %s exited before it was ready: %v: %s", id, e.err, stderr.String())
	case <-time.After(helperStartTimeout):
		_ = cmd.Process.Kill()
		<-done
		t.Fatalf("law helper for %s was not ready within %s: %s", id, helperStartTimeout, stderr.String())
	}
	select {
	case e := <-done:
		if e.err != nil {
			t.Fatalf("law helper for %s failed: %v: %s%s", id, e.err, e.out, stderr.String())
		}
		status, quoted, _ := strings.Cut(strings.TrimSpace(e.out), " ")
		text, err := strconv.Unquote(quoted)
		if err != nil {
			t.Fatalf("law helper for %s printed %q: %v", id, e.out, err)
		}
		switch status {
		case lawHelperMatched:
			return "", false, "", ""
		case lawHelperDeviated:
			return text, false, "", ""
		case lawHelperPanicked:
			return text, false, text, ""
		case lawHelperUnjudged:
			return "", false, "", text
		}
		t.Fatalf("law helper for %s printed %q", id, e.out)
	case <-time.After(hangTimeout):
		_ = cmd.Process.Kill()
		<-done // reap the killed child so it cannot leak
	}
	return "", true, "", ""
}

// checkLaw records a per-case result against knownDeviations.
func checkLaw(t *testing.T, id, kind string, deviated bool, detail string) {
	t.Helper()
	visitedMu.Lock()
	visited[id] = true
	visitedMu.Unlock()
	entry, ok := knownDeviations[id]
	switch {
	case deviated && !ok:
		t.Errorf("%s [%s]: %s (not a known deviation)", id, kind, detail)
	case deviated && ok:
		// A listed deviation of one kind covers no other: a hang, a panic or a
		// wrong value in place of the listed one is a new failure.
		if entry.kind != kind {
			t.Errorf("%s [%s]: %s (listed as %s, not as %s)", id, kind, detail, entry.kind, kind)
			return
		}
		// Any other wrong result is a new failure too, so the entry records the
		// detail it was listed with.
		if entry.kind != "hang" && detail != entry.observed {
			t.Errorf("%s [%s]: %s (listed with %q)", id, kind, detail, entry.observed)
			return
		}
		t.Logf("%s: expected deviation (%s: %s), still present", id, entry.kind, entry.ref)
	case !deviated && ok:
		t.Errorf("%s: known deviation (%s) is no longer present; remove its knownDeviations entry", id, entry.ref)
	case !deviated && !ok:
		// matches the exact value and is not listed: correct.
	}
}

// runLaw runs one in-process case under the per-case timeout and records it
// against knownDeviations under kind; a timeout or a panic is recorded under
// its own kind. A case package exact cannot judge fails on its own: it is
// neither a match nor a deviation.
func runLaw(t *testing.T, id, kind string, fn func() string) {
	t.Helper()
	detail, timedOut, panicked, unjudged := runLawCase(fn)
	recordLaw(t, id, kind, detail, timedOut, panicked, unjudged)
}

// recordLaw records one result of runLawCase or runHangCase.
func recordLaw(t *testing.T, id, kind, detail string, timedOut bool, panicked, unjudged string) {
	t.Helper()
	// An unjudged case is visited too, so it does not also read as a stale entry.
	visitedMu.Lock()
	visited[id] = true
	visitedMu.Unlock()
	switch {
	case timedOut:
		checkLaw(t, id, "hang", true, "timed out")
	case panicked != "":
		checkLaw(t, id, "panic", true, panicked)
	case unjudged != "":
		t.Errorf("%s [%s]: %s", id, kind, unjudged)
	default:
		checkLaw(t, id, kind, detail != "", detail)
	}
}

// runHangLaw runs one known hang case in a child process and records it like
// runLaw: a case that returns is held to its law under kind.
func runHangLaw(t *testing.T, id, kind string) {
	t.Helper()
	detail, timedOut, panicked, unjudged := runHangCase(t, id)
	recordLaw(t, id, kind, detail, timedOut, panicked, unjudged)
}

// visited holds every case id recordLaw or checkLaw has seen.
var (
	visitedMu sync.Mutex
	visited   = map[string]bool{}
)

// knownGapCmpIDs are the cmp/ ids that TestQuantityLawKnownGaps runs rather
// than the Compare law.
var knownGapCmpIDs = []string{"cmp/7e-2147483648|1", "cmp/1|7e-2147483648"}

// lawPrefixes are the id prefixes owned by the law tests; every other
// knownDeviations id belongs to TestQuantityLawKnownGaps.
var lawPrefixes = []string{"parse/", "sign/", "cmp/", "arith/", "accessor/", "canonical/", "roundtrip/", "misc/", "seam/"}

// checkVisited fails every knownDeviations entry that owns claims and no case
// visited, so an entry for a removed or misspelled case does not linger. Like
// checkParseLists it assumes the whole test ran.
func checkVisited(t *testing.T, owns func(id string) bool) {
	t.Helper()
	if !allSubtestsRan() {
		return
	}
	visitedMu.Lock()
	defer visitedMu.Unlock()
	for id := range knownDeviations {
		if owns(id) && !visited[id] {
			t.Errorf("knownDeviations entry %q is not visited by any case", id)
		}
	}
}

// hasPrefix returns an owns function for checkVisited.
func hasPrefix(prefixes ...string) func(string) bool {
	return func(id string) bool {
		for _, p := range prefixes {
			if strings.HasPrefix(id, p) && !slices.Contains(knownGapCmpIDs, id) {
				return true
			}
		}
		return false
	}
}

// allSubtestsRan reports whether no -run subtest pattern or -skip pattern can
// have left out cases, so that the list checks see every case.
func allSubtestsRan() bool {
	if f := flag.Lookup("test.run"); f != nil && strings.Contains(f.Value.String(), "/") {
		return false
	}
	f := flag.Lookup("test.skip")
	return f == nil || f.Value.String() == ""
}
