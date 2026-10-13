//go:build poolsdebug

// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package pools

import (
	"fmt"
	"runtime"
	"sync"
)

// This is the instrumented implementation of the pool tracking, enabled with
// -tags poolsdebug.
//
// Each pool carries a tracker that records, per recycled pointer, whether it is
// currently borrowed or redeemed, together with the call sites of the last
// borrow and redeem.
// It panics loudly (with those call sites) when it detects misuse:
//
//   - a double redeem (the same object returned to the pool twice — corrupts sync.Pool);
//   - for the redeemable pools, a redeem of a stale borrow (the slot was re-borrowed since — the
//     ABA case the production atomic guard cannot catch), thanks to a per-borrow generation;
//   - a redeem of an object the pool never handed out;
//   - a borrow of an object still checked out (a symptom of an earlier double-Put).
//
// Borrowed-but-never-redeemed objects (leaks) are reported by [AssertNoLeaks].

// debugBuild reports whether the pool instrumentation is compiled in (the
// poolsdebug tag).
const debugBuild = true

// DebugBuild reports whether the pool instrumentation is compiled in (the
// poolsdebug build tag).
//
// See the release-build doc for usage.
const DebugBuild = debugBuild

type trackStatus uint8

const (
	trackBorrowed trackStatus = iota + 1
	trackRedeemed
)

type trackEntry struct {
	status     trackStatus
	gen        uint64 // identifies the current borrow, to detect a redeem racing a re-borrow (ABA)
	borrowedAt string
	redeemedAt string
}

type tracker[T any] struct {
	mu      sync.Mutex
	entries map[*T]*trackEntry
	nextGen uint64
}

func (t *tracker[T]) register() {
	t.mu.Lock()
	if t.entries == nil {
		t.entries = make(map[*T]*trackEntry)
	}
	t.mu.Unlock()

	registerLeakChecker(t)
}

// markBorrow records a borrow of ptr and returns its generation.
//
// Caller must hold no lock.
func (t *tracker[T]) markBorrow(ptr *T, site string) uint64 {
	t.mu.Lock()
	defer t.mu.Unlock()

	e := t.entries[ptr]
	if e == nil {
		e = &trackEntry{}
		t.entries[ptr] = e
	} else if e.status == trackBorrowed {
		panic(fmt.Sprintf(
			"pools: borrow of an object still checked out (borrowed at %s); "+
				"this usually means it was redeemed twice earlier", e.borrowedAt))
	}

	t.nextGen++
	e.status = trackBorrowed
	e.gen = t.nextGen
	e.borrowedAt = site

	return t.nextGen
}

// markRedeem validates and records a redeem of ptr. gen is the borrow
// generation the caller is redeeming, or 0 to skip the ABA check (plain
// Pool[T], which has no per-borrow token).
func (t *tracker[T]) markRedeem(ptr *T, gen uint64, site string) {
	t.mu.Lock()
	defer t.mu.Unlock()

	e := t.entries[ptr]
	switch {
	case e == nil:
		panic("pools: redeem of an object this pool never handed out")
	case e.status != trackBorrowed:
		panic(fmt.Sprintf("pools: double redeem (first redeemed at %s)", e.redeemedAt))
	case gen != 0 && e.gen != gen:
		panic(fmt.Sprintf(
			"pools: redeem of a stale borrow (the slot was re-borrowed at %s since this borrow); "+
				"a redeem is racing a re-borrow of the same slot (ABA)", e.borrowedAt))
	}

	e.status = trackRedeemed
	e.redeemedAt = site
}

const stackOffset = 3

func (t *tracker[T]) onBorrow(ptr *T) {
	t.markBorrow(ptr, caller(stackOffset))
}

func (t *tracker[T]) onRedeem(ptr *T) {
	t.markRedeem(ptr, 0, caller(stackOffset))
}

// borrowRedeemer records the borrow and returns a generation-stamped redeemer
// that validates the redeem (catching double-redeem and ABA) before delegating
// to the cached redeemer.
func (t *tracker[T]) borrowRedeemer(ptr *T, cached func()) func() {
	gen := t.markBorrow(ptr, caller(stackOffset))

	return func() {
		t.markRedeem(ptr, gen, caller(stackOffset-1))
		cached()
	}
}

func (t *tracker[T]) checkLeaks(tb TB) bool {
	t.mu.Lock()
	defer t.mu.Unlock()

	ok := true
	for _, e := range t.entries {
		if e.status != trackRedeemed {
			tb.Logf("pools: object borrowed but never redeemed (borrowed at %s)", e.borrowedAt)
			ok = false
		}
	}

	return ok
}

func (t *tracker[T]) resetTracking() {
	t.mu.Lock()
	t.entries = make(map[*T]*trackEntry)
	t.nextGen = 0
	t.mu.Unlock()
}

// leakChecker is the build-erased view of a tracker that the global registry
// holds, so trackers of different element types can be checked uniformly.
type leakChecker interface {
	checkLeaks(tb TB) bool
	resetTracking()
}

var (
	registryMu sync.Mutex
	registry   []leakChecker
)

func registerLeakChecker(c leakChecker) {
	registryMu.Lock()
	registry = append(registry, c)
	registryMu.Unlock()
}

// AssertNoLeaks reports whether every borrowed object has been redeemed across
// all pools created so far.
//
// It logs the borrow call site of each leaked object and fails tb when any are
// found.
//
// Typical use, with [ResetTracking] to isolate the test from earlier ones:
//
//	func TestX(t *testing.T) {
//		pools.ResetTracking()
//		t.Cleanup(func() { pools.AssertNoLeaks(t) })
//		// ... exercise code that borrows/redeems ...
//	}
func AssertNoLeaks(tb TB) bool {
	tb.Helper()
	registryMu.Lock()
	defer registryMu.Unlock()

	ok := true
	for _, c := range registry {
		if !c.checkLeaks(tb) {
			ok = false
		}
	}
	if !ok {
		tb.Errorf("pools: leaked pooled objects detected (borrowed but never redeemed)")
	}

	return ok
}

// ResetTracking clears all recorded borrow/redeem tracking across every pool.
//
// Call it at the start of a test so leaks from earlier tests are not attributed
// to it.
func ResetTracking() {
	registryMu.Lock()
	defer registryMu.Unlock()

	for _, c := range registry {
		c.resetTracking()
	}
}

// caller returns "file:line" of the frame skip levels above caller itself.
func caller(skip int) string {
	pc, _, _, ok := runtime.Caller(skip)
	if !ok {
		return "unknown"
	}
	fn := runtime.FuncForPC(pc)
	if fn == nil {
		return "unknown"
	}
	file, line := fn.FileLine(pc)

	return fmt.Sprintf("%s:%d", file, line)
}
