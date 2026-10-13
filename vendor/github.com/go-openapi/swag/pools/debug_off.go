//go:build !poolsdebug

// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package pools

// This is the release implementation of the pool instrumentation: it does
// nothing.
//
// tracker is an empty struct, so it adds no field to the pool types and its
// methods inline away to nothing.
//
// Build with -tags poolsdebug to get the instrumented variant (see
// debug_on.go).

// debugBuild reports whether the pool instrumentation is compiled in (the
// poolsdebug tag).
const debugBuild = false

// DebugBuild reports whether the pool instrumentation is compiled in (the
// poolsdebug build tag).
//
// It lets a test that must run in both modes skip the parts that are invalid
// under instrumentation — e.g. an allocation-count assertion, since the
// instrumented build allocates a per-borrow tracker.
const DebugBuild = debugBuild

type tracker[T any] struct{}

func (tracker[T]) register() {}

func (tracker[T]) onBorrow(*T) {}

func (tracker[T]) onRedeem(*T) {}

func (tracker[T]) borrowRedeemer(_ *T, cached func()) func() { return cached }

// AssertNoLeaks reports whether every borrowed object has been redeemed across
// all pools.
//
// It is only meaningful in the instrumented build (-tags poolsdebug).
//
// In a release build it is a no-op that always reports true, so the same test
// can run in both modes.
func AssertNoLeaks(TB) bool { return true }

// ResetTracking clears all recorded borrow/redeem tracking.
//
// This is a no-op in a release build.
func ResetTracking() {}
