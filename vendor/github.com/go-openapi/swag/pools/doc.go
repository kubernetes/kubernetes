// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

// Package pools provide utilities to recycle allocated objects.
//
// This package provides:
//
// - a generic [Pool] type that wraps [sync.Pool],
// - a [PoolRedeemable] variant that hands out a cached redeem closure,
// - a [PoolSlice] for recycling slices without juggling pointers.
//
// # Debug build
//
// Building with the "poolsdebug" tag (go test -tags poolsdebug ./...) turns on
// instrumentation that tracks every borrow and redeem and panics on misuse:
//
// - double redeem (including the A -> B -> A case for the redeemable pools),
// - redeem of a foreign object,
// - borrow of an object still checked out
//
// It reports the offending call sites.
//
// [AssertNoLeaks] then reports any object borrowed but never redeemed.
//
// The instrumentation is a no-op with zero overhead when the tag is absent.
package pools
