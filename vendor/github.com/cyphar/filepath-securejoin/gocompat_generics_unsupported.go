//go:build linux && !go1.21

// Copyright (C) 2024 SUSE LLC. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package securejoin

import (
	"sync"
)

// These are very minimal implementations of functions that appear in Go 1.21's
// stdlib, included so that we can build on older Go versions. Most are
// borrowed directly from the stdlib, and a few are modified to be "obviously
// correct" without needing to copy too many other helpers.

// clearSlice is equivalent to the builtin clear from Go 1.21.
// Copied from the Go 1.24 stdlib implementation.
func clearSlice[S ~[]E, E any](slice S) {
	var zero E
	for i := range slice {
		slice[i] = zero
	}
}

// Copied from the Go 1.24 stdlib implementation.
func slices_IndexFunc[S ~[]E, E any](s S, f func(E) bool) int {
	for i := range s {
		if f(s[i]) {
			return i
		}
	}
	return -1
}

// Copied from the Go 1.24 stdlib implementation.
func slices_DeleteFunc[S ~[]E, E any](s S, del func(E) bool) S {
	i := slices_IndexFunc(s, del)
	if i == -1 {
		return s
	}
	// Don't start copying elements until we find one to delete.
	for j := i + 1; j < len(s); j++ {
		if v := s[j]; !del(v) {
			s[i] = v
			i++
		}
	}
	clearSlice(s[i:]) // zero/nil out the obsolete elements, for GC
	return s[:i]
}

// Similar to the stdlib slices.Contains, except that we don't have
// slices.Index so we need to use slices.IndexFunc for this non-Func helper.
func slices_Contains[S ~[]E, E comparable](s S, v E) bool {
	return slices_IndexFunc(s, func(e E) bool { return e == v }) >= 0
}

// Copied from the Go 1.24 stdlib implementation.
func slices_Clone[S ~[]E, E any](s S) S {
	// Preserve nil in case it matters.
	if s == nil {
		return nil
	}
	return append(S([]E{}), s...)
}

// Copied from the Go 1.24 stdlib implementation.
func sync_OnceValue[T any](f func() T) func() T {
	var (
		once   sync.Once
		valid  bool
		p      any
		result T
	)
	g := func() {
		defer func() {
			p = recover()
			if !valid {
				panic(p)
			}
		}()
		result = f()
		f = nil
		valid = true
	}
	return func() T {
		once.Do(g)
		if !valid {
			panic(p)
		}
		return result
	}
}

// Copied from the Go 1.24 stdlib implementation.
func sync_OnceValues[T1, T2 any](f func() (T1, T2)) func() (T1, T2) {
	var (
		once  sync.Once
		valid bool
		p     any
		r1    T1
		r2    T2
	)
	g := func() {
		defer func() {
			p = recover()
			if !valid {
				panic(p)
			}
		}()
		r1, r2 = f()
		f = nil
		valid = true
	}
	return func() (T1, T2) {
		once.Do(g)
		if !valid {
			panic(p)
		}
		return r1, r2
	}
}
