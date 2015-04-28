// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package store

import (
	"runtime"
	"sort"
	"strings"
	"testing"
)

func interestingGoroutines() (gs []string) {
	buf := make([]byte, 2<<20)
	buf = buf[:runtime.Stack(buf, true)]
	for _, g := range strings.Split(string(buf), "\n\n") {
		sl := strings.SplitN(g, "\n", 2)
		if len(sl) != 2 {
			continue
		}
		stack := strings.TrimSpace(sl[1])
		if stack == "" ||
			strings.Contains(stack, "testing.RunTests") ||
			strings.Contains(stack, "testing.Main(") ||
			strings.Contains(stack, "runtime.goexit") ||
			strings.Contains(stack, "created by runtime.gc") ||
			strings.Contains(stack, "runtime.MHeap_Scavenger") {
			continue
		}
		gs = append(gs, stack)
	}
	sort.Strings(gs)
	return
}

// Verify the other tests didn't leave any goroutines running.
// This is in a file named z_last_test.go so it sorts at the end.
func TestGoroutinesRunning(t *testing.T) {
	if testing.Short() {
		t.Skip("not counting goroutines for leakage in -short mode")
	}
	gs := interestingGoroutines()

	n := 0
	stackCount := make(map[string]int)
	for _, g := range gs {
		stackCount[g]++
		n++
	}

	t.Logf("num goroutines = %d", n)
	if n > 0 {
		t.Error("Too many goroutines.")
		for stack, count := range stackCount {
			t.Logf("%d instances of:\n%s", count, stack)
		}
	}
}
