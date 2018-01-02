// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE.BSD file,
// or at https://opensource.org/licenses/BSD-3-Clause

package imagestore

import (
	"fmt"
	"os"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"
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
			strings.Contains(stack, "created by testing.RunTests") ||
			strings.Contains(stack, "testing.Main(") ||
			strings.Contains(stack, "runtime.goexit") ||
			strings.Contains(stack, "github.com/coreos/rkt/store/imagestore.interestingGoroutines") ||
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
func TestMain(m *testing.M) {
	v := m.Run()
	if v == 0 && goroutineLeaked() {
		os.Exit(254)
	}
	os.Exit(v)
}

func goroutineLeaked() bool {
	if testing.Short() {
		// not counting goroutines for leakage in -short mode
		return false
	}

	var stackCount map[string]int
	for i := 0; i < 5; i++ {
		n := 0
		stackCount = make(map[string]int)
		gs := interestingGoroutines()
		for _, g := range gs {
			stackCount[g]++
			n++
		}
		if n == 0 {
			return false
		}
		// Wait for goroutines to schedule and die off:
		time.Sleep(100 * time.Millisecond)
	}
	fmt.Fprintf(os.Stderr, "Too many goroutines running after integration test(s).\n")
	for stack, count := range stackCount {
		fmt.Fprintf(os.Stderr, "%d instances of:\n%s\n", count, stack)
	}
	return true
}
