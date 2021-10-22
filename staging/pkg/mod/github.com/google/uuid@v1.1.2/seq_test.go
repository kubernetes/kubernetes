// Copyright 2016 Google Inc.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uuid

import (
	"flag"
	"runtime"
	"testing"
	"time"
)

// This test is only run when --regressions is passed on the go test line.
var regressions = flag.Bool("regressions", false, "run uuid regression tests")

// TestClockSeqRace tests for a particular race condition of returning two
// identical Version1 UUIDs.  The duration of 1 minute was chosen as the race
// condition, before being fixed, nearly always occurred in under 30 seconds.
func TestClockSeqRace(t *testing.T) {
	if !*regressions {
		t.Skip("skipping regression tests")
	}
	duration := time.Minute

	done := make(chan struct{})
	defer close(done)

	ch := make(chan UUID, 10000)
	ncpu := runtime.NumCPU()
	switch ncpu {
	case 0, 1:
		// We can't run the test effectively.
		t.Skip("skipping race test, only one CPU detected")
		return
	default:
		runtime.GOMAXPROCS(ncpu)
	}
	for i := 0; i < ncpu; i++ {
		go func() {
			for {
				select {
				case <-done:
					return
				case ch <- Must(NewUUID()):
				}
			}
		}()
	}

	uuids := make(map[string]bool)
	cnt := 0
	start := time.Now()
	for u := range ch {
		s := u.String()
		if uuids[s] {
			t.Errorf("duplicate uuid after %d in %v: %s", cnt, time.Since(start), s)
			return
		}
		uuids[s] = true
		if time.Since(start) > duration {
			return
		}
		cnt++
	}
}
