// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build plan9

package plan9_test

import (
	"testing"

	"golang.org/x/sys/plan9"
)

func testSetGetenv(t *testing.T, key, value string) {
	err := plan9.Setenv(key, value)
	if err != nil {
		t.Fatalf("Setenv failed to set %q: %v", value, err)
	}
	newvalue, found := plan9.Getenv(key)
	if !found {
		t.Fatalf("Getenv failed to find %v variable (want value %q)", key, value)
	}
	if newvalue != value {
		t.Fatalf("Getenv(%v) = %q; want %q", key, newvalue, value)
	}
}

func TestEnv(t *testing.T) {
	testSetGetenv(t, "TESTENV", "AVALUE")
	// make sure TESTENV gets set to "", not deleted
	testSetGetenv(t, "TESTENV", "")
}
