// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build illumos
// +build illumos

package unix_test

import (
	"testing"

	"golang.org/x/sys/unix"
)

func TestLifreqSetName(t *testing.T) {
	var l unix.Lifreq
	err := l.SetName("12345678901234356789012345678901234567890")
	if err == nil {
		t.Fatal(`Lifreq.SetName should reject names that are too long`)
	}
	err = l.SetName("tun0")
	if err != nil {
		t.Errorf(`Lifreq.SetName("tun0") failed: %v`, err)
	}
}
