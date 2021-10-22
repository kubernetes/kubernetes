// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build solaris
// +build solaris

package unix_test

import (
	"os/exec"
	"testing"

	"golang.org/x/sys/unix"
)

func TestStatvfs(t *testing.T) {
	if err := unix.Statvfs("", nil); err == nil {
		t.Fatal(`Statvfs("") expected failure`)
	}

	statvfs := unix.Statvfs_t{}
	if err := unix.Statvfs("/", &statvfs); err != nil {
		t.Errorf(`Statvfs("/") failed: %v`, err)
	}

	if t.Failed() {
		mount, err := exec.Command("mount").CombinedOutput()
		if err != nil {
			t.Logf("mount: %v\n%s", err, mount)
		} else {
			t.Logf("mount: %s", mount)
		}
	}
}

func TestSysconf(t *testing.T) {
	n, err := unix.Sysconf(3 /* SC_CLK_TCK */)
	if err != nil {
		t.Fatalf("Sysconf: %v", err)
	}
	t.Logf("Sysconf(SC_CLK_TCK) = %d", n)
}
