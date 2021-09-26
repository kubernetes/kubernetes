// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd openbsd

package unix_test

import (
	"os/exec"
	"runtime"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

func TestGetfsstat(t *testing.T) {
	n, err := unix.Getfsstat(nil, unix.MNT_NOWAIT)
	if err != nil {
		t.Fatal(err)
	}

	data := make([]unix.Statfs_t, n)
	n2, err := unix.Getfsstat(data, unix.MNT_NOWAIT)
	if err != nil {
		t.Fatal(err)
	}
	if n != n2 {
		t.Errorf("Getfsstat(nil) = %d, but subsequent Getfsstat(slice) = %d", n, n2)
	}
	for i, stat := range data {
		if stat == (unix.Statfs_t{}) {
			t.Errorf("index %v is an empty Statfs_t struct", i)
		}
	}
	if t.Failed() {
		for i, stat := range data[:n2] {
			t.Logf("data[%v] = %+v", i, stat)
		}
		mount, err := exec.Command("mount").CombinedOutput()
		if err != nil {
			t.Logf("mount: %v\n%s", err, mount)
		} else {
			t.Logf("mount: %s", mount)
		}
	}
}

func TestSysctlRaw(t *testing.T) {
	if runtime.GOOS == "openbsd" {
		t.Skip("kern.proc.pid does not exist on OpenBSD")
	}

	_, err := unix.SysctlRaw("kern.proc.pid", unix.Getpid())
	if err != nil {
		t.Fatal(err)
	}
}

func TestSysctlUint32(t *testing.T) {
	maxproc, err := unix.SysctlUint32("kern.maxproc")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("kern.maxproc: %v", maxproc)
}

func TestSysctlClockinfo(t *testing.T) {
	ci, err := unix.SysctlClockinfo("kern.clockrate")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("tick = %v, hz = %v, profhz = %v, stathz = %v",
		ci.Tick, ci.Hz, ci.Profhz, ci.Stathz)
}

func TestSysctlTimeval(t *testing.T) {
	tv, err := unix.SysctlTimeval("kern.boottime")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("boottime = %v", time.Unix(tv.Unix()))
}
