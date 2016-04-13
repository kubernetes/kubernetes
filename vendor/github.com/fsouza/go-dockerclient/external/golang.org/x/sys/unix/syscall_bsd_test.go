// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd openbsd

package unix_test

import (
	"testing"

	"github.com/fsouza/go-dockerclient/external/golang.org/x/sys/unix"
)

const MNT_WAIT = 1

func TestGetfsstat(t *testing.T) {
	n, err := unix.Getfsstat(nil, MNT_WAIT)
	if err != nil {
		t.Fatal(err)
	}

	data := make([]unix.Statfs_t, n)
	n, err = unix.Getfsstat(data, MNT_WAIT)
	if err != nil {
		t.Fatal(err)
	}

	empty := unix.Statfs_t{}
	for _, stat := range data {
		if stat == empty {
			t.Fatal("an empty Statfs_t struct was returned")
		}
	}
}

func TestSysctlRaw(t *testing.T) {
	_, err := unix.SysctlRaw("kern.proc.pid", unix.Getpid())
	if err != nil {
		t.Fatal(err)
	}
}
