// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build zos && s390x
// +build zos,s390x

package unix_test

// Modified from Linux tests for device numbers.

import (
	"fmt"
	"testing"

	"golang.org/x/sys/unix"
)

func TestDevices(t *testing.T) {
	testCases := []struct {
		path  string
		major uint32
		minor uint32
	}{
		// Device nums found using ls -l on z/OS
		{"/dev/null", 4, 0},
		{"/dev/zero", 4, 1},
		{"/dev/random", 4, 2},
		{"/dev/urandom", 4, 2},
		{"/dev/tty", 3, 0},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%s %v:%v", tc.path, tc.major, tc.minor), func(t *testing.T) {
			var stat unix.Stat_t
			err := unix.Stat(tc.path, &stat)
			if err != nil {
				if err == unix.EACCES {
					t.Skip("no permission to stat device, skipping test")
				}
				t.Errorf("failed to stat device: %v", err)
				return
			}

			dev := uint64(stat.Rdev)
			if unix.Major(dev) != tc.major {
				t.Errorf("for %s Major(%#x) == %d, want %d", tc.path, dev, unix.Major(dev), tc.major)
			}
			if unix.Minor(dev) != tc.minor {
				t.Errorf("for %s Minor(%#x) == %d, want %d", tc.path, dev, unix.Minor(dev), tc.minor)
			}
			if unix.Mkdev(tc.major, tc.minor) != dev {
				t.Errorf("for %s Mkdev(%d, %d) == %#x, want %#x", tc.path, tc.major, tc.minor, unix.Mkdev(tc.major, tc.minor), dev)
			}
		})

	}
}
