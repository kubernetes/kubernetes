// Copyright (c) 2014 The fileutil authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fileutil

import (
	"io/ioutil"
	"os"
	"testing"
)

func TestPunch(t *testing.T) {
	file, err := ioutil.TempFile("", "punchhole-")
	if err != nil {
		t.Error(err)
	}
	defer os.Remove(file.Name())
	defer file.Close()
	buf := make([]byte, 10<<20)
	for i := range buf {
		buf[i] = byte(1 + (i+1)&0xfe)
	}
	if _, err = file.Write(buf); err != nil {
		t.Errorf("error writing to the temp file: %v", err)
		t.FailNow()
	}
	if err = file.Sync(); err != nil {
		t.Logf("error syncing %q: %v", file.Name(), err)
	}
	for i, j := range []int{1, 31, 1 << 10} {
		if err = PunchHole(file, int64(j), int64(j)); err != nil {
			t.Errorf("%d. error punching at %d, size %d: %v", i, j, j, err)
			continue
		}
		// read back, with 1-1 bytes overlaid
		n, err := file.ReadAt(buf[:j+2], int64(j-1))
		if err != nil {
			t.Errorf("%d. error reading file: %v", i, err)
			continue
		}
		buf = buf[:n]
		if buf[0] == 0 {
			t.Errorf("%d. file at %d has been overwritten with 0!", i, j-1)
		}
		if buf[n-1] == 0 {
			t.Errorf("%d. file at %d has been overwritten with 0!", i, j-1+n)
		}
		for k, v := range buf[1 : n-1] {
			if v != 0 {
				t.Errorf("%d. error reading file at %d got %d, want 0.", i, k, v)
			}
		}
	}
}
