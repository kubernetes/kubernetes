// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix

package unix_test

import (
	"os"
	"runtime"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

func TestIoctlGetInt(t *testing.T) {
	f, err := os.Open("/dev/random")
	if err != nil {
		t.Fatalf("failed to open device: %v", err)
	}
	defer f.Close()

	v, err := unix.IoctlGetInt(int(f.Fd()), unix.RNDGETENTCNT)
	if err != nil {
		t.Fatalf("failed to perform ioctl: %v", err)
	}

	t.Logf("%d bits of entropy available", v)
}

func TestTime(t *testing.T) {
	var ut unix.Time_t
	ut2, err := unix.Time(&ut)
	if err != nil {
		t.Fatalf("Time: %v", err)
	}
	if ut != ut2 {
		t.Errorf("Time: return value %v should be equal to argument %v", ut2, ut)
	}

	var now time.Time

	for i := 0; i < 10; i++ {
		ut, err = unix.Time(nil)
		if err != nil {
			t.Fatalf("Time: %v", err)
		}

		now = time.Now()
		diff := int64(ut) - now.Unix()
		if -1 <= diff && diff <= 1 {
			return
		}
	}

	t.Errorf("Time: return value %v should be nearly equal to time.Now().Unix() %v±1", ut, now.Unix())
}

func TestUtime(t *testing.T) {
	defer chtmpdir(t)()

	touch(t, "file1")

	buf := &unix.Utimbuf{
		Modtime: 12345,
	}

	err := unix.Utime("file1", buf)
	if err != nil {
		t.Fatalf("Utime: %v", err)
	}

	fi, err := os.Stat("file1")
	if err != nil {
		t.Fatal(err)
	}

	if fi.ModTime().Unix() != 12345 {
		t.Errorf("Utime: failed to change modtime: expected %v, got %v", 12345, fi.ModTime().Unix())
	}
}

func TestPselect(t *testing.T) {
	if runtime.GOARCH == "ppc64" {
		t.Skip("pselect issue with structure timespec on AIX 7.2 tl0, skipping test")
	}

	_, err := unix.Pselect(0, nil, nil, nil, &unix.Timespec{Sec: 0, Nsec: 0}, nil)
	if err != nil {
		t.Fatalf("Pselect: %v", err)
	}

	dur := 2500 * time.Microsecond
	ts := unix.NsecToTimespec(int64(dur))
	start := time.Now()
	_, err = unix.Pselect(0, nil, nil, nil, &ts, nil)
	took := time.Since(start)
	if err != nil {
		t.Fatalf("Pselect: %v", err)
	}

	if took < dur {
		t.Errorf("Pselect: timeout should have been at least %v, got %v", dur, took)
	}
}
