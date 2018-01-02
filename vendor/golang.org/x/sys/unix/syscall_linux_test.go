// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package unix_test

import (
	"io/ioutil"
	"os"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

func TestFchmodat(t *testing.T) {
	defer chtmpdir(t)()

	touch(t, "file1")
	os.Symlink("file1", "symlink1")

	err := unix.Fchmodat(unix.AT_FDCWD, "symlink1", 0444, 0)
	if err != nil {
		t.Fatalf("Fchmodat: unexpected error: %v", err)
	}

	fi, err := os.Stat("file1")
	if err != nil {
		t.Fatal(err)
	}

	if fi.Mode() != 0444 {
		t.Errorf("Fchmodat: failed to change mode: expected %v, got %v", 0444, fi.Mode())
	}

	err = unix.Fchmodat(unix.AT_FDCWD, "symlink1", 0444, unix.AT_SYMLINK_NOFOLLOW)
	if err != unix.EOPNOTSUPP {
		t.Fatalf("Fchmodat: unexpected error: %v, expected EOPNOTSUPP", err)
	}
}

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

func TestPpoll(t *testing.T) {
	f, cleanup := mktmpfifo(t)
	defer cleanup()

	const timeout = 100 * time.Millisecond

	ok := make(chan bool, 1)
	go func() {
		select {
		case <-time.After(10 * timeout):
			t.Errorf("Ppoll: failed to timeout after %d", 10*timeout)
		case <-ok:
		}
	}()

	fds := []unix.PollFd{{Fd: int32(f.Fd()), Events: unix.POLLIN}}
	timeoutTs := unix.NsecToTimespec(int64(timeout))
	n, err := unix.Ppoll(fds, &timeoutTs, nil)
	ok <- true
	if err != nil {
		t.Errorf("Ppoll: unexpected error: %v", err)
		return
	}
	if n != 0 {
		t.Errorf("Ppoll: wrong number of events: got %v, expected %v", n, 0)
		return
	}
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

		if int64(ut) == now.Unix() {
			return
		}
	}

	t.Errorf("Time: return value %v should be nearly equal to time.Now().Unix() %v", ut, now.Unix())
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

func TestUtimesNanoAt(t *testing.T) {
	defer chtmpdir(t)()

	symlink := "symlink1"
	os.Remove(symlink)
	err := os.Symlink("nonexisting", symlink)
	if err != nil {
		t.Fatal(err)
	}

	ts := []unix.Timespec{
		{Sec: 1111, Nsec: 2222},
		{Sec: 3333, Nsec: 4444},
	}
	err = unix.UtimesNanoAt(unix.AT_FDCWD, symlink, ts, unix.AT_SYMLINK_NOFOLLOW)
	if err != nil {
		t.Fatalf("UtimesNanoAt: %v", err)
	}

	var st unix.Stat_t
	err = unix.Lstat(symlink, &st)
	if err != nil {
		t.Fatalf("Lstat: %v", err)
	}
	if st.Atim != ts[0] {
		t.Errorf("UtimesNanoAt: wrong atime: %v", st.Atim)
	}
	if st.Mtim != ts[1] {
		t.Errorf("UtimesNanoAt: wrong mtime: %v", st.Mtim)
	}
}

func TestGetrlimit(t *testing.T) {
	var rlim unix.Rlimit
	err := unix.Getrlimit(unix.RLIMIT_AS, &rlim)
	if err != nil {
		t.Fatalf("Getrlimit: %v", err)
	}
}

func TestSelect(t *testing.T) {
	_, err := unix.Select(0, nil, nil, nil, &unix.Timeval{0, 0})
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
}

func TestUname(t *testing.T) {
	var utsname unix.Utsname
	err := unix.Uname(&utsname)
	if err != nil {
		t.Fatalf("Uname: %v", err)
	}

	// conversion from []byte to string, golang.org/issue/20753
	t.Logf("OS: %s/%s %s", string(utsname.Sysname[:]), string(utsname.Machine[:]), string(utsname.Release[:]))
}

// utilities taken from os/os_test.go

func touch(t *testing.T, name string) {
	f, err := os.Create(name)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

// chtmpdir changes the working directory to a new temporary directory and
// provides a cleanup function. Used when PWD is read-only.
func chtmpdir(t *testing.T) func() {
	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("chtmpdir: %v", err)
	}
	d, err := ioutil.TempDir("", "test")
	if err != nil {
		t.Fatalf("chtmpdir: %v", err)
	}
	if err := os.Chdir(d); err != nil {
		t.Fatalf("chtmpdir: %v", err)
	}
	return func() {
		if err := os.Chdir(oldwd); err != nil {
			t.Fatalf("chtmpdir: %v", err)
		}
		os.RemoveAll(d)
	}
}
