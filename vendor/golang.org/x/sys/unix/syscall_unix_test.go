// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package unix_test

import (
	"flag"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

// Tests that below functions, structures and constants are consistent
// on all Unix-like systems.
func _() {
	// program scheduling priority functions and constants
	var (
		_ func(int, int, int) error   = unix.Setpriority
		_ func(int, int) (int, error) = unix.Getpriority
	)
	const (
		_ int = unix.PRIO_USER
		_ int = unix.PRIO_PROCESS
		_ int = unix.PRIO_PGRP
	)

	// termios constants
	const (
		_ int = unix.TCIFLUSH
		_ int = unix.TCIOFLUSH
		_ int = unix.TCOFLUSH
	)

	// fcntl file locking structure and constants
	var (
		_ = unix.Flock_t{
			Type:   int16(0),
			Whence: int16(0),
			Start:  int64(0),
			Len:    int64(0),
			Pid:    int32(0),
		}
	)
	const (
		_ = unix.F_GETLK
		_ = unix.F_SETLK
		_ = unix.F_SETLKW
	)
}

// TestFcntlFlock tests whether the file locking structure matches
// the calling convention of each kernel.
func TestFcntlFlock(t *testing.T) {
	name := filepath.Join(os.TempDir(), "TestFcntlFlock")
	fd, err := unix.Open(name, unix.O_CREAT|unix.O_RDWR|unix.O_CLOEXEC, 0)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer unix.Unlink(name)
	defer unix.Close(fd)
	flock := unix.Flock_t{
		Type:  unix.F_RDLCK,
		Start: 0, Len: 0, Whence: 1,
	}
	if err := unix.FcntlFlock(uintptr(fd), unix.F_GETLK, &flock); err != nil {
		t.Fatalf("FcntlFlock failed: %v", err)
	}
}

// TestPassFD tests passing a file descriptor over a Unix socket.
//
// This test involved both a parent and child process. The parent
// process is invoked as a normal test, with "go test", which then
// runs the child process by running the current test binary with args
// "-test.run=^TestPassFD$" and an environment variable used to signal
// that the test should become the child process instead.
func TestPassFD(t *testing.T) {
	switch runtime.GOOS {
	case "dragonfly":
		// TODO(jsing): Figure out why sendmsg is returning EINVAL.
		t.Skip("skipping test on dragonfly")
	case "solaris":
		// TODO(aram): Figure out why ReadMsgUnix is returning empty message.
		t.Skip("skipping test on solaris, see issue 7402")
	}
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		passFDChild()
		return
	}

	tempDir, err := ioutil.TempDir("", "TestPassFD")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	fds, err := unix.Socketpair(unix.AF_LOCAL, unix.SOCK_STREAM, 0)
	if err != nil {
		t.Fatalf("Socketpair: %v", err)
	}
	defer unix.Close(fds[0])
	defer unix.Close(fds[1])
	writeFile := os.NewFile(uintptr(fds[0]), "child-writes")
	readFile := os.NewFile(uintptr(fds[1]), "parent-reads")
	defer writeFile.Close()
	defer readFile.Close()

	cmd := exec.Command(os.Args[0], "-test.run=^TestPassFD$", "--", tempDir)
	cmd.Env = []string{"GO_WANT_HELPER_PROCESS=1"}
	if lp := os.Getenv("LD_LIBRARY_PATH"); lp != "" {
		cmd.Env = append(cmd.Env, "LD_LIBRARY_PATH="+lp)
	}
	cmd.ExtraFiles = []*os.File{writeFile}

	out, err := cmd.CombinedOutput()
	if len(out) > 0 || err != nil {
		t.Fatalf("child process: %q, %v", out, err)
	}

	c, err := net.FileConn(readFile)
	if err != nil {
		t.Fatalf("FileConn: %v", err)
	}
	defer c.Close()

	uc, ok := c.(*net.UnixConn)
	if !ok {
		t.Fatalf("unexpected FileConn type; expected UnixConn, got %T", c)
	}

	buf := make([]byte, 32) // expect 1 byte
	oob := make([]byte, 32) // expect 24 bytes
	closeUnix := time.AfterFunc(5*time.Second, func() {
		t.Logf("timeout reading from unix socket")
		uc.Close()
	})
	_, oobn, _, _, err := uc.ReadMsgUnix(buf, oob)
	closeUnix.Stop()

	scms, err := unix.ParseSocketControlMessage(oob[:oobn])
	if err != nil {
		t.Fatalf("ParseSocketControlMessage: %v", err)
	}
	if len(scms) != 1 {
		t.Fatalf("expected 1 SocketControlMessage; got scms = %#v", scms)
	}
	scm := scms[0]
	gotFds, err := unix.ParseUnixRights(&scm)
	if err != nil {
		t.Fatalf("unix.ParseUnixRights: %v", err)
	}
	if len(gotFds) != 1 {
		t.Fatalf("wanted 1 fd; got %#v", gotFds)
	}

	f := os.NewFile(uintptr(gotFds[0]), "fd-from-child")
	defer f.Close()

	got, err := ioutil.ReadAll(f)
	want := "Hello from child process!\n"
	if string(got) != want {
		t.Errorf("child process ReadAll: %q, %v; want %q", got, err, want)
	}
}

// passFDChild is the child process used by TestPassFD.
func passFDChild() {
	defer os.Exit(0)

	// Look for our fd. It should be fd 3, but we work around an fd leak
	// bug here (http://golang.org/issue/2603) to let it be elsewhere.
	var uc *net.UnixConn
	for fd := uintptr(3); fd <= 10; fd++ {
		f := os.NewFile(fd, "unix-conn")
		var ok bool
		netc, _ := net.FileConn(f)
		uc, ok = netc.(*net.UnixConn)
		if ok {
			break
		}
	}
	if uc == nil {
		fmt.Println("failed to find unix fd")
		return
	}

	// Make a file f to send to our parent process on uc.
	// We make it in tempDir, which our parent will clean up.
	flag.Parse()
	tempDir := flag.Arg(0)
	f, err := ioutil.TempFile(tempDir, "")
	if err != nil {
		fmt.Printf("TempFile: %v", err)
		return
	}

	f.Write([]byte("Hello from child process!\n"))
	f.Seek(0, 0)

	rights := unix.UnixRights(int(f.Fd()))
	dummyByte := []byte("x")
	n, oobn, err := uc.WriteMsgUnix(dummyByte, rights, nil)
	if err != nil {
		fmt.Printf("WriteMsgUnix: %v", err)
		return
	}
	if n != 1 || oobn != len(rights) {
		fmt.Printf("WriteMsgUnix = %d, %d; want 1, %d", n, oobn, len(rights))
		return
	}
}

// TestUnixRightsRoundtrip tests that UnixRights, ParseSocketControlMessage,
// and ParseUnixRights are able to successfully round-trip lists of file descriptors.
func TestUnixRightsRoundtrip(t *testing.T) {
	testCases := [...][][]int{
		{{42}},
		{{1, 2}},
		{{3, 4, 5}},
		{{}},
		{{1, 2}, {3, 4, 5}, {}, {7}},
	}
	for _, testCase := range testCases {
		b := []byte{}
		var n int
		for _, fds := range testCase {
			// Last assignment to n wins
			n = len(b) + unix.CmsgLen(4*len(fds))
			b = append(b, unix.UnixRights(fds...)...)
		}
		// Truncate b
		b = b[:n]

		scms, err := unix.ParseSocketControlMessage(b)
		if err != nil {
			t.Fatalf("ParseSocketControlMessage: %v", err)
		}
		if len(scms) != len(testCase) {
			t.Fatalf("expected %v SocketControlMessage; got scms = %#v", len(testCase), scms)
		}
		for i, scm := range scms {
			gotFds, err := unix.ParseUnixRights(&scm)
			if err != nil {
				t.Fatalf("ParseUnixRights: %v", err)
			}
			wantFds := testCase[i]
			if len(gotFds) != len(wantFds) {
				t.Fatalf("expected %v fds, got %#v", len(wantFds), gotFds)
			}
			for j, fd := range gotFds {
				if fd != wantFds[j] {
					t.Fatalf("expected fd %v, got %v", wantFds[j], fd)
				}
			}
		}
	}
}

func TestRlimit(t *testing.T) {
	var rlimit, zero unix.Rlimit
	err := unix.Getrlimit(unix.RLIMIT_NOFILE, &rlimit)
	if err != nil {
		t.Fatalf("Getrlimit: save failed: %v", err)
	}
	if zero == rlimit {
		t.Fatalf("Getrlimit: save failed: got zero value %#v", rlimit)
	}
	set := rlimit
	set.Cur = set.Max - 1
	err = unix.Setrlimit(unix.RLIMIT_NOFILE, &set)
	if err != nil {
		t.Fatalf("Setrlimit: set failed: %#v %v", set, err)
	}
	var get unix.Rlimit
	err = unix.Getrlimit(unix.RLIMIT_NOFILE, &get)
	if err != nil {
		t.Fatalf("Getrlimit: get failed: %v", err)
	}
	set = rlimit
	set.Cur = set.Max - 1
	if set != get {
		// Seems like Darwin requires some privilege to
		// increase the soft limit of rlimit sandbox, though
		// Setrlimit never reports an error.
		switch runtime.GOOS {
		case "darwin":
		default:
			t.Fatalf("Rlimit: change failed: wanted %#v got %#v", set, get)
		}
	}
	err = unix.Setrlimit(unix.RLIMIT_NOFILE, &rlimit)
	if err != nil {
		t.Fatalf("Setrlimit: restore failed: %#v %v", rlimit, err)
	}
}

func TestSeekFailure(t *testing.T) {
	_, err := unix.Seek(-1, 0, 0)
	if err == nil {
		t.Fatalf("Seek(-1, 0, 0) did not fail")
	}
	str := err.Error() // used to crash on Linux
	t.Logf("Seek: %v", str)
	if str == "" {
		t.Fatalf("Seek(-1, 0, 0) return error with empty message")
	}
}
