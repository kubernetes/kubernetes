// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package unix_test

import (
	"bytes"
	"net"
	"os"
	"syscall"
	"testing"

	"golang.org/x/sys/unix"
)

// TestSCMCredentials tests the sending and receiving of credentials
// (PID, UID, GID) in an ancillary message between two UNIX
// sockets. The SO_PASSCRED socket option is enabled on the sending
// socket for this to work.
func TestSCMCredentials(t *testing.T) {
	socketTypeTests := []struct {
		socketType int
		dataLen    int
	}{
		{
			unix.SOCK_STREAM,
			1,
		}, {
			unix.SOCK_DGRAM,
			0,
		},
	}

	for _, tt := range socketTypeTests {
		fds, err := unix.Socketpair(unix.AF_LOCAL, tt.socketType, 0)
		if err != nil {
			t.Fatalf("Socketpair: %v", err)
		}
		defer unix.Close(fds[0])
		defer unix.Close(fds[1])

		err = unix.SetsockoptInt(fds[0], unix.SOL_SOCKET, unix.SO_PASSCRED, 1)
		if err != nil {
			t.Fatalf("SetsockoptInt: %v", err)
		}

		srvFile := os.NewFile(uintptr(fds[0]), "server")
		defer srvFile.Close()
		srv, err := net.FileConn(srvFile)
		if err != nil {
			t.Errorf("FileConn: %v", err)
			return
		}
		defer srv.Close()

		cliFile := os.NewFile(uintptr(fds[1]), "client")
		defer cliFile.Close()
		cli, err := net.FileConn(cliFile)
		if err != nil {
			t.Errorf("FileConn: %v", err)
			return
		}
		defer cli.Close()

		var ucred unix.Ucred
		if os.Getuid() != 0 {
			ucred.Pid = int32(os.Getpid())
			ucred.Uid = 0
			ucred.Gid = 0
			oob := unix.UnixCredentials(&ucred)
			_, _, err := cli.(*net.UnixConn).WriteMsgUnix(nil, oob, nil)
			if op, ok := err.(*net.OpError); ok {
				err = op.Err
			}
			if sys, ok := err.(*os.SyscallError); ok {
				err = sys.Err
			}
			if err != syscall.EPERM {
				t.Fatalf("WriteMsgUnix failed with %v, want EPERM", err)
			}
		}

		ucred.Pid = int32(os.Getpid())
		ucred.Uid = uint32(os.Getuid())
		ucred.Gid = uint32(os.Getgid())
		oob := unix.UnixCredentials(&ucred)

		// On SOCK_STREAM, this is internally going to send a dummy byte
		n, oobn, err := cli.(*net.UnixConn).WriteMsgUnix(nil, oob, nil)
		if err != nil {
			t.Fatalf("WriteMsgUnix: %v", err)
		}
		if n != 0 {
			t.Fatalf("WriteMsgUnix n = %d, want 0", n)
		}
		if oobn != len(oob) {
			t.Fatalf("WriteMsgUnix oobn = %d, want %d", oobn, len(oob))
		}

		oob2 := make([]byte, 10*len(oob))
		n, oobn2, flags, _, err := srv.(*net.UnixConn).ReadMsgUnix(nil, oob2)
		if err != nil {
			t.Fatalf("ReadMsgUnix: %v", err)
		}
		if flags != 0 {
			t.Fatalf("ReadMsgUnix flags = 0x%x, want 0", flags)
		}
		if n != tt.dataLen {
			t.Fatalf("ReadMsgUnix n = %d, want %d", n, tt.dataLen)
		}
		if oobn2 != oobn {
			// without SO_PASSCRED set on the socket, ReadMsgUnix will
			// return zero oob bytes
			t.Fatalf("ReadMsgUnix oobn = %d, want %d", oobn2, oobn)
		}
		oob2 = oob2[:oobn2]
		if !bytes.Equal(oob, oob2) {
			t.Fatal("ReadMsgUnix oob bytes don't match")
		}

		scm, err := unix.ParseSocketControlMessage(oob2)
		if err != nil {
			t.Fatalf("ParseSocketControlMessage: %v", err)
		}
		newUcred, err := unix.ParseUnixCredentials(&scm[0])
		if err != nil {
			t.Fatalf("ParseUnixCredentials: %v", err)
		}
		if *newUcred != ucred {
			t.Fatalf("ParseUnixCredentials = %+v, want %+v", newUcred, ucred)
		}
	}
}
