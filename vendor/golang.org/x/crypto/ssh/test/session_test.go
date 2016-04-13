// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows

package test

// Session functional tests.

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"

	"golang.org/x/crypto/ssh"
)

func TestRunCommandSuccess(t *testing.T) {
	server := newServer(t)
	defer server.Shutdown()
	conn := server.Dial(clientConfig())
	defer conn.Close()

	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("session failed: %v", err)
	}
	defer session.Close()
	err = session.Run("true")
	if err != nil {
		t.Fatalf("session failed: %v", err)
	}
}

func TestHostKeyCheck(t *testing.T) {
	server := newServer(t)
	defer server.Shutdown()

	conf := clientConfig()
	hostDB := hostKeyDB()
	conf.HostKeyCallback = hostDB.Check

	// change the keys.
	hostDB.keys[ssh.KeyAlgoRSA][25]++
	hostDB.keys[ssh.KeyAlgoDSA][25]++
	hostDB.keys[ssh.KeyAlgoECDSA256][25]++

	conn, err := server.TryDial(conf)
	if err == nil {
		conn.Close()
		t.Fatalf("dial should have failed.")
	} else if !strings.Contains(err.Error(), "host key mismatch") {
		t.Fatalf("'host key mismatch' not found in %v", err)
	}
}

func TestRunCommandStdin(t *testing.T) {
	server := newServer(t)
	defer server.Shutdown()
	conn := server.Dial(clientConfig())
	defer conn.Close()

	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("session failed: %v", err)
	}
	defer session.Close()

	r, w := io.Pipe()
	defer r.Close()
	defer w.Close()
	session.Stdin = r

	err = session.Run("true")
	if err != nil {
		t.Fatalf("session failed: %v", err)
	}
}

func TestRunCommandStdinError(t *testing.T) {
	server := newServer(t)
	defer server.Shutdown()
	conn := server.Dial(clientConfig())
	defer conn.Close()

	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("session failed: %v", err)
	}
	defer session.Close()

	r, w := io.Pipe()
	defer r.Close()
	session.Stdin = r
	pipeErr := errors.New("closing write end of pipe")
	w.CloseWithError(pipeErr)

	err = session.Run("true")
	if err != pipeErr {
		t.Fatalf("expected %v, found %v", pipeErr, err)
	}
}

func TestRunCommandFailed(t *testing.T) {
	server := newServer(t)
	defer server.Shutdown()
	conn := server.Dial(clientConfig())
	defer conn.Close()

	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("session failed: %v", err)
	}
	defer session.Close()
	err = session.Run(`bash -c "kill -9 $$"`)
	if err == nil {
		t.Fatalf("session succeeded: %v", err)
	}
}

func TestRunCommandWeClosed(t *testing.T) {
	server := newServer(t)
	defer server.Shutdown()
	conn := server.Dial(clientConfig())
	defer conn.Close()

	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("session failed: %v", err)
	}
	err = session.Shell()
	if err != nil {
		t.Fatalf("shell failed: %v", err)
	}
	err = session.Close()
	if err != nil {
		t.Fatalf("shell failed: %v", err)
	}
}

func TestFuncLargeRead(t *testing.T) {
	server := newServer(t)
	defer server.Shutdown()
	conn := server.Dial(clientConfig())
	defer conn.Close()

	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("unable to create new session: %s", err)
	}

	stdout, err := session.StdoutPipe()
	if err != nil {
		t.Fatalf("unable to acquire stdout pipe: %s", err)
	}

	err = session.Start("dd if=/dev/urandom bs=2048 count=1024")
	if err != nil {
		t.Fatalf("unable to execute remote command: %s", err)
	}

	buf := new(bytes.Buffer)
	n, err := io.Copy(buf, stdout)
	if err != nil {
		t.Fatalf("error reading from remote stdout: %s", err)
	}

	if n != 2048*1024 {
		t.Fatalf("Expected %d bytes but read only %d from remote command", 2048, n)
	}
}

func TestKeyChange(t *testing.T) {
	server := newServer(t)
	defer server.Shutdown()
	conf := clientConfig()
	hostDB := hostKeyDB()
	conf.HostKeyCallback = hostDB.Check
	conf.RekeyThreshold = 1024
	conn := server.Dial(conf)
	defer conn.Close()

	for i := 0; i < 4; i++ {
		session, err := conn.NewSession()
		if err != nil {
			t.Fatalf("unable to create new session: %s", err)
		}

		stdout, err := session.StdoutPipe()
		if err != nil {
			t.Fatalf("unable to acquire stdout pipe: %s", err)
		}

		err = session.Start("dd if=/dev/urandom bs=1024 count=1")
		if err != nil {
			t.Fatalf("unable to execute remote command: %s", err)
		}
		buf := new(bytes.Buffer)
		n, err := io.Copy(buf, stdout)
		if err != nil {
			t.Fatalf("error reading from remote stdout: %s", err)
		}

		want := int64(1024)
		if n != want {
			t.Fatalf("Expected %d bytes but read only %d from remote command", want, n)
		}
	}

	if changes := hostDB.checkCount; changes < 4 {
		t.Errorf("got %d key changes, want 4", changes)
	}
}

func TestInvalidTerminalMode(t *testing.T) {
	server := newServer(t)
	defer server.Shutdown()
	conn := server.Dial(clientConfig())
	defer conn.Close()

	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("session failed: %v", err)
	}
	defer session.Close()

	if err = session.RequestPty("vt100", 80, 40, ssh.TerminalModes{255: 1984}); err == nil {
		t.Fatalf("req-pty failed: successful request with invalid mode")
	}
}

func TestValidTerminalMode(t *testing.T) {
	server := newServer(t)
	defer server.Shutdown()
	conn := server.Dial(clientConfig())
	defer conn.Close()

	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("session failed: %v", err)
	}
	defer session.Close()

	stdout, err := session.StdoutPipe()
	if err != nil {
		t.Fatalf("unable to acquire stdout pipe: %s", err)
	}

	stdin, err := session.StdinPipe()
	if err != nil {
		t.Fatalf("unable to acquire stdin pipe: %s", err)
	}

	tm := ssh.TerminalModes{ssh.ECHO: 0}
	if err = session.RequestPty("xterm", 80, 40, tm); err != nil {
		t.Fatalf("req-pty failed: %s", err)
	}

	err = session.Shell()
	if err != nil {
		t.Fatalf("session failed: %s", err)
	}

	stdin.Write([]byte("stty -a && exit\n"))

	var buf bytes.Buffer
	if _, err := io.Copy(&buf, stdout); err != nil {
		t.Fatalf("reading failed: %s", err)
	}

	if sttyOutput := buf.String(); !strings.Contains(sttyOutput, "-echo ") {
		t.Fatalf("terminal mode failure: expected -echo in stty output, got %s", sttyOutput)
	}
}

func TestCiphers(t *testing.T) {
	var config ssh.Config
	config.SetDefaults()
	cipherOrder := config.Ciphers
	// This cipher will not be tested when commented out in cipher.go it will
	// fallback to the next available as per line 292.
	cipherOrder = append(cipherOrder, "aes128-cbc")

	for _, ciph := range cipherOrder {
		server := newServer(t)
		defer server.Shutdown()
		conf := clientConfig()
		conf.Ciphers = []string{ciph}
		// Don't fail if sshd doesnt have the cipher.
		conf.Ciphers = append(conf.Ciphers, cipherOrder...)
		conn, err := server.TryDial(conf)
		if err == nil {
			conn.Close()
		} else {
			t.Fatalf("failed for cipher %q", ciph)
		}
	}
}

func TestMACs(t *testing.T) {
	var config ssh.Config
	config.SetDefaults()
	macOrder := config.MACs

	for _, mac := range macOrder {
		server := newServer(t)
		defer server.Shutdown()
		conf := clientConfig()
		conf.MACs = []string{mac}
		// Don't fail if sshd doesnt have the MAC.
		conf.MACs = append(conf.MACs, macOrder...)
		if conn, err := server.TryDial(conf); err == nil {
			conn.Close()
		} else {
			t.Fatalf("failed for MAC %q", mac)
		}
	}
}
