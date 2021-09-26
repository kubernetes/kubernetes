// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows,!solaris,!js

package test

// Session functional tests.

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"runtime"
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

func TestValidTerminalMode(t *testing.T) {
	if runtime.GOOS == "aix" {
		// On AIX, sshd cannot acquire /dev/pts/* if launched as
		// a non-root user.
		t.Skipf("skipping on %s", runtime.GOOS)
	}
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

func TestWindowChange(t *testing.T) {
	if runtime.GOOS == "aix" {
		// On AIX, sshd cannot acquire /dev/pts/* if launched as
		// a non-root user.
		t.Skipf("skipping on %s", runtime.GOOS)
	}
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

	if err := session.WindowChange(100, 100); err != nil {
		t.Fatalf("window-change failed: %s", err)
	}

	err = session.Shell()
	if err != nil {
		t.Fatalf("session failed: %s", err)
	}

	stdin.Write([]byte("stty size && exit\n"))

	var buf bytes.Buffer
	if _, err := io.Copy(&buf, stdout); err != nil {
		t.Fatalf("reading failed: %s", err)
	}

	if sttyOutput := buf.String(); !strings.Contains(sttyOutput, "100 100") {
		t.Fatalf("terminal WindowChange failure: expected \"100 100\" stty output, got %s", sttyOutput)
	}
}

func testOneCipher(t *testing.T, cipher string, cipherOrder []string) {
	server := newServer(t)
	defer server.Shutdown()
	conf := clientConfig()
	conf.Ciphers = []string{cipher}
	// Don't fail if sshd doesn't have the cipher.
	conf.Ciphers = append(conf.Ciphers, cipherOrder...)
	conn, err := server.TryDial(conf)
	if err != nil {
		t.Fatalf("TryDial: %v", err)
	}
	defer conn.Close()

	numBytes := 4096

	// Exercise sending data to the server
	if _, _, err := conn.Conn.SendRequest("drop-me", false, make([]byte, numBytes)); err != nil {
		t.Fatalf("SendRequest: %v", err)
	}

	// Exercise receiving data from the server
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}

	out, err := session.Output(fmt.Sprintf("dd if=/dev/zero bs=%d count=1", numBytes))
	if err != nil {
		t.Fatalf("Output: %v", err)
	}

	if len(out) != numBytes {
		t.Fatalf("got %d bytes, want %d bytes", len(out), numBytes)
	}
}

var deprecatedCiphers = []string{
	"aes128-cbc", "3des-cbc",
	"arcfour128", "arcfour256",
}

func TestCiphers(t *testing.T) {
	var config ssh.Config
	config.SetDefaults()
	cipherOrder := append(config.Ciphers, deprecatedCiphers...)

	for _, ciph := range cipherOrder {
		t.Run(ciph, func(t *testing.T) {
			testOneCipher(t, ciph, cipherOrder)
		})
	}
}

func TestMACs(t *testing.T) {
	var config ssh.Config
	config.SetDefaults()
	macOrder := config.MACs

	for _, mac := range macOrder {
		t.Run(mac, func(t *testing.T) {
			server := newServer(t)
			defer server.Shutdown()
			conf := clientConfig()
			conf.MACs = []string{mac}
			// Don't fail if sshd doesn't have the MAC.
			conf.MACs = append(conf.MACs, macOrder...)
			if conn, err := server.TryDial(conf); err == nil {
				conn.Close()
			} else {
				t.Fatalf("failed for MAC %q", mac)
			}
		})
	}
}

func TestKeyExchanges(t *testing.T) {
	var config ssh.Config
	config.SetDefaults()
	kexOrder := config.KeyExchanges
	// Based on the discussion in #17230, the key exchange algorithms
	// diffie-hellman-group-exchange-sha1 and diffie-hellman-group-exchange-sha256
	// are not included in the default list of supported kex so we have to add them
	// here manually.
	kexOrder = append(kexOrder, "diffie-hellman-group-exchange-sha1", "diffie-hellman-group-exchange-sha256")
	for _, kex := range kexOrder {
		t.Run(kex, func(t *testing.T) {
			server := newServer(t)
			defer server.Shutdown()
			conf := clientConfig()
			// Don't fail if sshd doesn't have the kex.
			conf.KeyExchanges = append([]string{kex}, kexOrder...)
			conn, err := server.TryDial(conf)
			if err == nil {
				conn.Close()
			} else {
				t.Errorf("failed for kex %q", kex)
			}
		})
	}
}

func TestClientAuthAlgorithms(t *testing.T) {
	for _, key := range []string{
		"rsa",
		"dsa",
		"ecdsa",
		"ed25519",
	} {
		t.Run(key, func(t *testing.T) {
			server := newServer(t)
			conf := clientConfig()
			conf.SetDefaults()
			conf.Auth = []ssh.AuthMethod{
				ssh.PublicKeys(testSigners[key]),
			}

			conn, err := server.TryDial(conf)
			if err == nil {
				conn.Close()
			} else {
				t.Errorf("failed for key %q", key)
			}

			server.Shutdown()
		})
	}
}
