// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agent

import (
	"bytes"
	"crypto/rand"
	"errors"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"golang.org/x/crypto/ssh"
)

// startAgent executes ssh-agent, and returns a Agent interface to it.
func startAgent(t *testing.T) (client Agent, socket string, cleanup func()) {
	if testing.Short() {
		// ssh-agent is not always available, and the key
		// types supported vary by platform.
		t.Skip("skipping test due to -short")
	}

	bin, err := exec.LookPath("ssh-agent")
	if err != nil {
		t.Skip("could not find ssh-agent")
	}

	cmd := exec.Command(bin, "-s")
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("cmd.Output: %v", err)
	}

	/* Output looks like:

		   SSH_AUTH_SOCK=/tmp/ssh-P65gpcqArqvH/agent.15541; export SSH_AUTH_SOCK;
	           SSH_AGENT_PID=15542; export SSH_AGENT_PID;
	           echo Agent pid 15542;
	*/
	fields := bytes.Split(out, []byte(";"))
	line := bytes.SplitN(fields[0], []byte("="), 2)
	line[0] = bytes.TrimLeft(line[0], "\n")
	if string(line[0]) != "SSH_AUTH_SOCK" {
		t.Fatalf("could not find key SSH_AUTH_SOCK in %q", fields[0])
	}
	socket = string(line[1])

	line = bytes.SplitN(fields[2], []byte("="), 2)
	line[0] = bytes.TrimLeft(line[0], "\n")
	if string(line[0]) != "SSH_AGENT_PID" {
		t.Fatalf("could not find key SSH_AGENT_PID in %q", fields[2])
	}
	pidStr := line[1]
	pid, err := strconv.Atoi(string(pidStr))
	if err != nil {
		t.Fatalf("Atoi(%q): %v", pidStr, err)
	}

	conn, err := net.Dial("unix", string(socket))
	if err != nil {
		t.Fatalf("net.Dial: %v", err)
	}

	ac := NewClient(conn)
	return ac, socket, func() {
		proc, _ := os.FindProcess(pid)
		if proc != nil {
			proc.Kill()
		}
		conn.Close()
		os.RemoveAll(filepath.Dir(socket))
	}
}

func testAgent(t *testing.T, key interface{}, cert *ssh.Certificate, lifetimeSecs uint32) {
	agent, _, cleanup := startAgent(t)
	defer cleanup()

	testAgentInterface(t, agent, key, cert, lifetimeSecs)
}

func testKeyring(t *testing.T, key interface{}, cert *ssh.Certificate, lifetimeSecs uint32) {
	a := NewKeyring()
	testAgentInterface(t, a, key, cert, lifetimeSecs)
}

func testAgentInterface(t *testing.T, agent Agent, key interface{}, cert *ssh.Certificate, lifetimeSecs uint32) {
	signer, err := ssh.NewSignerFromKey(key)
	if err != nil {
		t.Fatalf("NewSignerFromKey(%T): %v", key, err)
	}
	// The agent should start up empty.
	if keys, err := agent.List(); err != nil {
		t.Fatalf("RequestIdentities: %v", err)
	} else if len(keys) > 0 {
		t.Fatalf("got %d keys, want 0: %v", len(keys), keys)
	}

	// Attempt to insert the key, with certificate if specified.
	var pubKey ssh.PublicKey
	if cert != nil {
		err = agent.Add(AddedKey{
			PrivateKey:   key,
			Certificate:  cert,
			Comment:      "comment",
			LifetimeSecs: lifetimeSecs,
		})
		pubKey = cert
	} else {
		err = agent.Add(AddedKey{PrivateKey: key, Comment: "comment", LifetimeSecs: lifetimeSecs})
		pubKey = signer.PublicKey()
	}
	if err != nil {
		t.Fatalf("insert(%T): %v", key, err)
	}

	// Did the key get inserted successfully?
	if keys, err := agent.List(); err != nil {
		t.Fatalf("List: %v", err)
	} else if len(keys) != 1 {
		t.Fatalf("got %v, want 1 key", keys)
	} else if keys[0].Comment != "comment" {
		t.Fatalf("key comment: got %v, want %v", keys[0].Comment, "comment")
	} else if !bytes.Equal(keys[0].Blob, pubKey.Marshal()) {
		t.Fatalf("key mismatch")
	}

	// Can the agent make a valid signature?
	data := []byte("hello")
	sig, err := agent.Sign(pubKey, data)
	if err != nil {
		t.Fatalf("Sign(%s): %v", pubKey.Type(), err)
	}

	if err := pubKey.Verify(data, sig); err != nil {
		t.Fatalf("Verify(%s): %v", pubKey.Type(), err)
	}

	// If the key has a lifetime, is it removed when it should be?
	if lifetimeSecs > 0 {
		time.Sleep(time.Second*time.Duration(lifetimeSecs) + 100*time.Millisecond)
		keys, err := agent.List()
		if err != nil {
			t.Fatalf("List: %v", err)
		}
		if len(keys) > 0 {
			t.Fatalf("key not expired")
		}
	}

}

func TestAgent(t *testing.T) {
	for _, keyType := range []string{"rsa", "dsa", "ecdsa", "ed25519"} {
		testAgent(t, testPrivateKeys[keyType], nil, 0)
		testKeyring(t, testPrivateKeys[keyType], nil, 1)
	}
}

func TestCert(t *testing.T) {
	cert := &ssh.Certificate{
		Key:         testPublicKeys["rsa"],
		ValidBefore: ssh.CertTimeInfinity,
		CertType:    ssh.UserCert,
	}
	cert.SignCert(rand.Reader, testSigners["ecdsa"])

	testAgent(t, testPrivateKeys["rsa"], cert, 0)
	testKeyring(t, testPrivateKeys["rsa"], cert, 1)
}

// netPipe is analogous to net.Pipe, but it uses a real net.Conn, and
// therefore is buffered (net.Pipe deadlocks if both sides start with
// a write.)
func netPipe() (net.Conn, net.Conn, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, nil, err
	}
	defer listener.Close()
	c1, err := net.Dial("tcp", listener.Addr().String())
	if err != nil {
		return nil, nil, err
	}

	c2, err := listener.Accept()
	if err != nil {
		c1.Close()
		return nil, nil, err
	}

	return c1, c2, nil
}

func TestAuth(t *testing.T) {
	a, b, err := netPipe()
	if err != nil {
		t.Fatalf("netPipe: %v", err)
	}

	defer a.Close()
	defer b.Close()

	agent, _, cleanup := startAgent(t)
	defer cleanup()

	if err := agent.Add(AddedKey{PrivateKey: testPrivateKeys["rsa"], Comment: "comment"}); err != nil {
		t.Errorf("Add: %v", err)
	}

	serverConf := ssh.ServerConfig{}
	serverConf.AddHostKey(testSigners["rsa"])
	serverConf.PublicKeyCallback = func(c ssh.ConnMetadata, key ssh.PublicKey) (*ssh.Permissions, error) {
		if bytes.Equal(key.Marshal(), testPublicKeys["rsa"].Marshal()) {
			return nil, nil
		}

		return nil, errors.New("pubkey rejected")
	}

	go func() {
		conn, _, _, err := ssh.NewServerConn(a, &serverConf)
		if err != nil {
			t.Fatalf("Server: %v", err)
		}
		conn.Close()
	}()

	conf := ssh.ClientConfig{}
	conf.Auth = append(conf.Auth, ssh.PublicKeysCallback(agent.Signers))
	conn, _, _, err := ssh.NewClientConn(b, "", &conf)
	if err != nil {
		t.Fatalf("NewClientConn: %v", err)
	}
	conn.Close()
}

func TestLockClient(t *testing.T) {
	agent, _, cleanup := startAgent(t)
	defer cleanup()
	testLockAgent(agent, t)
}

func testLockAgent(agent Agent, t *testing.T) {
	if err := agent.Add(AddedKey{PrivateKey: testPrivateKeys["rsa"], Comment: "comment 1"}); err != nil {
		t.Errorf("Add: %v", err)
	}
	if err := agent.Add(AddedKey{PrivateKey: testPrivateKeys["dsa"], Comment: "comment dsa"}); err != nil {
		t.Errorf("Add: %v", err)
	}
	if keys, err := agent.List(); err != nil {
		t.Errorf("List: %v", err)
	} else if len(keys) != 2 {
		t.Errorf("Want 2 keys, got %v", keys)
	}

	passphrase := []byte("secret")
	if err := agent.Lock(passphrase); err != nil {
		t.Errorf("Lock: %v", err)
	}

	if keys, err := agent.List(); err != nil {
		t.Errorf("List: %v", err)
	} else if len(keys) != 0 {
		t.Errorf("Want 0 keys, got %v", keys)
	}

	signer, _ := ssh.NewSignerFromKey(testPrivateKeys["rsa"])
	if _, err := agent.Sign(signer.PublicKey(), []byte("hello")); err == nil {
		t.Fatalf("Sign did not fail")
	}

	if err := agent.Remove(signer.PublicKey()); err == nil {
		t.Fatalf("Remove did not fail")
	}

	if err := agent.RemoveAll(); err == nil {
		t.Fatalf("RemoveAll did not fail")
	}

	if err := agent.Unlock(nil); err == nil {
		t.Errorf("Unlock with wrong passphrase succeeded")
	}
	if err := agent.Unlock(passphrase); err != nil {
		t.Errorf("Unlock: %v", err)
	}

	if err := agent.Remove(signer.PublicKey()); err != nil {
		t.Fatalf("Remove: %v", err)
	}

	if keys, err := agent.List(); err != nil {
		t.Errorf("List: %v", err)
	} else if len(keys) != 1 {
		t.Errorf("Want 1 keys, got %v", keys)
	}
}

func TestAgentLifetime(t *testing.T) {
	agent, _, cleanup := startAgent(t)
	defer cleanup()

	for _, keyType := range []string{"rsa", "dsa", "ecdsa"} {
		// Add private keys to the agent.
		err := agent.Add(AddedKey{
			PrivateKey:   testPrivateKeys[keyType],
			Comment:      "comment",
			LifetimeSecs: 1,
		})
		if err != nil {
			t.Fatalf("add: %v", err)
		}
		// Add certs to the agent.
		cert := &ssh.Certificate{
			Key:         testPublicKeys[keyType],
			ValidBefore: ssh.CertTimeInfinity,
			CertType:    ssh.UserCert,
		}
		cert.SignCert(rand.Reader, testSigners[keyType])
		err = agent.Add(AddedKey{
			PrivateKey:   testPrivateKeys[keyType],
			Certificate:  cert,
			Comment:      "comment",
			LifetimeSecs: 1,
		})
		if err != nil {
			t.Fatalf("add: %v", err)
		}
	}
	time.Sleep(1100 * time.Millisecond)
	if keys, err := agent.List(); err != nil {
		t.Errorf("List: %v", err)
	} else if len(keys) != 0 {
		t.Errorf("Want 0 keys, got %v", len(keys))
	}
}
