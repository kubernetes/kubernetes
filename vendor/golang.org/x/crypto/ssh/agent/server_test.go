// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agent

import (
	"testing"

	"golang.org/x/crypto/ssh"
)

func TestServer(t *testing.T) {
	c1, c2, err := netPipe()
	if err != nil {
		t.Fatalf("netPipe: %v", err)
	}
	defer c1.Close()
	defer c2.Close()
	client := NewClient(c1)

	go ServeAgent(NewKeyring(), c2)

	testAgentInterface(t, client, testPrivateKeys["rsa"], nil)
}

func TestLockServer(t *testing.T) {
	testLockAgent(NewKeyring(), t)
}

func TestSetupForwardAgent(t *testing.T) {
	a, b, err := netPipe()
	if err != nil {
		t.Fatalf("netPipe: %v", err)
	}

	defer a.Close()
	defer b.Close()

	_, socket, cleanup := startAgent(t)
	defer cleanup()

	serverConf := ssh.ServerConfig{
		NoClientAuth: true,
	}
	serverConf.AddHostKey(testSigners["rsa"])
	incoming := make(chan *ssh.ServerConn, 1)
	go func() {
		conn, _, _, err := ssh.NewServerConn(a, &serverConf)
		if err != nil {
			t.Fatalf("Server: %v", err)
		}
		incoming <- conn
	}()

	conf := ssh.ClientConfig{}
	conn, chans, reqs, err := ssh.NewClientConn(b, "", &conf)
	if err != nil {
		t.Fatalf("NewClientConn: %v", err)
	}
	client := ssh.NewClient(conn, chans, reqs)

	if err := ForwardToRemote(client, socket); err != nil {
		t.Fatalf("SetupForwardAgent: %v", err)
	}

	server := <-incoming
	ch, reqs, err := server.OpenChannel(channelType, nil)
	if err != nil {
		t.Fatalf("OpenChannel(%q): %v", channelType, err)
	}
	go ssh.DiscardRequests(reqs)

	agentClient := NewClient(ch)
	testAgentInterface(t, agentClient, testPrivateKeys["rsa"], nil)
	conn.Close()
}
