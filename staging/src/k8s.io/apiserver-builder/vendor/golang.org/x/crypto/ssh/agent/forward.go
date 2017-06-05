// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agent

import (
	"errors"
	"io"
	"net"
	"sync"

	"golang.org/x/crypto/ssh"
)

// RequestAgentForwarding sets up agent forwarding for the session.
// ForwardToAgent or ForwardToRemote should be called to route
// the authentication requests.
func RequestAgentForwarding(session *ssh.Session) error {
	ok, err := session.SendRequest("auth-agent-req@openssh.com", true, nil)
	if err != nil {
		return err
	}
	if !ok {
		return errors.New("forwarding request denied")
	}
	return nil
}

// ForwardToAgent routes authentication requests to the given keyring.
func ForwardToAgent(client *ssh.Client, keyring Agent) error {
	channels := client.HandleChannelOpen(channelType)
	if channels == nil {
		return errors.New("agent: already have handler for " + channelType)
	}

	go func() {
		for ch := range channels {
			channel, reqs, err := ch.Accept()
			if err != nil {
				continue
			}
			go ssh.DiscardRequests(reqs)
			go func() {
				ServeAgent(keyring, channel)
				channel.Close()
			}()
		}
	}()
	return nil
}

const channelType = "auth-agent@openssh.com"

// ForwardToRemote routes authentication requests to the ssh-agent
// process serving on the given unix socket.
func ForwardToRemote(client *ssh.Client, addr string) error {
	channels := client.HandleChannelOpen(channelType)
	if channels == nil {
		return errors.New("agent: already have handler for " + channelType)
	}
	conn, err := net.Dial("unix", addr)
	if err != nil {
		return err
	}
	conn.Close()

	go func() {
		for ch := range channels {
			channel, reqs, err := ch.Accept()
			if err != nil {
				continue
			}
			go ssh.DiscardRequests(reqs)
			go forwardUnixSocket(channel, addr)
		}
	}()
	return nil
}

func forwardUnixSocket(channel ssh.Channel, addr string) {
	conn, err := net.Dial("unix", addr)
	if err != nil {
		return
	}

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		io.Copy(conn, channel)
		conn.(*net.UnixConn).CloseWrite()
		wg.Done()
	}()
	go func() {
		io.Copy(channel, conn)
		channel.CloseWrite()
		wg.Done()
	}()

	wg.Wait()
	conn.Close()
	channel.Close()
}
