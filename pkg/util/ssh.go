/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package util

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"os"

	"github.com/golang/glog"
	"golang.org/x/crypto/ssh"
)

// TODO: Unit tests for this code, we can spin up a test SSH server with instructions here:
// https://godoc.org/golang.org/x/crypto/ssh#ServerConn
type SSHTunnel struct {
	Config     *ssh.ClientConfig
	Host       string
	SSHPort    int
	LocalPort  int
	RemoteHost string
	RemotePort int
	running    bool
	sock       net.Listener
	client     *ssh.Client
}

func (s *SSHTunnel) copyBytes(out io.Writer, in io.Reader) {
	if _, err := io.Copy(out, in); err != nil {
		glog.Errorf("Error in SSH tunnel: %v", err)
	}
}

func NewSSHTunnel(user, keyfile, host, remoteHost string, localPort, remotePort int) (*SSHTunnel, error) {
	signer, err := MakePrivateKeySigner(keyfile)
	if err != nil {
		return nil, err
	}
	config := ssh.ClientConfig{
		User: user,
		Auth: []ssh.AuthMethod{ssh.PublicKeys(signer)},
	}
	return &SSHTunnel{
		Config:     &config,
		Host:       host,
		SSHPort:    22,
		LocalPort:  localPort,
		RemotePort: remotePort,
		RemoteHost: remoteHost,
	}, nil
}

func (s *SSHTunnel) Open() error {
	var err error
	s.client, err = ssh.Dial("tcp", fmt.Sprintf("%s:%d", s.Host, s.SSHPort), s.Config)
	if err != nil {
		return err
	}
	s.sock, err = net.Listen("tcp", fmt.Sprintf("localhost:%d", s.LocalPort))
	if err != nil {
		return err
	}
	s.running = true
	return nil
}

func (s *SSHTunnel) Listen() {
	for s.running {
		conn, err := s.sock.Accept()
		if err != nil {
			glog.Errorf("Error listening for ssh tunnel to %s (%v)", s.RemoteHost, err)
			continue
		}
		if err := s.tunnel(conn); err != nil {
			glog.Errorf("Error starting tunnel: %v", err)
		}
	}
}

func (s *SSHTunnel) tunnel(conn net.Conn) error {
	tunnel, err := s.client.Dial("tcp", fmt.Sprintf("%s:%d", s.RemoteHost, s.RemotePort))
	if err != nil {
		return err
	}
	go s.copyBytes(tunnel, conn)
	go s.copyBytes(conn, tunnel)
	return nil
}

func (s *SSHTunnel) Close() error {
	// TODO: try to shutdown copying here?
	s.running = false
	// TODO: Aggregate errors and keep going?
	if err := s.sock.Close(); err != nil {
		return err
	}
	if err := s.client.Close(); err != nil {
		return err
	}
	return nil
}

func RunSSHCommand(cmd, host string, signer ssh.Signer) (string, string, int, error) {
	// Setup the config, dial the server, and open a session.
	config := &ssh.ClientConfig{
		User: os.Getenv("USER"),
		Auth: []ssh.AuthMethod{ssh.PublicKeys(signer)},
	}
	client, err := ssh.Dial("tcp", host, config)
	if err != nil {
		return "", "", 0, fmt.Errorf("error getting SSH client to host %s: '%v'", host, err)
	}
	session, err := client.NewSession()
	if err != nil {
		return "", "", 0, fmt.Errorf("error creating session to host %s: '%v'", host, err)
	}
	defer session.Close()

	// Run the command.
	code := 0
	var bout, berr bytes.Buffer
	session.Stdout, session.Stderr = &bout, &berr
	if err = session.Run(cmd); err != nil {
		// Check whether the command failed to run or didn't complete.
		if exiterr, ok := err.(*ssh.ExitError); ok {
			// If we got an ExitError and the exit code is nonzero, we'll
			// consider the SSH itself successful (just that the command run
			// errored on the host).
			if code = exiterr.ExitStatus(); code != 0 {
				err = nil
			}
		} else {
			// Some other kind of error happened (e.g. an IOError); consider the
			// SSH unsuccessful.
			err = fmt.Errorf("failed running `%s` on %s: '%v'", cmd, host, err)
		}
	}
	return bout.String(), berr.String(), code, err
}

func MakePrivateKeySigner(key string) (ssh.Signer, error) {
	// Create an actual signer.
	file, err := os.Open(key)
	if err != nil {
		return nil, fmt.Errorf("error opening SSH key %s: '%v'", key, err)
	}
	defer file.Close()
	buffer, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("error reading SSH key %s: '%v'", key, err)
	}
	signer, err := ssh.ParsePrivateKey(buffer)
	if err != nil {
		return nil, fmt.Errorf("error parsing SSH key %s: '%v'", key, err)
	}
	return signer, nil
}
