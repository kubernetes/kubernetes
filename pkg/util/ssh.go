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
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"io"
	"io/ioutil"
	mathrand "math/rand"
	"net"
	"os"
	"time"

	"github.com/golang/glog"
	"golang.org/x/crypto/ssh"
)

// TODO: Unit tests for this code, we can spin up a test SSH server with instructions here:
// https://godoc.org/golang.org/x/crypto/ssh#ServerConn
type SSHTunnel struct {
	Config  *ssh.ClientConfig
	Host    string
	SSHPort string
	running bool
	sock    net.Listener
	client  *ssh.Client
}

func (s *SSHTunnel) copyBytes(out io.Writer, in io.Reader) {
	if _, err := io.Copy(out, in); err != nil {
		glog.Errorf("Error in SSH tunnel: %v", err)
	}
}

func NewSSHTunnel(user, keyfile, host string) (*SSHTunnel, error) {
	signer, err := MakePrivateKeySigner(keyfile)
	if err != nil {
		return nil, err
	}
	config := ssh.ClientConfig{
		User: user,
		Auth: []ssh.AuthMethod{ssh.PublicKeys(signer)},
	}
	return &SSHTunnel{
		Config:  &config,
		Host:    host,
		SSHPort: "22",
	}, nil
}

func (s *SSHTunnel) Open() error {
	var err error
	s.client, err = ssh.Dial("tcp", net.JoinHostPort(s.Host, s.SSHPort), s.Config)
	if err != nil {
		return err
	}
	return nil
}

func (s *SSHTunnel) Dial(network, address string) (net.Conn, error) {
	return s.client.Dial(network, address)
}

func (s *SSHTunnel) tunnel(conn net.Conn, remoteHost, remotePort string) error {
	tunnel, err := s.client.Dial("tcp", net.JoinHostPort(remoteHost, remotePort))
	if err != nil {
		return err
	}
	go s.copyBytes(tunnel, conn)
	go s.copyBytes(conn, tunnel)
	return nil
}

func (s *SSHTunnel) Close() error {
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

type SSHTunnelEntry struct {
	Address string
	Tunnel  *SSHTunnel
}

type SSHTunnelList []SSHTunnelEntry

func MakeSSHTunnels(user, keyfile string, addresses []string) (SSHTunnelList, error) {
	tunnels := []SSHTunnelEntry{}
	for ix := range addresses {
		addr := addresses[ix]
		tunnel, err := NewSSHTunnel(user, keyfile, addr)
		if err != nil {
			return nil, err
		}
		tunnels = append(tunnels, SSHTunnelEntry{addr, tunnel})
	}
	return tunnels, nil
}

func (l SSHTunnelList) Open() error {
	for ix := range l {
		if err := l[ix].Tunnel.Open(); err != nil {
			return err
		}
	}
	return nil
}

// Close asynchronously closes all tunnels in the list after waiting for 1
// minute. Tunnels will still be open upon this function's return, but should
// no longer be used.
func (l SSHTunnelList) Close() {
	for ix := range l {
		entry := l[ix]
		go func() {
			time.Sleep(1 * time.Minute)
			if err := entry.Tunnel.Close(); err != nil {
				glog.Errorf("Failed to close tunnel %v: %v", entry, err)
			}
		}()
	}
}

func (l SSHTunnelList) Dial(network, addr string) (net.Conn, error) {
	if len(l) == 0 {
		return nil, fmt.Errorf("Empty tunnel list.")
	}
	return l[mathrand.Int()%len(l)].Tunnel.Dial(network, addr)
}

func (l SSHTunnelList) Has(addr string) bool {
	for ix := range l {
		if l[ix].Address == addr {
			return true
		}
	}
	return false
}

func EncodePrivateKey(private *rsa.PrivateKey) []byte {
	return pem.EncodeToMemory(&pem.Block{
		Bytes: x509.MarshalPKCS1PrivateKey(private),
		Type:  "RSA PRIVATE KEY",
	})
}

func EncodePublicKey(public *rsa.PublicKey) ([]byte, error) {
	publicBytes, err := x509.MarshalPKIXPublicKey(public)
	if err != nil {
		return nil, err
	}

	return pem.EncodeToMemory(&pem.Block{
		Bytes: publicBytes,
		Type:  "PUBLIC KEY",
	}), nil
}

func EncodeSSHKey(public *rsa.PublicKey) ([]byte, error) {
	publicKey, err := ssh.NewPublicKey(public)
	if err != nil {
		return nil, err
	}
	return ssh.MarshalAuthorizedKey(publicKey), nil
}

func GenerateKey(bits int) (*rsa.PrivateKey, *rsa.PublicKey, error) {
	private, err := rsa.GenerateKey(rand.Reader, bits)
	if err != nil {
		return nil, nil, err
	}
	return private, &private.PublicKey, nil
}
