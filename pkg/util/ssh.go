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
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	mathrand "math/rand"
	"net"
	"os"
	"time"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"golang.org/x/crypto/ssh"
)

var (
	tunnelOpenCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "ssh_tunnel_open_count",
			Help: "Counter of ssh tunnel total open attempts",
		},
	)
	tunnelOpenFailCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "ssh_tunnel_open_fail_count",
			Help: "Counter of ssh tunnel failed open attempts",
		},
	)
)

func init() {
	prometheus.MustRegister(tunnelOpenCounter)
	prometheus.MustRegister(tunnelOpenFailCounter)
}

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
	signer, err := MakePrivateKeySignerFromFile(keyfile)
	if err != nil {
		return nil, err
	}
	return makeSSHTunnel(user, signer, host)
}

func NewSSHTunnelFromBytes(user string, buffer []byte, host string) (*SSHTunnel, error) {
	signer, err := MakePrivateKeySignerFromBytes(buffer)
	if err != nil {
		return nil, err
	}
	return makeSSHTunnel(user, signer, host)
}

func makeSSHTunnel(user string, signer ssh.Signer, host string) (*SSHTunnel, error) {
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
	tunnelOpenCounter.Inc()
	if err != nil {
		tunnelOpenFailCounter.Inc()
		return err
	}
	return nil
}

func (s *SSHTunnel) Dial(network, address string) (net.Conn, error) {
	if s.client == nil {
		return nil, errors.New("tunnel is not opened.")
	}
	return s.client.Dial(network, address)
}

func (s *SSHTunnel) tunnel(conn net.Conn, remoteHost, remotePort string) error {
	if s.client == nil {
		return errors.New("tunnel is not opened.")
	}
	tunnel, err := s.client.Dial("tcp", net.JoinHostPort(remoteHost, remotePort))
	if err != nil {
		return err
	}
	go s.copyBytes(tunnel, conn)
	go s.copyBytes(conn, tunnel)
	return nil
}

func (s *SSHTunnel) Close() error {
	if s.client == nil {
		return errors.New("Cannot close tunnel. Tunnel was not opened.")
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

func MakePrivateKeySignerFromFile(key string) (ssh.Signer, error) {
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
	return MakePrivateKeySignerFromBytes(buffer)
}

func MakePrivateKeySignerFromBytes(buffer []byte) (ssh.Signer, error) {
	signer, err := ssh.ParsePrivateKey(buffer)
	if err != nil {
		return nil, fmt.Errorf("error parsing SSH key %s: '%v'", buffer, err)
	}
	return signer, nil
}

type SSHTunnelEntry struct {
	Address string
	Tunnel  *SSHTunnel
}

type SSHTunnelList struct {
	entries []SSHTunnelEntry
}

func MakeSSHTunnels(user, keyfile string, addresses []string) (*SSHTunnelList, error) {
	tunnels := []SSHTunnelEntry{}
	for ix := range addresses {
		addr := addresses[ix]
		tunnel, err := NewSSHTunnel(user, keyfile, addr)
		if err != nil {
			return nil, err
		}
		tunnels = append(tunnels, SSHTunnelEntry{addr, tunnel})
	}
	return &SSHTunnelList{tunnels}, nil
}

// Open attempts to open all tunnels in the list, and removes any tunnels that
// failed to open.
func (l *SSHTunnelList) Open() error {
	var openTunnels []SSHTunnelEntry
	for ix := range l.entries {
		if err := l.entries[ix].Tunnel.Open(); err != nil {
			glog.Errorf("Failed to open tunnel %v: %v", l.entries[ix], err)
		} else {
			openTunnels = append(openTunnels, l.entries[ix])
		}
	}
	l.entries = openTunnels
	if len(l.entries) == 0 {
		return errors.New("Failed to open any tunnels.")
	}
	return nil
}

// Close asynchronously closes all tunnels in the list after waiting for 1
// minute. Tunnels will still be open upon this function's return, but should
// no longer be used.
func (l *SSHTunnelList) Close() {
	for ix := range l.entries {
		entry := l.entries[ix]
		go func() {
			defer HandleCrash()
			time.Sleep(1 * time.Minute)
			if err := entry.Tunnel.Close(); err != nil {
				glog.Errorf("Failed to close tunnel %v: %v", entry, err)
			}
		}()
	}
}

func (l *SSHTunnelList) Dial(network, addr string) (net.Conn, error) {
	if len(l.entries) == 0 {
		return nil, fmt.Errorf("Empty tunnel list.")
	}
	return l.entries[mathrand.Int()%len(l.entries)].Tunnel.Dial(network, addr)
}

func (l *SSHTunnelList) Has(addr string) bool {
	for ix := range l.entries {
		if l.entries[ix].Address == addr {
			return true
		}
	}
	return false
}

func (l *SSHTunnelList) Len() int {
	return len(l.entries)
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
