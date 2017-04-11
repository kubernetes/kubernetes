// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd plan9

package test

// functional test harness for unix.

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"net"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"testing"
	"text/template"

	"golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/testdata"
)

const sshd_config = `
Protocol 2
HostKey {{.Dir}}/id_rsa
HostKey {{.Dir}}/id_dsa
HostKey {{.Dir}}/id_ecdsa
Pidfile {{.Dir}}/sshd.pid
#UsePrivilegeSeparation no
KeyRegenerationInterval 3600
ServerKeyBits 768
SyslogFacility AUTH
LogLevel DEBUG2
LoginGraceTime 120
PermitRootLogin no
StrictModes no
RSAAuthentication yes
PubkeyAuthentication yes
AuthorizedKeysFile	{{.Dir}}/authorized_keys
TrustedUserCAKeys {{.Dir}}/id_ecdsa.pub
IgnoreRhosts yes
RhostsRSAAuthentication no
HostbasedAuthentication no
PubkeyAcceptedKeyTypes=*
`

var configTmpl = template.Must(template.New("").Parse(sshd_config))

type server struct {
	t          *testing.T
	cleanup    func() // executed during Shutdown
	configfile string
	cmd        *exec.Cmd
	output     bytes.Buffer // holds stderr from sshd process

	// Client half of the network connection.
	clientConn net.Conn
}

func username() string {
	var username string
	if user, err := user.Current(); err == nil {
		username = user.Username
	} else {
		// user.Current() currently requires cgo. If an error is
		// returned attempt to get the username from the environment.
		log.Printf("user.Current: %v; falling back on $USER", err)
		username = os.Getenv("USER")
	}
	if username == "" {
		panic("Unable to get username")
	}
	return username
}

type storedHostKey struct {
	// keys map from an algorithm string to binary key data.
	keys map[string][]byte

	// checkCount counts the Check calls. Used for testing
	// rekeying.
	checkCount int
}

func (k *storedHostKey) Add(key ssh.PublicKey) {
	if k.keys == nil {
		k.keys = map[string][]byte{}
	}
	k.keys[key.Type()] = key.Marshal()
}

func (k *storedHostKey) Check(addr string, remote net.Addr, key ssh.PublicKey) error {
	k.checkCount++
	algo := key.Type()

	if k.keys == nil || bytes.Compare(key.Marshal(), k.keys[algo]) != 0 {
		return fmt.Errorf("host key mismatch. Got %q, want %q", key, k.keys[algo])
	}
	return nil
}

func hostKeyDB() *storedHostKey {
	keyChecker := &storedHostKey{}
	keyChecker.Add(testPublicKeys["ecdsa"])
	keyChecker.Add(testPublicKeys["rsa"])
	keyChecker.Add(testPublicKeys["dsa"])
	return keyChecker
}

func clientConfig() *ssh.ClientConfig {
	config := &ssh.ClientConfig{
		User: username(),
		Auth: []ssh.AuthMethod{
			ssh.PublicKeys(testSigners["user"]),
		},
		HostKeyCallback: hostKeyDB().Check,
	}
	return config
}

// unixConnection creates two halves of a connected net.UnixConn.  It
// is used for connecting the Go SSH client with sshd without opening
// ports.
func unixConnection() (*net.UnixConn, *net.UnixConn, error) {
	dir, err := ioutil.TempDir("", "unixConnection")
	if err != nil {
		return nil, nil, err
	}
	defer os.Remove(dir)

	addr := filepath.Join(dir, "ssh")
	listener, err := net.Listen("unix", addr)
	if err != nil {
		return nil, nil, err
	}
	defer listener.Close()
	c1, err := net.Dial("unix", addr)
	if err != nil {
		return nil, nil, err
	}

	c2, err := listener.Accept()
	if err != nil {
		c1.Close()
		return nil, nil, err
	}

	return c1.(*net.UnixConn), c2.(*net.UnixConn), nil
}

func (s *server) TryDial(config *ssh.ClientConfig) (*ssh.Client, error) {
	sshd, err := exec.LookPath("sshd")
	if err != nil {
		s.t.Skipf("skipping test: %v", err)
	}

	c1, c2, err := unixConnection()
	if err != nil {
		s.t.Fatalf("unixConnection: %v", err)
	}

	s.cmd = exec.Command(sshd, "-f", s.configfile, "-i", "-e")
	f, err := c2.File()
	if err != nil {
		s.t.Fatalf("UnixConn.File: %v", err)
	}
	defer f.Close()
	s.cmd.Stdin = f
	s.cmd.Stdout = f
	s.cmd.Stderr = &s.output
	if err := s.cmd.Start(); err != nil {
		s.t.Fail()
		s.Shutdown()
		s.t.Fatalf("s.cmd.Start: %v", err)
	}
	s.clientConn = c1
	conn, chans, reqs, err := ssh.NewClientConn(c1, "", config)
	if err != nil {
		return nil, err
	}
	return ssh.NewClient(conn, chans, reqs), nil
}

func (s *server) Dial(config *ssh.ClientConfig) *ssh.Client {
	conn, err := s.TryDial(config)
	if err != nil {
		s.t.Fail()
		s.Shutdown()
		s.t.Fatalf("ssh.Client: %v", err)
	}
	return conn
}

func (s *server) Shutdown() {
	if s.cmd != nil && s.cmd.Process != nil {
		// Don't check for errors; if it fails it's most
		// likely "os: process already finished", and we don't
		// care about that. Use os.Interrupt, so child
		// processes are killed too.
		s.cmd.Process.Signal(os.Interrupt)
		s.cmd.Wait()
	}
	if s.t.Failed() {
		// log any output from sshd process
		s.t.Logf("sshd: %s", s.output.String())
	}
	s.cleanup()
}

func writeFile(path string, contents []byte) {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_TRUNC|os.O_CREATE, 0600)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	if _, err := f.Write(contents); err != nil {
		panic(err)
	}
}

// newServer returns a new mock ssh server.
func newServer(t *testing.T) *server {
	if testing.Short() {
		t.Skip("skipping test due to -short")
	}
	dir, err := ioutil.TempDir("", "sshtest")
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(filepath.Join(dir, "sshd_config"))
	if err != nil {
		t.Fatal(err)
	}
	err = configTmpl.Execute(f, map[string]string{
		"Dir": dir,
	})
	if err != nil {
		t.Fatal(err)
	}
	f.Close()

	for k, v := range testdata.PEMBytes {
		filename := "id_" + k
		writeFile(filepath.Join(dir, filename), v)
		writeFile(filepath.Join(dir, filename+".pub"), ssh.MarshalAuthorizedKey(testPublicKeys[k]))
	}

	var authkeys bytes.Buffer
	for k, _ := range testdata.PEMBytes {
		authkeys.Write(ssh.MarshalAuthorizedKey(testPublicKeys[k]))
	}
	writeFile(filepath.Join(dir, "authorized_keys"), authkeys.Bytes())

	return &server{
		t:          t,
		configfile: f.Name(),
		cleanup: func() {
			if err := os.RemoveAll(dir); err != nil {
				t.Error(err)
			}
		},
	}
}
