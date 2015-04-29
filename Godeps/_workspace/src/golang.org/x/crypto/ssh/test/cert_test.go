// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd

package test

import (
	"crypto/rand"
	"testing"

	"golang.org/x/crypto/ssh"
)

func TestCertLogin(t *testing.T) {
	s := newServer(t)
	defer s.Shutdown()

	// Use a key different from the default.
	clientKey := testSigners["dsa"]
	caAuthKey := testSigners["ecdsa"]
	cert := &ssh.Certificate{
		Key:             clientKey.PublicKey(),
		ValidPrincipals: []string{username()},
		CertType:        ssh.UserCert,
		ValidBefore:     ssh.CertTimeInfinity,
	}
	if err := cert.SignCert(rand.Reader, caAuthKey); err != nil {
		t.Fatalf("SetSignature: %v", err)
	}

	certSigner, err := ssh.NewCertSigner(cert, clientKey)
	if err != nil {
		t.Fatalf("NewCertSigner: %v", err)
	}

	conf := &ssh.ClientConfig{
		User: username(),
	}
	conf.Auth = append(conf.Auth, ssh.PublicKeys(certSigner))
	client, err := s.TryDial(conf)
	if err != nil {
		t.Fatalf("TryDial: %v", err)
	}
	client.Close()
}
