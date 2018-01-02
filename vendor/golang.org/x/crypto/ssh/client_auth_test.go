// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bytes"
	"crypto/rand"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
)

type keyboardInteractive map[string]string

func (cr keyboardInteractive) Challenge(user string, instruction string, questions []string, echos []bool) ([]string, error) {
	var answers []string
	for _, q := range questions {
		answers = append(answers, cr[q])
	}
	return answers, nil
}

// reused internally by tests
var clientPassword = "tiger"

// tryAuth runs a handshake with a given config against an SSH server
// with config serverConfig
func tryAuth(t *testing.T, config *ClientConfig) error {
	c1, c2, err := netPipe()
	if err != nil {
		t.Fatalf("netPipe: %v", err)
	}
	defer c1.Close()
	defer c2.Close()

	certChecker := CertChecker{
		IsUserAuthority: func(k PublicKey) bool {
			return bytes.Equal(k.Marshal(), testPublicKeys["ecdsa"].Marshal())
		},
		UserKeyFallback: func(conn ConnMetadata, key PublicKey) (*Permissions, error) {
			if conn.User() == "testuser" && bytes.Equal(key.Marshal(), testPublicKeys["rsa"].Marshal()) {
				return nil, nil
			}

			return nil, fmt.Errorf("pubkey for %q not acceptable", conn.User())
		},
		IsRevoked: func(c *Certificate) bool {
			return c.Serial == 666
		},
	}

	serverConfig := &ServerConfig{
		PasswordCallback: func(conn ConnMetadata, pass []byte) (*Permissions, error) {
			if conn.User() == "testuser" && string(pass) == clientPassword {
				return nil, nil
			}
			return nil, errors.New("password auth failed")
		},
		PublicKeyCallback: certChecker.Authenticate,
		KeyboardInteractiveCallback: func(conn ConnMetadata, challenge KeyboardInteractiveChallenge) (*Permissions, error) {
			ans, err := challenge("user",
				"instruction",
				[]string{"question1", "question2"},
				[]bool{true, true})
			if err != nil {
				return nil, err
			}
			ok := conn.User() == "testuser" && ans[0] == "answer1" && ans[1] == "answer2"
			if ok {
				challenge("user", "motd", nil, nil)
				return nil, nil
			}
			return nil, errors.New("keyboard-interactive failed")
		},
	}
	serverConfig.AddHostKey(testSigners["rsa"])

	go newServer(c1, serverConfig)
	_, _, _, err = NewClientConn(c2, "", config)
	return err
}

func TestClientAuthPublicKey(t *testing.T) {
	config := &ClientConfig{
		User: "testuser",
		Auth: []AuthMethod{
			PublicKeys(testSigners["rsa"]),
		},
		HostKeyCallback: InsecureIgnoreHostKey(),
	}
	if err := tryAuth(t, config); err != nil {
		t.Fatalf("unable to dial remote side: %s", err)
	}
}

func TestAuthMethodPassword(t *testing.T) {
	config := &ClientConfig{
		User: "testuser",
		Auth: []AuthMethod{
			Password(clientPassword),
		},
		HostKeyCallback: InsecureIgnoreHostKey(),
	}

	if err := tryAuth(t, config); err != nil {
		t.Fatalf("unable to dial remote side: %s", err)
	}
}

func TestAuthMethodFallback(t *testing.T) {
	var passwordCalled bool
	config := &ClientConfig{
		User: "testuser",
		Auth: []AuthMethod{
			PublicKeys(testSigners["rsa"]),
			PasswordCallback(
				func() (string, error) {
					passwordCalled = true
					return "WRONG", nil
				}),
		},
		HostKeyCallback: InsecureIgnoreHostKey(),
	}

	if err := tryAuth(t, config); err != nil {
		t.Fatalf("unable to dial remote side: %s", err)
	}

	if passwordCalled {
		t.Errorf("password auth tried before public-key auth.")
	}
}

func TestAuthMethodWrongPassword(t *testing.T) {
	config := &ClientConfig{
		User: "testuser",
		Auth: []AuthMethod{
			Password("wrong"),
			PublicKeys(testSigners["rsa"]),
		},
		HostKeyCallback: InsecureIgnoreHostKey(),
	}

	if err := tryAuth(t, config); err != nil {
		t.Fatalf("unable to dial remote side: %s", err)
	}
}

func TestAuthMethodKeyboardInteractive(t *testing.T) {
	answers := keyboardInteractive(map[string]string{
		"question1": "answer1",
		"question2": "answer2",
	})
	config := &ClientConfig{
		User: "testuser",
		Auth: []AuthMethod{
			KeyboardInteractive(answers.Challenge),
		},
		HostKeyCallback: InsecureIgnoreHostKey(),
	}

	if err := tryAuth(t, config); err != nil {
		t.Fatalf("unable to dial remote side: %s", err)
	}
}

func TestAuthMethodWrongKeyboardInteractive(t *testing.T) {
	answers := keyboardInteractive(map[string]string{
		"question1": "answer1",
		"question2": "WRONG",
	})
	config := &ClientConfig{
		User: "testuser",
		Auth: []AuthMethod{
			KeyboardInteractive(answers.Challenge),
		},
	}

	if err := tryAuth(t, config); err == nil {
		t.Fatalf("wrong answers should not have authenticated with KeyboardInteractive")
	}
}

// the mock server will only authenticate ssh-rsa keys
func TestAuthMethodInvalidPublicKey(t *testing.T) {
	config := &ClientConfig{
		User: "testuser",
		Auth: []AuthMethod{
			PublicKeys(testSigners["dsa"]),
		},
	}

	if err := tryAuth(t, config); err == nil {
		t.Fatalf("dsa private key should not have authenticated with rsa public key")
	}
}

// the client should authenticate with the second key
func TestAuthMethodRSAandDSA(t *testing.T) {
	config := &ClientConfig{
		User: "testuser",
		Auth: []AuthMethod{
			PublicKeys(testSigners["dsa"], testSigners["rsa"]),
		},
		HostKeyCallback: InsecureIgnoreHostKey(),
	}
	if err := tryAuth(t, config); err != nil {
		t.Fatalf("client could not authenticate with rsa key: %v", err)
	}
}

func TestClientHMAC(t *testing.T) {
	for _, mac := range supportedMACs {
		config := &ClientConfig{
			User: "testuser",
			Auth: []AuthMethod{
				PublicKeys(testSigners["rsa"]),
			},
			Config: Config{
				MACs: []string{mac},
			},
			HostKeyCallback: InsecureIgnoreHostKey(),
		}
		if err := tryAuth(t, config); err != nil {
			t.Fatalf("client could not authenticate with mac algo %s: %v", mac, err)
		}
	}
}

// issue 4285.
func TestClientUnsupportedCipher(t *testing.T) {
	config := &ClientConfig{
		User: "testuser",
		Auth: []AuthMethod{
			PublicKeys(),
		},
		Config: Config{
			Ciphers: []string{"aes128-cbc"}, // not currently supported
		},
	}
	if err := tryAuth(t, config); err == nil {
		t.Errorf("expected no ciphers in common")
	}
}

func TestClientUnsupportedKex(t *testing.T) {
	if os.Getenv("GO_BUILDER_NAME") != "" {
		t.Skip("skipping known-flaky test on the Go build dashboard; see golang.org/issue/15198")
	}
	config := &ClientConfig{
		User: "testuser",
		Auth: []AuthMethod{
			PublicKeys(),
		},
		Config: Config{
			KeyExchanges: []string{"diffie-hellman-group-exchange-sha256"}, // not currently supported
		},
		HostKeyCallback: InsecureIgnoreHostKey(),
	}
	if err := tryAuth(t, config); err == nil || !strings.Contains(err.Error(), "common algorithm") {
		t.Errorf("got %v, expected 'common algorithm'", err)
	}
}

func TestClientLoginCert(t *testing.T) {
	cert := &Certificate{
		Key:         testPublicKeys["rsa"],
		ValidBefore: CertTimeInfinity,
		CertType:    UserCert,
	}
	cert.SignCert(rand.Reader, testSigners["ecdsa"])
	certSigner, err := NewCertSigner(cert, testSigners["rsa"])
	if err != nil {
		t.Fatalf("NewCertSigner: %v", err)
	}

	clientConfig := &ClientConfig{
		User:            "user",
		HostKeyCallback: InsecureIgnoreHostKey(),
	}
	clientConfig.Auth = append(clientConfig.Auth, PublicKeys(certSigner))

	// should succeed
	if err := tryAuth(t, clientConfig); err != nil {
		t.Errorf("cert login failed: %v", err)
	}

	// corrupted signature
	cert.Signature.Blob[0]++
	if err := tryAuth(t, clientConfig); err == nil {
		t.Errorf("cert login passed with corrupted sig")
	}

	// revoked
	cert.Serial = 666
	cert.SignCert(rand.Reader, testSigners["ecdsa"])
	if err := tryAuth(t, clientConfig); err == nil {
		t.Errorf("revoked cert login succeeded")
	}
	cert.Serial = 1

	// sign with wrong key
	cert.SignCert(rand.Reader, testSigners["dsa"])
	if err := tryAuth(t, clientConfig); err == nil {
		t.Errorf("cert login passed with non-authoritative key")
	}

	// host cert
	cert.CertType = HostCert
	cert.SignCert(rand.Reader, testSigners["ecdsa"])
	if err := tryAuth(t, clientConfig); err == nil {
		t.Errorf("cert login passed with wrong type")
	}
	cert.CertType = UserCert

	// principal specified
	cert.ValidPrincipals = []string{"user"}
	cert.SignCert(rand.Reader, testSigners["ecdsa"])
	if err := tryAuth(t, clientConfig); err != nil {
		t.Errorf("cert login failed: %v", err)
	}

	// wrong principal specified
	cert.ValidPrincipals = []string{"fred"}
	cert.SignCert(rand.Reader, testSigners["ecdsa"])
	if err := tryAuth(t, clientConfig); err == nil {
		t.Errorf("cert login passed with wrong principal")
	}
	cert.ValidPrincipals = nil

	// added critical option
	cert.CriticalOptions = map[string]string{"root-access": "yes"}
	cert.SignCert(rand.Reader, testSigners["ecdsa"])
	if err := tryAuth(t, clientConfig); err == nil {
		t.Errorf("cert login passed with unrecognized critical option")
	}

	// allowed source address
	cert.CriticalOptions = map[string]string{"source-address": "127.0.0.42/24,::42/120"}
	cert.SignCert(rand.Reader, testSigners["ecdsa"])
	if err := tryAuth(t, clientConfig); err != nil {
		t.Errorf("cert login with source-address failed: %v", err)
	}

	// disallowed source address
	cert.CriticalOptions = map[string]string{"source-address": "127.0.0.42,::42"}
	cert.SignCert(rand.Reader, testSigners["ecdsa"])
	if err := tryAuth(t, clientConfig); err == nil {
		t.Errorf("cert login with source-address succeeded")
	}
}

func testPermissionsPassing(withPermissions bool, t *testing.T) {
	serverConfig := &ServerConfig{
		PublicKeyCallback: func(conn ConnMetadata, key PublicKey) (*Permissions, error) {
			if conn.User() == "nopermissions" {
				return nil, nil
			}
			return &Permissions{}, nil
		},
	}
	serverConfig.AddHostKey(testSigners["rsa"])

	clientConfig := &ClientConfig{
		Auth: []AuthMethod{
			PublicKeys(testSigners["rsa"]),
		},
		HostKeyCallback: InsecureIgnoreHostKey(),
	}
	if withPermissions {
		clientConfig.User = "permissions"
	} else {
		clientConfig.User = "nopermissions"
	}

	c1, c2, err := netPipe()
	if err != nil {
		t.Fatalf("netPipe: %v", err)
	}
	defer c1.Close()
	defer c2.Close()

	go NewClientConn(c2, "", clientConfig)
	serverConn, err := newServer(c1, serverConfig)
	if err != nil {
		t.Fatal(err)
	}
	if p := serverConn.Permissions; (p != nil) != withPermissions {
		t.Fatalf("withPermissions is %t, but Permissions object is %#v", withPermissions, p)
	}
}

func TestPermissionsPassing(t *testing.T) {
	testPermissionsPassing(true, t)
}

func TestNoPermissionsPassing(t *testing.T) {
	testPermissionsPassing(false, t)
}

func TestRetryableAuth(t *testing.T) {
	n := 0
	passwords := []string{"WRONG1", "WRONG2"}

	config := &ClientConfig{
		User: "testuser",
		Auth: []AuthMethod{
			RetryableAuthMethod(PasswordCallback(func() (string, error) {
				p := passwords[n]
				n++
				return p, nil
			}), 2),
			PublicKeys(testSigners["rsa"]),
		},
		HostKeyCallback: InsecureIgnoreHostKey(),
	}

	if err := tryAuth(t, config); err != nil {
		t.Fatalf("unable to dial remote side: %s", err)
	}
	if n != 2 {
		t.Fatalf("Did not try all passwords")
	}
}

func ExampleRetryableAuthMethod(t *testing.T) {
	user := "testuser"
	NumberOfPrompts := 3

	// Normally this would be a callback that prompts the user to answer the
	// provided questions
	Cb := func(user, instruction string, questions []string, echos []bool) (answers []string, err error) {
		return []string{"answer1", "answer2"}, nil
	}

	config := &ClientConfig{
		HostKeyCallback: InsecureIgnoreHostKey(),
		User:            user,
		Auth: []AuthMethod{
			RetryableAuthMethod(KeyboardInteractiveChallenge(Cb), NumberOfPrompts),
		},
	}

	if err := tryAuth(t, config); err != nil {
		t.Fatalf("unable to dial remote side: %s", err)
	}
}

// Test if username is received on server side when NoClientAuth is used
func TestClientAuthNone(t *testing.T) {
	user := "testuser"
	serverConfig := &ServerConfig{
		NoClientAuth: true,
	}
	serverConfig.AddHostKey(testSigners["rsa"])

	clientConfig := &ClientConfig{
		User:            user,
		HostKeyCallback: InsecureIgnoreHostKey(),
	}

	c1, c2, err := netPipe()
	if err != nil {
		t.Fatalf("netPipe: %v", err)
	}
	defer c1.Close()
	defer c2.Close()

	go NewClientConn(c2, "", clientConfig)
	serverConn, err := newServer(c1, serverConfig)
	if err != nil {
		t.Fatalf("newServer: %v", err)
	}
	if serverConn.User() != user {
		t.Fatalf("server: got %q, want %q", serverConn.User(), user)
	}
}

// Test if authentication attempts are limited on server when MaxAuthTries is set
func TestClientAuthMaxAuthTries(t *testing.T) {
	user := "testuser"

	serverConfig := &ServerConfig{
		MaxAuthTries: 2,
		PasswordCallback: func(conn ConnMetadata, pass []byte) (*Permissions, error) {
			if conn.User() == "testuser" && string(pass) == "right" {
				return nil, nil
			}
			return nil, errors.New("password auth failed")
		},
	}
	serverConfig.AddHostKey(testSigners["rsa"])

	expectedErr := fmt.Errorf("ssh: handshake failed: %v", &disconnectMsg{
		Reason:  2,
		Message: "too many authentication failures",
	})

	for tries := 2; tries < 4; tries++ {
		n := tries
		clientConfig := &ClientConfig{
			User: user,
			Auth: []AuthMethod{
				RetryableAuthMethod(PasswordCallback(func() (string, error) {
					n--
					if n == 0 {
						return "right", nil
					}
					return "wrong", nil
				}), tries),
			},
			HostKeyCallback: InsecureIgnoreHostKey(),
		}

		c1, c2, err := netPipe()
		if err != nil {
			t.Fatalf("netPipe: %v", err)
		}
		defer c1.Close()
		defer c2.Close()

		go newServer(c1, serverConfig)
		_, _, _, err = NewClientConn(c2, "", clientConfig)
		if tries > 2 {
			if err == nil {
				t.Fatalf("client: got no error, want %s", expectedErr)
			} else if err.Error() != expectedErr.Error() {
				t.Fatalf("client: got %s, want %s", err, expectedErr)
			}
		} else {
			if err != nil {
				t.Fatalf("client: got %s, want no error", err)
			}
		}
	}
}

// Test if authentication attempts are correctly limited on server
// when more public keys are provided then MaxAuthTries
func TestClientAuthMaxAuthTriesPublicKey(t *testing.T) {
	signers := []Signer{}
	for i := 0; i < 6; i++ {
		signers = append(signers, testSigners["dsa"])
	}

	validConfig := &ClientConfig{
		User: "testuser",
		Auth: []AuthMethod{
			PublicKeys(append([]Signer{testSigners["rsa"]}, signers...)...),
		},
		HostKeyCallback: InsecureIgnoreHostKey(),
	}
	if err := tryAuth(t, validConfig); err != nil {
		t.Fatalf("unable to dial remote side: %s", err)
	}

	expectedErr := fmt.Errorf("ssh: handshake failed: %v", &disconnectMsg{
		Reason:  2,
		Message: "too many authentication failures",
	})
	invalidConfig := &ClientConfig{
		User: "testuser",
		Auth: []AuthMethod{
			PublicKeys(append(signers, testSigners["rsa"])...),
		},
		HostKeyCallback: InsecureIgnoreHostKey(),
	}
	if err := tryAuth(t, invalidConfig); err == nil {
		t.Fatalf("client: got no error, want %s", expectedErr)
	} else if err.Error() != expectedErr.Error() {
		t.Fatalf("client: got %s, want %s", err, expectedErr)
	}
}

// Test whether authentication errors are being properly logged if all
// authentication methods have been exhausted
func TestClientAuthErrorList(t *testing.T) {
	publicKeyErr := errors.New("This is an error from PublicKeyCallback")

	clientConfig := &ClientConfig{
		Auth: []AuthMethod{
			PublicKeys(testSigners["rsa"]),
		},
		HostKeyCallback: InsecureIgnoreHostKey(),
	}
	serverConfig := &ServerConfig{
		PublicKeyCallback: func(_ ConnMetadata, _ PublicKey) (*Permissions, error) {
			return nil, publicKeyErr
		},
	}
	serverConfig.AddHostKey(testSigners["rsa"])

	c1, c2, err := netPipe()
	if err != nil {
		t.Fatalf("netPipe: %v", err)
	}
	defer c1.Close()
	defer c2.Close()

	go NewClientConn(c2, "", clientConfig)
	_, err = newServer(c1, serverConfig)
	if err == nil {
		t.Fatal("newServer: got nil, expected errors")
	}

	authErrs, ok := err.(*ServerAuthError)
	if !ok {
		t.Fatalf("errors: got %T, want *ssh.ServerAuthError", err)
	}
	for i, e := range authErrs.Errors {
		switch i {
		case 0:
			if e.Error() != "no auth passed yet" {
				t.Fatalf("errors: got %v, want no auth passed yet", e.Error())
			}
		case 1:
			if e != publicKeyErr {
				t.Fatalf("errors: got %v, want %v", e, publicKeyErr)
			}
		default:
			t.Fatalf("errors: got %v, expected 2 errors", authErrs.Errors)
		}
	}
}
