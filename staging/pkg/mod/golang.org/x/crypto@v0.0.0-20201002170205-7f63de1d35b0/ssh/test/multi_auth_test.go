// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for ssh client multi-auth
//
// These tests run a simple go ssh client against OpenSSH server
// over unix domain sockets. The tests use multiple combinations
// of password, keyboard-interactive and publickey authentication
// methods.
//
// A wrapper library for making sshd PAM authentication use test
// passwords is required in ./sshd_test_pw.so. If the library does
// not exist these tests will be skipped. See compile instructions
// (for linux) in file ./sshd_test_pw.c.

// +build linux

package test

import (
	"fmt"
	"strings"
	"testing"

	"golang.org/x/crypto/ssh"
)

// test cases
type multiAuthTestCase struct {
	authMethods         []string
	expectedPasswordCbs int
	expectedKbdIntCbs   int
}

// test context
type multiAuthTestCtx struct {
	password       string
	numPasswordCbs int
	numKbdIntCbs   int
}

// create test context
func newMultiAuthTestCtx(t *testing.T) *multiAuthTestCtx {
	password, err := randomPassword()
	if err != nil {
		t.Fatalf("Failed to generate random test password: %s", err.Error())
	}

	return &multiAuthTestCtx{
		password: password,
	}
}

// password callback
func (ctx *multiAuthTestCtx) passwordCb() (secret string, err error) {
	ctx.numPasswordCbs++
	return ctx.password, nil
}

// keyboard-interactive callback
func (ctx *multiAuthTestCtx) kbdIntCb(user, instruction string, questions []string, echos []bool) (answers []string, err error) {
	if len(questions) == 0 {
		return nil, nil
	}

	ctx.numKbdIntCbs++
	if len(questions) == 1 {
		return []string{ctx.password}, nil
	}

	return nil, fmt.Errorf("unsupported keyboard-interactive flow")
}

// TestMultiAuth runs several subtests for different combinations of password, keyboard-interactive and publickey authentication methods
func TestMultiAuth(t *testing.T) {
	testCases := []multiAuthTestCase{
		// Test password,publickey authentication, assert that password callback is called 1 time
		multiAuthTestCase{
			authMethods:         []string{"password", "publickey"},
			expectedPasswordCbs: 1,
		},
		// Test keyboard-interactive,publickey authentication, assert that keyboard-interactive callback is called 1 time
		multiAuthTestCase{
			authMethods:       []string{"keyboard-interactive", "publickey"},
			expectedKbdIntCbs: 1,
		},
		// Test publickey,password authentication, assert that password callback is called 1 time
		multiAuthTestCase{
			authMethods:         []string{"publickey", "password"},
			expectedPasswordCbs: 1,
		},
		// Test publickey,keyboard-interactive authentication, assert that keyboard-interactive callback is called 1 time
		multiAuthTestCase{
			authMethods:       []string{"publickey", "keyboard-interactive"},
			expectedKbdIntCbs: 1,
		},
		// Test password,password authentication, assert that password callback is called 2 times
		multiAuthTestCase{
			authMethods:         []string{"password", "password"},
			expectedPasswordCbs: 2,
		},
	}

	for _, testCase := range testCases {
		t.Run(strings.Join(testCase.authMethods, ","), func(t *testing.T) {
			ctx := newMultiAuthTestCtx(t)

			server := newServerForConfig(t, "MultiAuth", map[string]string{"AuthMethods": strings.Join(testCase.authMethods, ",")})
			defer server.Shutdown()

			clientConfig := clientConfig()
			server.setTestPassword(clientConfig.User, ctx.password)

			publicKeyAuthMethod := clientConfig.Auth[0]
			clientConfig.Auth = nil
			for _, authMethod := range testCase.authMethods {
				switch authMethod {
				case "publickey":
					clientConfig.Auth = append(clientConfig.Auth, publicKeyAuthMethod)
				case "password":
					clientConfig.Auth = append(clientConfig.Auth,
						ssh.RetryableAuthMethod(ssh.PasswordCallback(ctx.passwordCb), 5))
				case "keyboard-interactive":
					clientConfig.Auth = append(clientConfig.Auth,
						ssh.RetryableAuthMethod(ssh.KeyboardInteractive(ctx.kbdIntCb), 5))
				default:
					t.Fatalf("Unknown authentication method %s", authMethod)
				}
			}

			conn := server.Dial(clientConfig)
			defer conn.Close()

			if ctx.numPasswordCbs != testCase.expectedPasswordCbs {
				t.Fatalf("passwordCallback was called %d times, expected %d times", ctx.numPasswordCbs, testCase.expectedPasswordCbs)
			}

			if ctx.numKbdIntCbs != testCase.expectedKbdIntCbs {
				t.Fatalf("keyboardInteractiveCallback was called %d times, expected %d times", ctx.numKbdIntCbs, testCase.expectedKbdIntCbs)
			}
		})
	}
}
