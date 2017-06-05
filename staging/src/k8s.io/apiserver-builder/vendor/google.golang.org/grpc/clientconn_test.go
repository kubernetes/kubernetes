/*
 *
 * Copyright 2014, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

package grpc

import (
	"testing"
	"time"

	"golang.org/x/net/context"

	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/oauth"
)

const tlsDir = "testdata/"

func TestDialTimeout(t *testing.T) {
	conn, err := Dial("Non-Existent.Server:80", WithTimeout(time.Millisecond), WithBlock(), WithInsecure())
	if err == nil {
		conn.Close()
	}
	if err != ErrClientConnTimeout {
		t.Fatalf("Dial(_, _) = %v, %v, want %v", conn, err, ErrClientConnTimeout)
	}
}

func TestTLSDialTimeout(t *testing.T) {
	creds, err := credentials.NewClientTLSFromFile(tlsDir+"ca.pem", "x.test.youtube.com")
	if err != nil {
		t.Fatalf("Failed to create credentials %v", err)
	}
	conn, err := Dial("Non-Existent.Server:80", WithTransportCredentials(creds), WithTimeout(time.Millisecond), WithBlock())
	if err == nil {
		conn.Close()
	}
	if err != ErrClientConnTimeout {
		t.Fatalf("grpc.Dial(_, _) = %v, %v, want %v", conn, err, ErrClientConnTimeout)
	}
}

func TestDialContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	go cancel()
	conn, err := DialContext(ctx, "Non-Existent.Server:80", WithBlock(), WithInsecure())
	if err == nil {
		conn.Close()
	}
	if err != context.Canceled {
		t.Fatalf("DialContext(_, _) = %v, %v, want %v", conn, err, context.Canceled)
	}
}

func TestCredentialsMisuse(t *testing.T) {
	tlsCreds, err := credentials.NewClientTLSFromFile(tlsDir+"ca.pem", "x.test.youtube.com")
	if err != nil {
		t.Fatalf("Failed to create authenticator %v", err)
	}
	// Two conflicting credential configurations
	if _, err := Dial("Non-Existent.Server:80", WithTransportCredentials(tlsCreds), WithBlock(), WithInsecure()); err != errCredentialsConflict {
		t.Fatalf("Dial(_, _) = _, %v, want _, %v", err, errCredentialsConflict)
	}
	rpcCreds, err := oauth.NewJWTAccessFromKey(nil)
	if err != nil {
		t.Fatalf("Failed to create credentials %v", err)
	}
	// security info on insecure connection
	if _, err := Dial("Non-Existent.Server:80", WithPerRPCCredentials(rpcCreds), WithBlock(), WithInsecure()); err != errTransportCredentialsMissing {
		t.Fatalf("Dial(_, _) = _, %v, want _, %v", err, errTransportCredentialsMissing)
	}
}

func TestWithBackoffConfigDefault(t *testing.T) {
	testBackoffConfigSet(t, &DefaultBackoffConfig)
}

func TestWithBackoffConfig(t *testing.T) {
	b := BackoffConfig{MaxDelay: DefaultBackoffConfig.MaxDelay / 2}
	expected := b
	setDefaults(&expected) // defaults should be set
	testBackoffConfigSet(t, &expected, WithBackoffConfig(b))
}

func TestWithBackoffMaxDelay(t *testing.T) {
	md := DefaultBackoffConfig.MaxDelay / 2
	expected := BackoffConfig{MaxDelay: md}
	setDefaults(&expected)
	testBackoffConfigSet(t, &expected, WithBackoffMaxDelay(md))
}

func testBackoffConfigSet(t *testing.T, expected *BackoffConfig, opts ...DialOption) {
	opts = append(opts, WithInsecure())
	conn, err := Dial("foo:80", opts...)
	if err != nil {
		t.Fatalf("unexpected error dialing connection: %v", err)
	}

	if conn.dopts.bs == nil {
		t.Fatalf("backoff config not set")
	}

	actual, ok := conn.dopts.bs.(BackoffConfig)
	if !ok {
		t.Fatalf("unexpected type of backoff config: %#v", conn.dopts.bs)
	}

	if actual != *expected {
		t.Fatalf("unexpected backoff config on connection: %v, want %v", actual, expected)
	}
	conn.Close()
}
