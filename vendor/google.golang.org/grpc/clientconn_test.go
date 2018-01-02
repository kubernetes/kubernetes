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
	"net"
	"testing"
	"time"

	"golang.org/x/net/context"

	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/keepalive"
)

const tlsDir = "testdata/"

func TestDialTimeout(t *testing.T) {
	conn, err := Dial("Non-Existent.Server:80", WithTimeout(time.Millisecond), WithBlock(), WithInsecure())
	if err == nil {
		conn.Close()
	}
	if err != context.DeadlineExceeded {
		t.Fatalf("Dial(_, _) = %v, %v, want %v", conn, err, context.DeadlineExceeded)
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
	if err != context.DeadlineExceeded {
		t.Fatalf("Dial(_, _) = %v, %v, want %v", conn, err, context.DeadlineExceeded)
	}
}

func TestDefaultAuthority(t *testing.T) {
	target := "Non-Existent.Server:8080"
	conn, err := Dial(target, WithInsecure())
	if err != nil {
		t.Fatalf("Dial(_, _) = _, %v, want _, <nil>", err)
	}
	conn.Close()
	if conn.authority != target {
		t.Fatalf("%v.authority = %v, want %v", conn, conn.authority, target)
	}
}

func TestTLSServerNameOverwrite(t *testing.T) {
	overwriteServerName := "over.write.server.name"
	creds, err := credentials.NewClientTLSFromFile(tlsDir+"ca.pem", overwriteServerName)
	if err != nil {
		t.Fatalf("Failed to create credentials %v", err)
	}
	conn, err := Dial("Non-Existent.Server:80", WithTransportCredentials(creds))
	if err != nil {
		t.Fatalf("Dial(_, _) = _, %v, want _, <nil>", err)
	}
	conn.Close()
	if conn.authority != overwriteServerName {
		t.Fatalf("%v.authority = %v, want %v", conn, conn.authority, overwriteServerName)
	}
}

func TestWithAuthority(t *testing.T) {
	overwriteServerName := "over.write.server.name"
	conn, err := Dial("Non-Existent.Server:80", WithInsecure(), WithAuthority(overwriteServerName))
	if err != nil {
		t.Fatalf("Dial(_, _) = _, %v, want _, <nil>", err)
	}
	conn.Close()
	if conn.authority != overwriteServerName {
		t.Fatalf("%v.authority = %v, want %v", conn, conn.authority, overwriteServerName)
	}
}

func TestWithAuthorityAndTLS(t *testing.T) {
	overwriteServerName := "over.write.server.name"
	creds, err := credentials.NewClientTLSFromFile(tlsDir+"ca.pem", overwriteServerName)
	if err != nil {
		t.Fatalf("Failed to create credentials %v", err)
	}
	conn, err := Dial("Non-Existent.Server:80", WithTransportCredentials(creds), WithAuthority("no.effect.authority"))
	if err != nil {
		t.Fatalf("Dial(_, _) = _, %v, want _, <nil>", err)
	}
	conn.Close()
	if conn.authority != overwriteServerName {
		t.Fatalf("%v.authority = %v, want %v", conn, conn.authority, overwriteServerName)
	}
}

func TestDialContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := DialContext(ctx, "Non-Existent.Server:80", WithBlock(), WithInsecure()); err != context.Canceled {
		t.Fatalf("DialContext(%v, _) = _, %v, want _, %v", ctx, err, context.Canceled)
	}
}

// blockingBalancer mimics the behavior of balancers whose initialization takes a long time.
// In this test, reading from blockingBalancer.Notify() blocks forever.
type blockingBalancer struct {
	ch chan []Address
}

func newBlockingBalancer() Balancer {
	return &blockingBalancer{ch: make(chan []Address)}
}
func (b *blockingBalancer) Start(target string, config BalancerConfig) error {
	return nil
}
func (b *blockingBalancer) Up(addr Address) func(error) {
	return nil
}
func (b *blockingBalancer) Get(ctx context.Context, opts BalancerGetOptions) (addr Address, put func(), err error) {
	return Address{}, nil, nil
}
func (b *blockingBalancer) Notify() <-chan []Address {
	return b.ch
}
func (b *blockingBalancer) Close() error {
	close(b.ch)
	return nil
}

func TestDialWithBlockingBalancer(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	dialDone := make(chan struct{})
	go func() {
		DialContext(ctx, "Non-Existent.Server:80", WithBlock(), WithInsecure(), WithBalancer(newBlockingBalancer()))
		close(dialDone)
	}()
	cancel()
	<-dialDone
}

// securePerRPCCredentials always requires transport security.
type securePerRPCCredentials struct{}

func (c securePerRPCCredentials) GetRequestMetadata(ctx context.Context, uri ...string) (map[string]string, error) {
	return nil, nil
}

func (c securePerRPCCredentials) RequireTransportSecurity() bool {
	return true
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
	// security info on insecure connection
	if _, err := Dial("Non-Existent.Server:80", WithPerRPCCredentials(securePerRPCCredentials{}), WithBlock(), WithInsecure()); err != errTransportCredentialsMissing {
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

type testErr struct {
	temp bool
}

func (e *testErr) Error() string {
	return "test error"
}

func (e *testErr) Temporary() bool {
	return e.temp
}

var nonTemporaryError = &testErr{false}

func nonTemporaryErrorDialer(addr string, timeout time.Duration) (net.Conn, error) {
	return nil, nonTemporaryError
}

func TestDialWithBlockErrorOnNonTemporaryErrorDialer(t *testing.T) {
	ctx, _ := context.WithTimeout(context.Background(), 100*time.Millisecond)
	if _, err := DialContext(ctx, "", WithInsecure(), WithDialer(nonTemporaryErrorDialer), WithBlock(), FailOnNonTempDialError(true)); err != nonTemporaryError {
		t.Fatalf("Dial(%q) = %v, want %v", "", err, nonTemporaryError)
	}

	// Without FailOnNonTempDialError, gRPC will retry to connect, and dial should exit with time out error.
	if _, err := DialContext(ctx, "", WithInsecure(), WithDialer(nonTemporaryErrorDialer), WithBlock()); err != context.DeadlineExceeded {
		t.Fatalf("Dial(%q) = %v, want %v", "", err, context.DeadlineExceeded)
	}
}

// emptyBalancer returns an empty set of servers.
type emptyBalancer struct {
	ch chan []Address
}

func newEmptyBalancer() Balancer {
	return &emptyBalancer{ch: make(chan []Address, 1)}
}
func (b *emptyBalancer) Start(_ string, _ BalancerConfig) error {
	b.ch <- nil
	return nil
}
func (b *emptyBalancer) Up(_ Address) func(error) {
	return nil
}
func (b *emptyBalancer) Get(_ context.Context, _ BalancerGetOptions) (Address, func(), error) {
	return Address{}, nil, nil
}
func (b *emptyBalancer) Notify() <-chan []Address {
	return b.ch
}
func (b *emptyBalancer) Close() error {
	close(b.ch)
	return nil
}

func TestNonblockingDialWithEmptyBalancer(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	dialDone := make(chan struct{})
	go func() {
		conn, err := DialContext(ctx, "Non-Existent.Server:80", WithInsecure(), WithBalancer(newEmptyBalancer()))
		if err != nil {
			t.Fatalf("unexpected error dialing connection: %v", err)
		}
		conn.Close()
		close(dialDone)
	}()
	<-dialDone
	cancel()
}

func TestClientUpdatesParamsAfterGoAway(t *testing.T) {
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("Failed to listen. Err: %v", err)
	}
	defer lis.Close()
	addr := lis.Addr().String()
	s := NewServer()
	go s.Serve(lis)
	defer s.Stop()
	cc, err := Dial(addr, WithBlock(), WithInsecure(), WithKeepaliveParams(keepalive.ClientParameters{
		Time:                50 * time.Millisecond,
		Timeout:             1 * time.Millisecond,
		PermitWithoutStream: true,
	}))
	if err != nil {
		t.Fatalf("Dial(%s, _) = _, %v, want _, <nil>", addr, err)
	}
	defer cc.Close()
	time.Sleep(1 * time.Second)
	cc.mu.RLock()
	defer cc.mu.RUnlock()
	v := cc.mkp.Time
	if v < 100*time.Millisecond {
		t.Fatalf("cc.dopts.copts.Keepalive.Time = %v , want 100ms", v)
	}
}
