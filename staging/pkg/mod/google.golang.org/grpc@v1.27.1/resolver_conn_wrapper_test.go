/*
 *
 * Copyright 2017 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package grpc

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/resolver/manual"
	"google.golang.org/grpc/serviceconfig"
	"google.golang.org/grpc/status"
)

func (s) TestParseTarget(t *testing.T) {
	for _, test := range []resolver.Target{
		{Scheme: "dns", Authority: "", Endpoint: "google.com"},
		{Scheme: "dns", Authority: "a.server.com", Endpoint: "google.com"},
		{Scheme: "dns", Authority: "a.server.com", Endpoint: "google.com/?a=b"},
		{Scheme: "passthrough", Authority: "", Endpoint: "/unix/socket/address"},
	} {
		str := test.Scheme + "://" + test.Authority + "/" + test.Endpoint
		got := parseTarget(str)
		if got != test {
			t.Errorf("parseTarget(%q) = %+v, want %+v", str, got, test)
		}
	}
}

func (s) TestParseTargetString(t *testing.T) {
	for _, test := range []struct {
		targetStr string
		want      resolver.Target
	}{
		{targetStr: "", want: resolver.Target{Scheme: "", Authority: "", Endpoint: ""}},
		{targetStr: ":///", want: resolver.Target{Scheme: "", Authority: "", Endpoint: ""}},
		{targetStr: "a:///", want: resolver.Target{Scheme: "a", Authority: "", Endpoint: ""}},
		{targetStr: "://a/", want: resolver.Target{Scheme: "", Authority: "a", Endpoint: ""}},
		{targetStr: ":///a", want: resolver.Target{Scheme: "", Authority: "", Endpoint: "a"}},
		{targetStr: "a://b/", want: resolver.Target{Scheme: "a", Authority: "b", Endpoint: ""}},
		{targetStr: "a:///b", want: resolver.Target{Scheme: "a", Authority: "", Endpoint: "b"}},
		{targetStr: "://a/b", want: resolver.Target{Scheme: "", Authority: "a", Endpoint: "b"}},
		{targetStr: "a://b/c", want: resolver.Target{Scheme: "a", Authority: "b", Endpoint: "c"}},
		{targetStr: "dns:///google.com", want: resolver.Target{Scheme: "dns", Authority: "", Endpoint: "google.com"}},
		{targetStr: "dns://a.server.com/google.com", want: resolver.Target{Scheme: "dns", Authority: "a.server.com", Endpoint: "google.com"}},
		{targetStr: "dns://a.server.com/google.com/?a=b", want: resolver.Target{Scheme: "dns", Authority: "a.server.com", Endpoint: "google.com/?a=b"}},

		{targetStr: "/", want: resolver.Target{Scheme: "", Authority: "", Endpoint: "/"}},
		{targetStr: "google.com", want: resolver.Target{Scheme: "", Authority: "", Endpoint: "google.com"}},
		{targetStr: "google.com/?a=b", want: resolver.Target{Scheme: "", Authority: "", Endpoint: "google.com/?a=b"}},
		{targetStr: "/unix/socket/address", want: resolver.Target{Scheme: "", Authority: "", Endpoint: "/unix/socket/address"}},

		// If we can only parse part of the target.
		{targetStr: "://", want: resolver.Target{Scheme: "", Authority: "", Endpoint: "://"}},
		{targetStr: "unix://domain", want: resolver.Target{Scheme: "", Authority: "", Endpoint: "unix://domain"}},
		{targetStr: "a:b", want: resolver.Target{Scheme: "", Authority: "", Endpoint: "a:b"}},
		{targetStr: "a/b", want: resolver.Target{Scheme: "", Authority: "", Endpoint: "a/b"}},
		{targetStr: "a:/b", want: resolver.Target{Scheme: "", Authority: "", Endpoint: "a:/b"}},
		{targetStr: "a//b", want: resolver.Target{Scheme: "", Authority: "", Endpoint: "a//b"}},
		{targetStr: "a://b", want: resolver.Target{Scheme: "", Authority: "", Endpoint: "a://b"}},
	} {
		got := parseTarget(test.targetStr)
		if got != test.want {
			t.Errorf("parseTarget(%q) = %+v, want %+v", test.targetStr, got, test.want)
		}
	}
}

// The target string with unknown scheme should be kept unchanged and passed to
// the dialer.
func (s) TestDialParseTargetUnknownScheme(t *testing.T) {
	for _, test := range []struct {
		targetStr string
		want      string
	}{
		{"/unix/socket/address", "/unix/socket/address"},

		// Special test for "unix:///".
		{"unix:///unix/socket/address", "unix:///unix/socket/address"},

		// For known scheme.
		{"passthrough://a.server.com/google.com", "google.com"},
	} {
		dialStrCh := make(chan string, 1)
		cc, err := Dial(test.targetStr, WithInsecure(), WithDialer(func(addr string, _ time.Duration) (net.Conn, error) {
			select {
			case dialStrCh <- addr:
			default:
			}
			return nil, fmt.Errorf("test dialer, always error")
		}))
		if err != nil {
			t.Fatalf("Failed to create ClientConn: %v", err)
		}
		got := <-dialStrCh
		cc.Close()
		if got != test.want {
			t.Errorf("Dial(%q), dialer got %q, want %q", test.targetStr, got, test.want)
		}
	}
}

func testResolverErrorPolling(t *testing.T, badUpdate func(*manual.Resolver), goodUpdate func(*manual.Resolver), dopts ...DialOption) {
	boIter := make(chan int)
	resolverBackoff := func(v int) time.Duration {
		boIter <- v
		return 0
	}

	r, rcleanup := manual.GenerateAndRegisterManualResolver()
	defer rcleanup()
	rn := make(chan struct{})
	defer func() { close(rn) }()
	r.ResolveNowCallback = func(resolver.ResolveNowOptions) { rn <- struct{}{} }

	defaultDialOptions := []DialOption{
		WithInsecure(),
		withResolveNowBackoff(resolverBackoff),
	}
	cc, err := Dial(r.Scheme()+":///test.server", append(defaultDialOptions, dopts...)...)
	if err != nil {
		t.Fatalf("Dial(_, _) = _, %v; want _, nil", err)
	}
	defer cc.Close()
	badUpdate(r)

	panicAfter := time.AfterFunc(5*time.Second, func() { panic("timed out polling resolver") })
	defer panicAfter.Stop()

	// Ensure ResolveNow is called, then Backoff with the right parameter, several times
	for i := 0; i < 7; i++ {
		<-rn
		if v := <-boIter; v != i {
			t.Errorf("Backoff call %v uses value %v", i, v)
		}
	}

	// UpdateState will block if ResolveNow is being called (which blocks on
	// rn), so call it in a goroutine.
	goodUpdate(r)

	// Wait awhile to ensure ResolveNow and Backoff stop being called when the
	// state is OK (i.e. polling was cancelled).
	for {
		t := time.NewTimer(50 * time.Millisecond)
		select {
		case <-rn:
			// ClientConn is still calling ResolveNow
			<-boIter
			time.Sleep(5 * time.Millisecond)
			continue
		case <-t.C:
			// ClientConn stopped calling ResolveNow; success
		}
		break
	}
}

const happyBalancerName = "happy balancer"

func init() {
	// Register a balancer that never returns an error from
	// UpdateClientConnState, and doesn't do anything else either.
	fb := &funcBalancer{
		updateClientConnState: func(s balancer.ClientConnState) error {
			return nil
		},
	}
	balancer.Register(&funcBalancerBuilder{name: happyBalancerName, instance: fb})
}

// TestResolverErrorPolling injects resolver errors and verifies ResolveNow is
// called with the appropriate backoff strategy being consulted between
// ResolveNow calls.
func (s) TestResolverErrorPolling(t *testing.T) {
	testResolverErrorPolling(t, func(r *manual.Resolver) {
		r.CC.ReportError(errors.New("res err"))
	}, func(r *manual.Resolver) {
		// UpdateState will block if ResolveNow is being called (which blocks on
		// rn), so call it in a goroutine.
		go r.CC.UpdateState(resolver.State{})
	},
		WithDefaultServiceConfig(fmt.Sprintf(`{ "loadBalancingConfig": [{"%v": {}}] }`, happyBalancerName)))
}

// TestServiceConfigErrorPolling injects a service config error and verifies
// ResolveNow is called with the appropriate backoff strategy being consulted
// between ResolveNow calls.
func (s) TestServiceConfigErrorPolling(t *testing.T) {
	testResolverErrorPolling(t, func(r *manual.Resolver) {
		badsc := r.CC.ParseServiceConfig("bad config")
		r.UpdateState(resolver.State{ServiceConfig: badsc})
	}, func(r *manual.Resolver) {
		// UpdateState will block if ResolveNow is being called (which blocks on
		// rn), so call it in a goroutine.
		go r.CC.UpdateState(resolver.State{})
	},
		WithDefaultServiceConfig(fmt.Sprintf(`{ "loadBalancingConfig": [{"%v": {}}] }`, happyBalancerName)))
}

// TestResolverErrorInBuild makes the resolver.Builder call into the ClientConn
// during the Build call. We use two separate mutexes in the code which make
// sure there is no data race in this code path, and also that there is no
// deadlock.
func (s) TestResolverErrorInBuild(t *testing.T) {
	r, rcleanup := manual.GenerateAndRegisterManualResolver()
	defer rcleanup()
	r.InitialState(resolver.State{ServiceConfig: &serviceconfig.ParseResult{Err: errors.New("resolver build err")}})

	cc, err := Dial(r.Scheme()+":///test.server", WithInsecure())
	if err != nil {
		t.Fatalf("Dial(_, _) = _, %v; want _, nil", err)
	}
	defer cc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var dummy int
	const wantMsg = "error parsing service config"
	const wantCode = codes.Unavailable
	if err := cc.Invoke(ctx, "/foo/bar", &dummy, &dummy); status.Code(err) != wantCode || !strings.Contains(status.Convert(err).Message(), wantMsg) {
		t.Fatalf("cc.Invoke(_, _, _, _) = %v; want status.Code()==%v, status.Message() contains %q", err, wantCode, wantMsg)
	}
}

func (s) TestServiceConfigErrorRPC(t *testing.T) {
	r, rcleanup := manual.GenerateAndRegisterManualResolver()
	defer rcleanup()

	cc, err := Dial(r.Scheme()+":///test.server", WithInsecure())
	if err != nil {
		t.Fatalf("Dial(_, _) = _, %v; want _, nil", err)
	}
	defer cc.Close()
	badsc := r.CC.ParseServiceConfig("bad config")
	r.UpdateState(resolver.State{ServiceConfig: badsc})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var dummy int
	const wantMsg = "error parsing service config"
	const wantCode = codes.Unavailable
	if err := cc.Invoke(ctx, "/foo/bar", &dummy, &dummy); status.Code(err) != wantCode || !strings.Contains(status.Convert(err).Message(), wantMsg) {
		t.Fatalf("cc.Invoke(_, _, _, _) = %v; want status.Code()==%v, status.Message() contains %q", err, wantCode, wantMsg)
	}
}
