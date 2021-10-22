/*
 *
 * Copyright 2019 gRPC authors.
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

package resolver

import (
	"context"
	"errors"
	"fmt"
	"net"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/serviceconfig"
	xdsinternal "google.golang.org/grpc/xds/internal"
	_ "google.golang.org/grpc/xds/internal/balancer/cdsbalancer" // To parse LB config
	xdsclient "google.golang.org/grpc/xds/internal/client"
	"google.golang.org/grpc/xds/internal/client/bootstrap"
	"google.golang.org/grpc/xds/internal/testutils"
	"google.golang.org/grpc/xds/internal/testutils/fakeclient"

	corepb "github.com/envoyproxy/go-control-plane/envoy/api/v2/core"
)

const (
	targetStr    = "target"
	cluster      = "cluster"
	balancerName = "dummyBalancer"
)

var (
	validConfig = bootstrap.Config{
		BalancerName: balancerName,
		Creds:        grpc.WithInsecure(),
		NodeProto:    &corepb.Node{},
	}
	target = resolver.Target{Endpoint: targetStr}
)

// testClientConn is a fake implemetation of resolver.ClientConn. All is does
// is to store the state received from the resolver locally and signal that
// event through a channel.
type testClientConn struct {
	resolver.ClientConn
	stateCh *testutils.Channel
	errorCh *testutils.Channel
}

func (t *testClientConn) UpdateState(s resolver.State) {
	t.stateCh.Send(s)
}

func (t *testClientConn) ReportError(err error) {
	t.errorCh.Send(err)
}

func (t *testClientConn) ParseServiceConfig(jsonSC string) *serviceconfig.ParseResult {
	return internal.ParseServiceConfigForTesting.(func(string) *serviceconfig.ParseResult)(jsonSC)
}

func newTestClientConn() *testClientConn {
	return &testClientConn{
		stateCh: testutils.NewChannel(),
		errorCh: testutils.NewChannel(),
	}
}

func getXDSClientMakerFunc(wantOpts xdsclient.Options) func(xdsclient.Options) (xdsClientInterface, error) {
	return func(gotOpts xdsclient.Options) (xdsClientInterface, error) {
		if gotOpts.Config.BalancerName != wantOpts.Config.BalancerName {
			return nil, fmt.Errorf("got balancerName: %s, want: %s", gotOpts.Config.BalancerName, wantOpts.Config.BalancerName)
		}
		// We cannot compare two DialOption objects to see if they are equal
		// because each of these is a function pointer. So, the only thing we
		// can do here is to check if the got option is nil or not based on
		// what the want option is. We should be able to do extensive
		// credential testing in e2e tests.
		if (gotOpts.Config.Creds != nil) != (wantOpts.Config.Creds != nil) {
			return nil, fmt.Errorf("got len(creds): %s, want: %s", gotOpts.Config.Creds, wantOpts.Config.Creds)
		}
		if len(gotOpts.DialOpts) != len(wantOpts.DialOpts) {
			return nil, fmt.Errorf("got len(DialOpts): %v, want: %v", len(gotOpts.DialOpts), len(wantOpts.DialOpts))
		}
		return fakeclient.NewClient(), nil
	}
}

func errorDialer(_ context.Context, _ string) (net.Conn, error) {
	return nil, errors.New("dial error")
}

// TestResolverBuilder tests the xdsResolverBuilder's Build method with
// different parameters.
func TestResolverBuilder(t *testing.T) {
	tests := []struct {
		name          string
		rbo           resolver.BuildOptions
		config        bootstrap.Config
		xdsClientFunc func(xdsclient.Options) (xdsClientInterface, error)
		wantErr       bool
	}{
		{
			name:    "empty-config",
			rbo:     resolver.BuildOptions{},
			config:  bootstrap.Config{},
			wantErr: true,
		},
		{
			name: "no-balancer-name-in-config",
			rbo:  resolver.BuildOptions{},
			config: bootstrap.Config{
				Creds:     grpc.WithInsecure(),
				NodeProto: &corepb.Node{},
			},
			wantErr: true,
		},
		{
			name: "no-creds-in-config",
			rbo:  resolver.BuildOptions{},
			config: bootstrap.Config{
				BalancerName: balancerName,
				NodeProto:    &corepb.Node{},
			},
			xdsClientFunc: getXDSClientMakerFunc(xdsclient.Options{Config: validConfig}),
			wantErr:       false,
		},
		{
			name:   "error-dialer-in-rbo",
			rbo:    resolver.BuildOptions{Dialer: errorDialer},
			config: validConfig,
			xdsClientFunc: getXDSClientMakerFunc(xdsclient.Options{
				Config:   validConfig,
				DialOpts: []grpc.DialOption{grpc.WithContextDialer(errorDialer)},
			}),
			wantErr: false,
		},
		{
			name:          "simple-good",
			rbo:           resolver.BuildOptions{},
			config:        validConfig,
			xdsClientFunc: getXDSClientMakerFunc(xdsclient.Options{Config: validConfig}),
			wantErr:       false,
		},
		{
			name:   "newXDSClient-throws-error",
			rbo:    resolver.BuildOptions{},
			config: validConfig,
			xdsClientFunc: func(_ xdsclient.Options) (xdsClientInterface, error) {
				return nil, errors.New("newXDSClient-throws-error")
			},
			wantErr: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Fake out the bootstrap process by providing our own config.
			oldConfigMaker := newXDSConfig
			newXDSConfig = func() (*bootstrap.Config, error) {
				if test.config.BalancerName == "" {
					return nil, fmt.Errorf("no balancer name found in config")
				}
				return &test.config, nil
			}
			// Fake out the xdsClient creation process by providing a fake.
			oldClientMaker := newXDSClient
			newXDSClient = test.xdsClientFunc
			defer func() {
				newXDSConfig = oldConfigMaker
				newXDSClient = oldClientMaker
			}()

			builder := resolver.Get(xdsScheme)
			if builder == nil {
				t.Fatalf("resolver.Get(%v) returned nil", xdsScheme)
			}

			r, err := builder.Build(target, newTestClientConn(), test.rbo)
			if (err != nil) != test.wantErr {
				t.Fatalf("builder.Build(%v) returned err: %v, wantErr: %v", target, err, test.wantErr)
			}
			if err != nil {
				// This is the case where we expect an error and got it.
				return
			}
			r.Close()
		})
	}
}

type setupOpts struct {
	config        *bootstrap.Config
	xdsClientFunc func(xdsclient.Options) (xdsClientInterface, error)
}

func testSetup(t *testing.T, opts setupOpts) (*xdsResolver, *testClientConn, func()) {
	t.Helper()

	oldConfigMaker := newXDSConfig
	newXDSConfig = func() (*bootstrap.Config, error) { return opts.config, nil }
	oldClientMaker := newXDSClient
	newXDSClient = opts.xdsClientFunc
	cancel := func() {
		newXDSConfig = oldConfigMaker
		newXDSClient = oldClientMaker
	}

	builder := resolver.Get(xdsScheme)
	if builder == nil {
		t.Fatalf("resolver.Get(%v) returned nil", xdsScheme)
	}

	tcc := newTestClientConn()
	r, err := builder.Build(target, tcc, resolver.BuildOptions{})
	if err != nil {
		t.Fatalf("builder.Build(%v) returned err: %v", target, err)
	}
	return r.(*xdsResolver), tcc, cancel
}

// waitForWatchService waits for the WatchService method to be called on the
// xdsClient within a reasonable amount of time, and also verifies that the
// watch is called with the expected target.
func waitForWatchService(t *testing.T, xdsC *fakeclient.Client, wantTarget string) {
	t.Helper()

	gotTarget, err := xdsC.WaitForWatchService()
	if err != nil {
		t.Fatalf("xdsClient.WatchService failed with error: %v", err)
	}
	if gotTarget != wantTarget {
		t.Fatalf("xdsClient.WatchService() called with target: %v, want %v", gotTarget, wantTarget)
	}
}

// TestXDSResolverWatchCallbackAfterClose tests the case where a service update
// from the underlying xdsClient is received after the resolver is closed.
func TestXDSResolverWatchCallbackAfterClose(t *testing.T) {
	xdsC := fakeclient.NewClient()
	xdsR, tcc, cancel := testSetup(t, setupOpts{
		config:        &validConfig,
		xdsClientFunc: func(_ xdsclient.Options) (xdsClientInterface, error) { return xdsC, nil },
	})
	defer cancel()

	waitForWatchService(t, xdsC, targetStr)

	// Call the watchAPI callback after closing the resolver, and make sure no
	// update is triggerred on the ClientConn.
	xdsR.Close()
	xdsC.InvokeWatchServiceCallback(cluster, nil)
	if gotVal, gotErr := tcc.stateCh.Receive(); gotErr != testutils.ErrRecvTimeout {
		t.Fatalf("ClientConn.UpdateState called after xdsResolver is closed: %v", gotVal)
	}
}

// TestXDSResolverBadServiceUpdate tests the case the xdsClient returns a bad
// service update.
func TestXDSResolverBadServiceUpdate(t *testing.T) {
	xdsC := fakeclient.NewClient()
	xdsR, tcc, cancel := testSetup(t, setupOpts{
		config:        &validConfig,
		xdsClientFunc: func(_ xdsclient.Options) (xdsClientInterface, error) { return xdsC, nil },
	})
	defer func() {
		cancel()
		xdsR.Close()
	}()

	waitForWatchService(t, xdsC, targetStr)

	// Invoke the watchAPI callback with a bad service update and wait for the
	// ReportError method to be called on the ClientConn.
	suErr := errors.New("bad serviceupdate")
	xdsC.InvokeWatchServiceCallback("", suErr)
	if gotErrVal, gotErr := tcc.errorCh.Receive(); gotErr != nil || gotErrVal != suErr {
		t.Fatalf("ClientConn.ReportError() received %v, want %v", gotErrVal, suErr)
	}
}

// TestXDSResolverGoodServiceUpdate tests the happy case where the resolver
// gets a good service update from the xdsClient.
func TestXDSResolverGoodServiceUpdate(t *testing.T) {
	xdsC := fakeclient.NewClient()
	xdsR, tcc, cancel := testSetup(t, setupOpts{
		config:        &validConfig,
		xdsClientFunc: func(_ xdsclient.Options) (xdsClientInterface, error) { return xdsC, nil },
	})
	defer func() {
		cancel()
		xdsR.Close()
	}()

	waitForWatchService(t, xdsC, targetStr)

	// Invoke the watchAPI callback with a good service update and wait for the
	// UpdateState method to be called on the ClientConn.
	xdsC.InvokeWatchServiceCallback(cluster, nil)
	gotState, err := tcc.stateCh.Receive()
	if err != nil {
		t.Fatalf("ClientConn.UpdateState returned error: %v", err)
	}
	rState := gotState.(resolver.State)
	if gotClient := rState.Attributes.Value(xdsinternal.XDSClientID); gotClient != xdsC {
		t.Fatalf("ClientConn.UpdateState got xdsClient: %v, want %v", gotClient, xdsC)
	}
	if err := rState.ServiceConfig.Err; err != nil {
		t.Fatalf("ClientConn.UpdateState received error in service config: %v", rState.ServiceConfig.Err)
	}
}
