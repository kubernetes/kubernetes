/*
 *
 * Copyright 2018 gRPC authors.
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

package test

// TODO(https://github.com/grpc/grpc-go/issues/2330): move all creds related
// tests to this file.

import (
	"context"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	testpb "google.golang.org/grpc/test/grpc_testing"
	"google.golang.org/grpc/testdata"
)

const (
	bundlePerRPCOnly = "perRPCOnly"
	bundleTLSOnly    = "tlsOnly"
)

type testCredsBundle struct {
	t    *testing.T
	mode string
}

func (c *testCredsBundle) TransportCredentials() credentials.TransportCredentials {
	if c.mode == bundlePerRPCOnly {
		return nil
	}

	creds, err := credentials.NewClientTLSFromFile(testdata.Path("ca.pem"), "x.test.youtube.com")
	if err != nil {
		c.t.Logf("Failed to load credentials: %v", err)
		return nil
	}
	return creds
}

func (c *testCredsBundle) PerRPCCredentials() credentials.PerRPCCredentials {
	if c.mode == bundleTLSOnly {
		return nil
	}
	return testPerRPCCredentials{}
}

func (c *testCredsBundle) NewWithMode(mode string) (credentials.Bundle, error) {
	return &testCredsBundle{mode: mode}, nil
}

func (s) TestCredsBundleBoth(t *testing.T) {
	te := newTest(t, env{name: "creds-bundle", network: "tcp", balancer: "v1", security: "empty"})
	te.tapHandle = authHandle
	te.customDialOptions = []grpc.DialOption{
		grpc.WithCredentialsBundle(&testCredsBundle{t: t}),
	}
	creds, err := credentials.NewServerTLSFromFile(testdata.Path("server1.pem"), testdata.Path("server1.key"))
	if err != nil {
		t.Fatalf("Failed to generate credentials %v", err)
	}
	te.customServerOptions = []grpc.ServerOption{
		grpc.Creds(creds),
	}
	te.startServer(&testServer{})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}); err != nil {
		t.Fatalf("Test failed. Reason: %v", err)
	}
}

func (s) TestCredsBundleTransportCredentials(t *testing.T) {
	te := newTest(t, env{name: "creds-bundle", network: "tcp", balancer: "v1", security: "empty"})
	te.customDialOptions = []grpc.DialOption{
		grpc.WithCredentialsBundle(&testCredsBundle{t: t, mode: bundleTLSOnly}),
	}
	creds, err := credentials.NewServerTLSFromFile(testdata.Path("server1.pem"), testdata.Path("server1.key"))
	if err != nil {
		t.Fatalf("Failed to generate credentials %v", err)
	}
	te.customServerOptions = []grpc.ServerOption{
		grpc.Creds(creds),
	}
	te.startServer(&testServer{})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}); err != nil {
		t.Fatalf("Test failed. Reason: %v", err)
	}
}

func (s) TestCredsBundlePerRPCCredentials(t *testing.T) {
	te := newTest(t, env{name: "creds-bundle", network: "tcp", balancer: "v1", security: "empty"})
	te.tapHandle = authHandle
	te.customDialOptions = []grpc.DialOption{
		grpc.WithCredentialsBundle(&testCredsBundle{t: t, mode: bundlePerRPCOnly}),
	}
	te.startServer(&testServer{})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}); err != nil {
		t.Fatalf("Test failed. Reason: %v", err)
	}
}
