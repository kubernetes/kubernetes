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

import (
	"context"
	"reflect"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal/balancerload"
	"google.golang.org/grpc/internal/testutils"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/resolver/manual"
	testpb "google.golang.org/grpc/test/grpc_testing"
	"google.golang.org/grpc/testdata"
)

const testBalancerName = "testbalancer"

// testBalancer creates one subconn with the first address from resolved
// addresses.
//
// It's used to test options for NewSubConn are applies correctly.
type testBalancer struct {
	cc balancer.ClientConn
	sc balancer.SubConn

	newSubConnOptions balancer.NewSubConnOptions
	pickInfos         []balancer.PickInfo
	doneInfo          []balancer.DoneInfo
}

func (b *testBalancer) Build(cc balancer.ClientConn, opt balancer.BuildOptions) balancer.Balancer {
	b.cc = cc
	return b
}

func (*testBalancer) Name() string {
	return testBalancerName
}

func (b *testBalancer) HandleResolvedAddrs(addrs []resolver.Address, err error) {
	// Only create a subconn at the first time.
	if err == nil && b.sc == nil {
		b.sc, err = b.cc.NewSubConn(addrs, b.newSubConnOptions)
		if err != nil {
			grpclog.Errorf("testBalancer: failed to NewSubConn: %v", err)
			return
		}
		b.cc.UpdateState(balancer.State{ConnectivityState: connectivity.Connecting, Picker: &picker{sc: b.sc, bal: b}})
		b.sc.Connect()
	}
}

func (b *testBalancer) HandleSubConnStateChange(sc balancer.SubConn, s connectivity.State) {
	grpclog.Infof("testBalancer: HandleSubConnStateChange: %p, %v", sc, s)
	if b.sc != sc {
		grpclog.Infof("testBalancer: ignored state change because sc is not recognized")
		return
	}
	if s == connectivity.Shutdown {
		b.sc = nil
		return
	}

	switch s {
	case connectivity.Ready, connectivity.Idle:
		b.cc.UpdateState(balancer.State{ConnectivityState: s, Picker: &picker{sc: sc, bal: b}})
	case connectivity.Connecting:
		b.cc.UpdateState(balancer.State{ConnectivityState: s, Picker: &picker{err: balancer.ErrNoSubConnAvailable, bal: b}})
	case connectivity.TransientFailure:
		b.cc.UpdateState(balancer.State{ConnectivityState: s, Picker: &picker{err: balancer.ErrTransientFailure, bal: b}})
	}
}

func (b *testBalancer) Close() {
}

type picker struct {
	err error
	sc  balancer.SubConn
	bal *testBalancer
}

func (p *picker) Pick(info balancer.PickInfo) (balancer.PickResult, error) {
	if p.err != nil {
		return balancer.PickResult{}, p.err
	}
	info.Ctx = nil // Do not validate context.
	p.bal.pickInfos = append(p.bal.pickInfos, info)
	return balancer.PickResult{SubConn: p.sc, Done: func(d balancer.DoneInfo) { p.bal.doneInfo = append(p.bal.doneInfo, d) }}, nil
}

func (s) TestCredsBundleFromBalancer(t *testing.T) {
	balancer.Register(&testBalancer{
		newSubConnOptions: balancer.NewSubConnOptions{
			CredsBundle: &testCredsBundle{},
		},
	})
	te := newTest(t, env{name: "creds-bundle", network: "tcp", balancer: ""})
	te.tapHandle = authHandle
	te.customDialOptions = []grpc.DialOption{
		grpc.WithBalancerName(testBalancerName),
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

func (s) TestDoneInfo(t *testing.T) {
	for _, e := range listTestEnv() {
		testDoneInfo(t, e)
	}
}

func testDoneInfo(t *testing.T, e env) {
	te := newTest(t, e)
	b := &testBalancer{}
	balancer.Register(b)
	te.customDialOptions = []grpc.DialOption{
		grpc.WithBalancerName(testBalancerName),
	}
	te.userAgent = failAppUA
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	wantErr := detailedError
	if _, err := tc.EmptyCall(ctx, &testpb.Empty{}); !testutils.StatusErrEqual(err, wantErr) {
		t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want _, %v", err, wantErr)
	}
	if _, err := tc.UnaryCall(ctx, &testpb.SimpleRequest{}); err != nil {
		t.Fatalf("TestService.UnaryCall(%v, _, _, _) = _, %v; want _, <nil>", ctx, err)
	}

	if len(b.doneInfo) < 1 || !testutils.StatusErrEqual(b.doneInfo[0].Err, wantErr) {
		t.Fatalf("b.doneInfo = %v; want b.doneInfo[0].Err = %v", b.doneInfo, wantErr)
	}
	if len(b.doneInfo) < 2 || !reflect.DeepEqual(b.doneInfo[1].Trailer, testTrailerMetadata) {
		t.Fatalf("b.doneInfo = %v; want b.doneInfo[1].Trailer = %v", b.doneInfo, testTrailerMetadata)
	}
	if len(b.pickInfos) != len(b.doneInfo) {
		t.Fatalf("Got %d picks, but %d doneInfo, want equal amount", len(b.pickInfos), len(b.doneInfo))
	}
	// To test done() is always called, even if it's returned with a non-Ready
	// SubConn.
	//
	// Stop server and at the same time send RPCs. There are chances that picker
	// is not updated in time, causing a non-Ready SubConn to be returned.
	finished := make(chan struct{})
	go func() {
		for i := 0; i < 20; i++ {
			tc.UnaryCall(ctx, &testpb.SimpleRequest{})
		}
		close(finished)
	}()
	te.srv.Stop()
	<-finished
	if len(b.pickInfos) != len(b.doneInfo) {
		t.Fatalf("Got %d picks, %d doneInfo, want equal amount", len(b.pickInfos), len(b.doneInfo))
	}
}

const loadMDKey = "X-Endpoint-Load-Metrics-Bin"

type testLoadParser struct{}

func (*testLoadParser) Parse(md metadata.MD) interface{} {
	vs := md.Get(loadMDKey)
	if len(vs) == 0 {
		return nil
	}
	return vs[0]
}

func init() {
	balancerload.SetParser(&testLoadParser{})
}

func (s) TestDoneLoads(t *testing.T) {
	for _, e := range listTestEnv() {
		testDoneLoads(t, e)
	}
}

func testDoneLoads(t *testing.T, e env) {
	b := &testBalancer{}
	balancer.Register(b)

	const testLoad = "test-load-,-should-be-orca"

	ss := &stubServer{
		emptyCall: func(ctx context.Context, in *testpb.Empty) (*testpb.Empty, error) {
			grpc.SetTrailer(ctx, metadata.Pairs(loadMDKey, testLoad))
			return &testpb.Empty{}, nil
		},
	}
	if err := ss.Start(nil, grpc.WithBalancerName(testBalancerName)); err != nil {
		t.Fatalf("error starting testing server: %v", err)
	}
	defer ss.Stop()

	tc := testpb.NewTestServiceClient(ss.cc)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := tc.EmptyCall(ctx, &testpb.Empty{}); err != nil {
		t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want _, %v", err, nil)
	}

	piWant := []balancer.PickInfo{
		{FullMethodName: "/grpc.testing.TestService/EmptyCall"},
	}
	if !reflect.DeepEqual(b.pickInfos, piWant) {
		t.Fatalf("b.pickInfos = %v; want %v", b.pickInfos, piWant)
	}

	if len(b.doneInfo) < 1 {
		t.Fatalf("b.doneInfo = %v, want length 1", b.doneInfo)
	}
	gotLoad, _ := b.doneInfo[0].ServerLoad.(string)
	if gotLoad != testLoad {
		t.Fatalf("b.doneInfo[0].ServerLoad = %v; want = %v", b.doneInfo[0].ServerLoad, testLoad)
	}
}

const testBalancerKeepAddressesName = "testbalancer-keepingaddresses"

// testBalancerKeepAddresses keeps the addresses in the builder instead of
// creating SubConns.
//
// It's used to test the addresses balancer gets are correct.
type testBalancerKeepAddresses struct {
	addrsChan chan []resolver.Address
}

func newTestBalancerKeepAddresses() *testBalancerKeepAddresses {
	return &testBalancerKeepAddresses{
		addrsChan: make(chan []resolver.Address, 10),
	}
}

func (b *testBalancerKeepAddresses) Build(cc balancer.ClientConn, opt balancer.BuildOptions) balancer.Balancer {
	return b
}

func (*testBalancerKeepAddresses) Name() string {
	return testBalancerKeepAddressesName
}

func (b *testBalancerKeepAddresses) HandleResolvedAddrs(addrs []resolver.Address, err error) {
	b.addrsChan <- addrs
}

func (testBalancerKeepAddresses) HandleSubConnStateChange(sc balancer.SubConn, s connectivity.State) {
	panic("not used")
}

func (testBalancerKeepAddresses) Close() {
}

// Make sure that non-grpclb balancers don't get grpclb addresses even if name
// resolver sends them
func (s) TestNonGRPCLBBalancerGetsNoGRPCLBAddress(t *testing.T) {
	r, rcleanup := manual.GenerateAndRegisterManualResolver()
	defer rcleanup()

	b := newTestBalancerKeepAddresses()
	balancer.Register(b)

	cc, err := grpc.Dial(r.Scheme()+":///test.server", grpc.WithInsecure(),
		grpc.WithBalancerName(b.Name()))
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	defer cc.Close()

	grpclbAddresses := []resolver.Address{{
		Addr:       "grpc.lb.com",
		Type:       resolver.GRPCLB,
		ServerName: "grpc.lb.com",
	}}

	nonGRPCLBAddresses := []resolver.Address{{
		Addr: "localhost",
		Type: resolver.Backend,
	}}

	r.UpdateState(resolver.State{
		Addresses: nonGRPCLBAddresses,
	})
	if got := <-b.addrsChan; !reflect.DeepEqual(got, nonGRPCLBAddresses) {
		t.Fatalf("With only backend addresses, balancer got addresses %v, want %v", got, nonGRPCLBAddresses)
	}

	r.UpdateState(resolver.State{
		Addresses: grpclbAddresses,
	})
	if got := <-b.addrsChan; len(got) != 0 {
		t.Fatalf("With only grpclb addresses, balancer got addresses %v, want empty", got)
	}

	r.UpdateState(resolver.State{
		Addresses: append(grpclbAddresses, nonGRPCLBAddresses...),
	})
	if got := <-b.addrsChan; !reflect.DeepEqual(got, nonGRPCLBAddresses) {
		t.Fatalf("With both backend and grpclb addresses, balancer got addresses %v, want %v", got, nonGRPCLBAddresses)
	}
}
