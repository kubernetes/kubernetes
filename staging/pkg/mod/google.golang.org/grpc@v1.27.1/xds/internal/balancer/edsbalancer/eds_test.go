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

package edsbalancer

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"testing"

	corepb "github.com/envoyproxy/go-control-plane/envoy/api/v2/core"
	"github.com/golang/protobuf/jsonpb"
	wrapperspb "github.com/golang/protobuf/ptypes/wrappers"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/internal/grpctest"
	"google.golang.org/grpc/internal/leakcheck"
	scpb "google.golang.org/grpc/internal/proto/grpc_service_config"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/serviceconfig"
	"google.golang.org/grpc/xds/internal/balancer/lrs"
	xdsclient "google.golang.org/grpc/xds/internal/client"
	"google.golang.org/grpc/xds/internal/client/bootstrap"
	"google.golang.org/grpc/xds/internal/testutils"
	"google.golang.org/grpc/xds/internal/testutils/fakeclient"
)

func init() {
	balancer.Register(&edsBalancerBuilder{})

	bootstrapConfigNew = func() (*bootstrap.Config, error) {
		return &bootstrap.Config{
			BalancerName: testBalancerNameFooBar,
			Creds:        grpc.WithInsecure(),
			NodeProto:    &corepb.Node{},
		}, nil
	}
}

type s struct{}

func (s) Teardown(t *testing.T) {
	leakcheck.Check(t)
}

func Test(t *testing.T) {
	grpctest.RunSubTests(t, s{})
}

const testBalancerNameFooBar = "foo.bar"

func newNoopTestClientConn() *noopTestClientConn {
	return &noopTestClientConn{}
}

// noopTestClientConn is used in EDS balancer config update tests that only
// cover the config update handling, but not SubConn/load-balancing.
type noopTestClientConn struct {
	balancer.ClientConn
}

func (t *noopTestClientConn) NewSubConn(addrs []resolver.Address, opts balancer.NewSubConnOptions) (balancer.SubConn, error) {
	return nil, nil
}

func (noopTestClientConn) Target() string { return testServiceName }

type scStateChange struct {
	sc    balancer.SubConn
	state connectivity.State
}

type fakeEDSBalancer struct {
	cc                 balancer.ClientConn
	childPolicy        *testutils.Channel
	subconnStateChange *testutils.Channel
	loadStore          lrs.Store
}

func (f *fakeEDSBalancer) HandleSubConnStateChange(sc balancer.SubConn, state connectivity.State) {
	f.subconnStateChange.Send(&scStateChange{sc: sc, state: state})
}

func (f *fakeEDSBalancer) HandleChildPolicy(name string, config json.RawMessage) {
	f.childPolicy.Send(&loadBalancingConfig{Name: name, Config: config})
}

func (f *fakeEDSBalancer) Close()                                         {}
func (f *fakeEDSBalancer) HandleEDSResponse(edsResp *xdsclient.EDSUpdate) {}

func (f *fakeEDSBalancer) waitForChildPolicy(wantPolicy *loadBalancingConfig) error {
	val, err := f.childPolicy.Receive()
	if err != nil {
		return fmt.Errorf("error waiting for childPolicy: %v", err)
	}
	gotPolicy := val.(*loadBalancingConfig)
	if !reflect.DeepEqual(gotPolicy, wantPolicy) {
		return fmt.Errorf("got childPolicy %v, want %v", gotPolicy, wantPolicy)
	}
	return nil
}

func (f *fakeEDSBalancer) waitForSubConnStateChange(wantState *scStateChange) error {
	val, err := f.subconnStateChange.Receive()
	if err != nil {
		return fmt.Errorf("error waiting for subconnStateChange: %v", err)
	}
	gotState := val.(*scStateChange)
	if !reflect.DeepEqual(gotState, wantState) {
		return fmt.Errorf("got subconnStateChange %v, want %v", gotState, wantState)
	}
	return nil
}

func newFakeEDSBalancer(cc balancer.ClientConn, loadStore lrs.Store) edsBalancerImplInterface {
	return &fakeEDSBalancer{
		cc:                 cc,
		childPolicy:        testutils.NewChannelWithSize(10),
		subconnStateChange: testutils.NewChannelWithSize(10),
		loadStore:          loadStore,
	}
}

type fakeSubConn struct{}

func (*fakeSubConn) UpdateAddresses([]resolver.Address) { panic("implement me") }
func (*fakeSubConn) Connect()                           { panic("implement me") }

// waitForNewXDSClientWithEDSWatch makes sure that a new xdsClient is created
// with the provided name. It also make sure that the newly created client
// registers an eds watcher.
func waitForNewXDSClientWithEDSWatch(t *testing.T, ch *testutils.Channel, wantName string) *fakeclient.Client {
	t.Helper()

	val, err := ch.Receive()
	if err != nil {
		t.Fatalf("error when waiting for a new xds client: %v", err)
		return nil
	}
	xdsC := val.(*fakeclient.Client)
	if xdsC.Name() != wantName {
		t.Fatalf("xdsClient created to balancer: %v, want %v", xdsC.Name(), wantName)
		return nil
	}
	_, err = xdsC.WaitForWatchEDS()
	if err != nil {
		t.Fatalf("xdsClient.WatchEDS failed with error: %v", err)
		return nil
	}
	return xdsC
}

// waitForNewEDSLB makes sure that a new edsLB is created by the top-level
// edsBalancer.
func waitForNewEDSLB(t *testing.T, ch *testutils.Channel) *fakeEDSBalancer {
	t.Helper()

	val, err := ch.Receive()
	if err != nil {
		t.Fatalf("error when waiting for a new edsLB: %v", err)
		return nil
	}
	return val.(*fakeEDSBalancer)
}

// setup overrides the functions which are used to create the xdsClient and the
// edsLB, creates fake version of them and makes them available on the provided
// channels. The returned cancel function should be called by the test for
// cleanup.
func setup(edsLBCh *testutils.Channel, xdsClientCh *testutils.Channel) func() {
	origNewEDSBalancer := newEDSBalancer
	newEDSBalancer = func(cc balancer.ClientConn, loadStore lrs.Store) edsBalancerImplInterface {
		edsLB := newFakeEDSBalancer(cc, loadStore)
		defer func() { edsLBCh.Send(edsLB) }()
		return edsLB
	}

	origXdsClientNew := xdsclientNew
	xdsclientNew = func(opts xdsclient.Options) (xdsClientInterface, error) {
		xdsC := fakeclient.NewClientWithName(opts.Config.BalancerName)
		defer func() { xdsClientCh.Send(xdsC) }()
		return xdsC, nil
	}
	return func() {
		newEDSBalancer = origNewEDSBalancer
		xdsclientNew = origXdsClientNew
	}
}

// TestXDSConfigBalancerNameUpdate verifies different scenarios where the
// balancer name in the lbConfig is updated.
//
// The test does the following:
// * Builds a new xds balancer.
// * Repeatedly pushes new ClientConnState which specifies different
//   balancerName in the lbConfig. We expect xdsClient objects to created
//   whenever the balancerName changes.
func (s) TestXDSConfigBalancerNameUpdate(t *testing.T) {
	oldBootstrapConfigNew := bootstrapConfigNew
	bootstrapConfigNew = func() (*bootstrap.Config, error) {
		// Return an error from bootstrap, so the eds balancer will use
		// BalancerName from the config.
		//
		// TODO: remove this when deleting BalancerName from config.
		return nil, fmt.Errorf("no bootstrap available")
	}
	defer func() { bootstrapConfigNew = oldBootstrapConfigNew }()
	edsLBCh := testutils.NewChannel()
	xdsClientCh := testutils.NewChannel()
	cancel := setup(edsLBCh, xdsClientCh)
	defer cancel()

	builder := balancer.Get(edsName)
	cc := newNoopTestClientConn()
	edsB, ok := builder.Build(cc, balancer.BuildOptions{Target: resolver.Target{Endpoint: testEDSClusterName}}).(*edsBalancer)
	if !ok {
		t.Fatalf("builder.Build(%s) returned type {%T}, want {*edsBalancer}", edsName, edsB)
	}
	defer edsB.Close()

	addrs := []resolver.Address{{Addr: "1.1.1.1:10001"}, {Addr: "2.2.2.2:10002"}, {Addr: "3.3.3.3:10003"}}
	for i := 0; i < 2; i++ {
		balancerName := fmt.Sprintf("balancer-%d", i)
		edsB.UpdateClientConnState(balancer.ClientConnState{
			ResolverState: resolver.State{Addresses: addrs},
			BalancerConfig: &EDSConfig{
				BalancerName:   balancerName,
				EDSServiceName: testEDSClusterName,
			},
		})

		xdsC := waitForNewXDSClientWithEDSWatch(t, xdsClientCh, balancerName)
		xdsC.InvokeWatchEDSCallback(&xdsclient.EDSUpdate{}, nil)
	}
}

const (
	fakeBalancerA = "fake_balancer_A"
	fakeBalancerB = "fake_balancer_B"
)

// Install two fake balancers for service config update tests.
//
// ParseConfig only accepts the json if the balancer specified is registered.

func init() {
	balancer.Register(&fakeBalancerBuilder{name: fakeBalancerA})
	balancer.Register(&fakeBalancerBuilder{name: fakeBalancerB})
}

type fakeBalancerBuilder struct {
	name string
}

func (b *fakeBalancerBuilder) Build(cc balancer.ClientConn, opts balancer.BuildOptions) balancer.Balancer {
	return &fakeBalancer{cc: cc}
}

func (b *fakeBalancerBuilder) Name() string {
	return b.name
}

type fakeBalancer struct {
	cc balancer.ClientConn
}

func (b *fakeBalancer) HandleResolvedAddrs(addrs []resolver.Address, err error) {
	panic("implement me")
}

func (b *fakeBalancer) HandleSubConnStateChange(sc balancer.SubConn, state connectivity.State) {
	panic("implement me")
}

func (b *fakeBalancer) Close() {}

// TestXDSConnfigChildPolicyUpdate verifies scenarios where the childPolicy
// section of the lbConfig is updated.
//
// The test does the following:
// * Builds a new xds balancer.
// * Pushes a new ClientConnState with a childPolicy set to fakeBalancerA.
//   Verifies that a new xdsClient is created. It then pushes a new edsUpdate
//   through the fakexds client. Verifies that a new edsLB is created and it
//   receives the expected childPolicy.
// * Pushes a new ClientConnState with a childPolicy set to fakeBalancerB.
//   This time around, we expect no new xdsClient or edsLB to be created.
//   Instead, we expect the existing edsLB to receive the new child policy.
func (s) TestXDSConnfigChildPolicyUpdate(t *testing.T) {
	edsLBCh := testutils.NewChannel()
	xdsClientCh := testutils.NewChannel()
	cancel := setup(edsLBCh, xdsClientCh)
	defer cancel()

	builder := balancer.Get(edsName)
	cc := newNoopTestClientConn()
	edsB, ok := builder.Build(cc, balancer.BuildOptions{Target: resolver.Target{Endpoint: testServiceName}}).(*edsBalancer)
	if !ok {
		t.Fatalf("builder.Build(%s) returned type {%T}, want {*edsBalancer}", edsName, edsB)
	}
	defer edsB.Close()

	edsB.UpdateClientConnState(balancer.ClientConnState{
		BalancerConfig: &EDSConfig{
			BalancerName: testBalancerNameFooBar,
			ChildPolicy: &loadBalancingConfig{
				Name:   fakeBalancerA,
				Config: json.RawMessage("{}"),
			},
			EDSServiceName: testEDSClusterName,
		},
	})
	xdsC := waitForNewXDSClientWithEDSWatch(t, xdsClientCh, testBalancerNameFooBar)
	xdsC.InvokeWatchEDSCallback(&xdsclient.EDSUpdate{}, nil)
	edsLB := waitForNewEDSLB(t, edsLBCh)
	edsLB.waitForChildPolicy(&loadBalancingConfig{
		Name:   string(fakeBalancerA),
		Config: json.RawMessage(`{}`),
	})

	edsB.UpdateClientConnState(balancer.ClientConnState{
		BalancerConfig: &EDSConfig{
			BalancerName: testBalancerNameFooBar,
			ChildPolicy: &loadBalancingConfig{
				Name:   fakeBalancerB,
				Config: json.RawMessage("{}"),
			},
			EDSServiceName: testEDSClusterName,
		},
	})
	edsLB.waitForChildPolicy(&loadBalancingConfig{
		Name:   string(fakeBalancerA),
		Config: json.RawMessage(`{}`),
	})
}

// TestXDSSubConnStateChange verifies if the top-level edsBalancer passes on
// the subConnStateChange to appropriate child balancers.
func (s) TestXDSSubConnStateChange(t *testing.T) {
	edsLBCh := testutils.NewChannel()
	xdsClientCh := testutils.NewChannel()
	cancel := setup(edsLBCh, xdsClientCh)
	defer cancel()

	builder := balancer.Get(edsName)
	cc := newNoopTestClientConn()
	edsB, ok := builder.Build(cc, balancer.BuildOptions{Target: resolver.Target{Endpoint: testEDSClusterName}}).(*edsBalancer)
	if !ok {
		t.Fatalf("builder.Build(%s) returned type {%T}, want {*edsBalancer}", edsName, edsB)
	}
	defer edsB.Close()

	addrs := []resolver.Address{{Addr: "1.1.1.1:10001"}, {Addr: "2.2.2.2:10002"}, {Addr: "3.3.3.3:10003"}}
	edsB.UpdateClientConnState(balancer.ClientConnState{
		ResolverState: resolver.State{Addresses: addrs},
		BalancerConfig: &EDSConfig{
			BalancerName:   testBalancerNameFooBar,
			EDSServiceName: testEDSClusterName,
		},
	})

	xdsC := waitForNewXDSClientWithEDSWatch(t, xdsClientCh, testBalancerNameFooBar)
	xdsC.InvokeWatchEDSCallback(&xdsclient.EDSUpdate{}, nil)
	edsLB := waitForNewEDSLB(t, edsLBCh)

	fsc := &fakeSubConn{}
	state := connectivity.Ready
	edsB.UpdateSubConnState(fsc, balancer.SubConnState{ConnectivityState: state})
	edsLB.waitForSubConnStateChange(&scStateChange{sc: fsc, state: state})
}

func TestXDSBalancerConfigParsing(t *testing.T) {
	const testEDSName = "eds.service"
	var testLRSName = "lrs.server"
	b := bytes.NewBuffer(nil)
	if err := (&jsonpb.Marshaler{}).Marshal(b, &scpb.XdsConfig{
		ChildPolicy: []*scpb.LoadBalancingConfig{
			{Policy: &scpb.LoadBalancingConfig_Xds{}},
			{Policy: &scpb.LoadBalancingConfig_RoundRobin{
				RoundRobin: &scpb.RoundRobinConfig{},
			}},
		},
		FallbackPolicy: []*scpb.LoadBalancingConfig{
			{Policy: &scpb.LoadBalancingConfig_Xds{}},
			{Policy: &scpb.LoadBalancingConfig_PickFirst{
				PickFirst: &scpb.PickFirstConfig{},
			}},
		},
		EdsServiceName:             testEDSName,
		LrsLoadReportingServerName: &wrapperspb.StringValue{Value: testLRSName},
	}); err != nil {
		t.Fatalf("%v", err)
	}

	tests := []struct {
		name    string
		js      json.RawMessage
		want    serviceconfig.LoadBalancingConfig
		wantErr bool
	}{
		{
			name: "jsonpb-generated",
			js:   b.Bytes(),
			want: &EDSConfig{
				ChildPolicy: &loadBalancingConfig{
					Name:   "round_robin",
					Config: json.RawMessage("{}"),
				},
				FallBackPolicy: &loadBalancingConfig{
					Name:   "pick_first",
					Config: json.RawMessage("{}"),
				},
				EDSServiceName:             testEDSName,
				LrsLoadReportingServerName: &testLRSName,
			},
			wantErr: false,
		},
		{
			// json with random balancers, and the first is not registered.
			name: "manually-generated",
			js: json.RawMessage(`
{
  "balancerName": "fake.foo.bar",
  "childPolicy": [
    {"fake_balancer_C": {}},
    {"fake_balancer_A": {}},
    {"fake_balancer_B": {}}
  ],
  "fallbackPolicy": [
    {"fake_balancer_C": {}},
    {"fake_balancer_B": {}},
    {"fake_balancer_A": {}}
  ],
  "edsServiceName": "eds.service",
  "lrsLoadReportingServerName": "lrs.server"
}`),
			want: &EDSConfig{
				BalancerName: "fake.foo.bar",
				ChildPolicy: &loadBalancingConfig{
					Name:   "fake_balancer_A",
					Config: json.RawMessage("{}"),
				},
				FallBackPolicy: &loadBalancingConfig{
					Name:   "fake_balancer_B",
					Config: json.RawMessage("{}"),
				},
				EDSServiceName:             testEDSName,
				LrsLoadReportingServerName: &testLRSName,
			},
			wantErr: false,
		},
		{
			// json with no lrs server name, LrsLoadReportingServerName should
			// be nil (not an empty string).
			name: "no-lrs-server-name",
			js: json.RawMessage(`
{
  "balancerName": "fake.foo.bar",
  "edsServiceName": "eds.service"
}`),
			want: &EDSConfig{
				BalancerName:               "fake.foo.bar",
				EDSServiceName:             testEDSName,
				LrsLoadReportingServerName: nil,
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := &edsBalancerBuilder{}
			got, err := b.ParseConfig(tt.js)
			if (err != nil) != tt.wantErr {
				t.Errorf("edsBalancerBuilder.ParseConfig() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !cmp.Equal(got, tt.want) {
				t.Errorf(cmp.Diff(got, tt.want))
			}
		})
	}
}
func TestLoadbalancingConfigParsing(t *testing.T) {
	tests := []struct {
		name string
		s    string
		want *EDSConfig
	}{
		{
			name: "empty",
			s:    "{}",
			want: &EDSConfig{},
		},
		{
			name: "success1",
			s:    `{"childPolicy":[{"pick_first":{}}]}`,
			want: &EDSConfig{
				ChildPolicy: &loadBalancingConfig{
					Name:   "pick_first",
					Config: json.RawMessage(`{}`),
				},
			},
		},
		{
			name: "success2",
			s:    `{"childPolicy":[{"round_robin":{}},{"pick_first":{}}]}`,
			want: &EDSConfig{
				ChildPolicy: &loadBalancingConfig{
					Name:   "round_robin",
					Config: json.RawMessage(`{}`),
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cfg EDSConfig
			if err := json.Unmarshal([]byte(tt.s), &cfg); err != nil || !reflect.DeepEqual(&cfg, tt.want) {
				t.Errorf("test name: %s, parseFullServiceConfig() = %+v, err: %v, want %+v, <nil>", tt.name, cfg, err, tt.want)
			}
		})
	}
}

func TestEqualStringPointers(t *testing.T) {
	var (
		ta1 = "test-a"
		ta2 = "test-a"
		tb  = "test-b"
	)
	tests := []struct {
		name string
		a    *string
		b    *string
		want bool
	}{
		{"both-nil", nil, nil, true},
		{"a-non-nil", &ta1, nil, false},
		{"b-non-nil", nil, &tb, false},
		{"equal", &ta1, &ta2, true},
		{"different", &ta1, &tb, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := equalStringPointers(tt.a, tt.b); got != tt.want {
				t.Errorf("equalStringPointers() = %v, want %v", got, tt.want)
			}
		})
	}
}
