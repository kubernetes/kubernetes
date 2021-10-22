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
	"crypto/tls"
	"fmt"
	"net"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/net/http2"
	"google.golang.org/grpc"
	_ "google.golang.org/grpc/balancer/grpclb"
	"google.golang.org/grpc/balancer/roundrobin"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/internal/channelz"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/resolver/manual"
	"google.golang.org/grpc/status"
	testpb "google.golang.org/grpc/test/grpc_testing"
	"google.golang.org/grpc/testdata"
)

func czCleanupWrapper(cleanup func() error, t *testing.T) {
	if err := cleanup(); err != nil {
		t.Error(err)
	}
}

func verifyResultWithDelay(f func() (bool, error)) error {
	var ok bool
	var err error
	for i := 0; i < 1000; i++ {
		if ok, err = f(); ok {
			return nil
		}
		time.Sleep(10 * time.Millisecond)
	}
	return err
}

func (s) TestCZServerRegistrationAndDeletion(t *testing.T) {
	testcases := []struct {
		total  int
		start  int64
		max    int64
		length int64
		end    bool
	}{
		{total: int(channelz.EntryPerPage), start: 0, max: 0, length: channelz.EntryPerPage, end: true},
		{total: int(channelz.EntryPerPage) - 1, start: 0, max: 0, length: channelz.EntryPerPage - 1, end: true},
		{total: int(channelz.EntryPerPage) + 1, start: 0, max: 0, length: channelz.EntryPerPage, end: false},
		{total: int(channelz.EntryPerPage) + 1, start: int64(2*(channelz.EntryPerPage+1) + 1), max: 0, length: 0, end: true},
		{total: int(channelz.EntryPerPage), start: 0, max: 1, length: 1, end: false},
		{total: int(channelz.EntryPerPage), start: 0, max: channelz.EntryPerPage - 1, length: channelz.EntryPerPage - 1, end: false},
	}

	for _, c := range testcases {
		czCleanup := channelz.NewChannelzStorage()
		defer czCleanupWrapper(czCleanup, t)
		e := tcpClearRREnv
		te := newTest(t, e)
		te.startServers(&testServer{security: e.security}, c.total)

		ss, end := channelz.GetServers(c.start, c.max)
		if int64(len(ss)) != c.length || end != c.end {
			t.Fatalf("GetServers(%d) = %+v (len of which: %d), end: %+v, want len(GetServers(%d)) = %d, end: %+v", c.start, ss, len(ss), end, c.start, c.length, c.end)
		}
		te.tearDown()
		ss, end = channelz.GetServers(c.start, c.max)
		if len(ss) != 0 || !end {
			t.Fatalf("GetServers(0) = %+v (len of which: %d), end: %+v, want len(GetServers(0)) = 0, end: true", ss, len(ss), end)
		}
	}
}

func (s) TestCZGetServer(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()

	ss, _ := channelz.GetServers(0, 0)
	if len(ss) != 1 {
		t.Fatalf("there should only be one server, not %d", len(ss))
	}

	serverID := ss[0].ID
	srv := channelz.GetServer(serverID)
	if srv == nil {
		t.Fatalf("server %d does not exist", serverID)
	}
	if srv.ID != serverID {
		t.Fatalf("server want id %d, but got %d", serverID, srv.ID)
	}

	te.tearDown()

	if err := verifyResultWithDelay(func() (bool, error) {
		srv := channelz.GetServer(serverID)
		if srv != nil {
			return false, fmt.Errorf("server %d should not exist", serverID)
		}

		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func (s) TestCZTopChannelRegistrationAndDeletion(t *testing.T) {
	testcases := []struct {
		total  int
		start  int64
		max    int64
		length int64
		end    bool
	}{
		{total: int(channelz.EntryPerPage), start: 0, max: 0, length: channelz.EntryPerPage, end: true},
		{total: int(channelz.EntryPerPage) - 1, start: 0, max: 0, length: channelz.EntryPerPage - 1, end: true},
		{total: int(channelz.EntryPerPage) + 1, start: 0, max: 0, length: channelz.EntryPerPage, end: false},
		{total: int(channelz.EntryPerPage) + 1, start: int64(2*(channelz.EntryPerPage+1) + 1), max: 0, length: 0, end: true},
		{total: int(channelz.EntryPerPage), start: 0, max: 1, length: 1, end: false},
		{total: int(channelz.EntryPerPage), start: 0, max: channelz.EntryPerPage - 1, length: channelz.EntryPerPage - 1, end: false},
	}

	for _, c := range testcases {
		czCleanup := channelz.NewChannelzStorage()
		defer czCleanupWrapper(czCleanup, t)
		e := tcpClearRREnv
		te := newTest(t, e)
		var ccs []*grpc.ClientConn
		for i := 0; i < c.total; i++ {
			cc := te.clientConn()
			te.cc = nil
			// avoid making next dial blocking
			te.srvAddr = ""
			ccs = append(ccs, cc)
		}
		if err := verifyResultWithDelay(func() (bool, error) {
			if tcs, end := channelz.GetTopChannels(c.start, c.max); int64(len(tcs)) != c.length || end != c.end {
				return false, fmt.Errorf("getTopChannels(%d) = %+v (len of which: %d), end: %+v, want len(GetTopChannels(%d)) = %d, end: %+v", c.start, tcs, len(tcs), end, c.start, c.length, c.end)
			}
			return true, nil
		}); err != nil {
			t.Fatal(err)
		}

		for _, cc := range ccs {
			cc.Close()
		}

		if err := verifyResultWithDelay(func() (bool, error) {
			if tcs, end := channelz.GetTopChannels(c.start, c.max); len(tcs) != 0 || !end {
				return false, fmt.Errorf("getTopChannels(0) = %+v (len of which: %d), end: %+v, want len(GetTopChannels(0)) = 0, end: true", tcs, len(tcs), end)
			}
			return true, nil
		}); err != nil {
			t.Fatal(err)
		}
		te.tearDown()
	}
}

func (s) TestCZTopChannelRegistrationAndDeletionWhenDialFail(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	// Make dial fails (due to no transport security specified)
	_, err := grpc.Dial("fake.addr")
	if err == nil {
		t.Fatal("expecting dial to fail")
	}
	if tcs, end := channelz.GetTopChannels(0, 0); tcs != nil || !end {
		t.Fatalf("GetTopChannels(0, 0) = %v, %v, want <nil>, true", tcs, end)
	}
}

func (s) TestCZNestedChannelRegistrationAndDeletion(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	// avoid calling API to set balancer type, which will void service config's change of balancer.
	e.balancer = ""
	te := newTest(t, e)
	r, cleanup := manual.GenerateAndRegisterManualResolver()
	defer cleanup()
	resolvedAddrs := []resolver.Address{{Addr: "127.0.0.1:0", Type: resolver.GRPCLB, ServerName: "grpclb.server"}}
	r.InitialState(resolver.State{Addresses: resolvedAddrs})
	te.resolverScheme = r.Scheme()
	te.clientConn()
	defer te.tearDown()

	if err := verifyResultWithDelay(func() (bool, error) {
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tcs))
		}
		if len(tcs[0].NestedChans) != 1 {
			return false, fmt.Errorf("there should be one nested channel from grpclb, not %d", len(tcs[0].NestedChans))
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "127.0.0.1:0"}}, ServiceConfig: parseCfg(r, `{"loadBalancingPolicy": "round_robin"}`)})

	// wait for the shutdown of grpclb balancer
	if err := verifyResultWithDelay(func() (bool, error) {
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tcs))
		}
		if len(tcs[0].NestedChans) != 0 {
			return false, fmt.Errorf("there should be 0 nested channel from grpclb, not %d", len(tcs[0].NestedChans))
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func (s) TestCZClientSubChannelSocketRegistrationAndDeletion(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	num := 3 // number of backends
	te := newTest(t, e)
	var svrAddrs []resolver.Address
	te.startServers(&testServer{security: e.security}, num)
	r, cleanup := manual.GenerateAndRegisterManualResolver()
	defer cleanup()
	for _, a := range te.srvAddrs {
		svrAddrs = append(svrAddrs, resolver.Address{Addr: a})
	}
	r.InitialState(resolver.State{Addresses: svrAddrs})
	te.resolverScheme = r.Scheme()
	te.clientConn()
	defer te.tearDown()
	// Here, we just wait for all sockets to be up. In the future, if we implement
	// IDLE, we may need to make several rpc calls to create the sockets.
	if err := verifyResultWithDelay(func() (bool, error) {
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tcs))
		}
		if len(tcs[0].SubChans) != num {
			return false, fmt.Errorf("there should be %d subchannel not %d", num, len(tcs[0].SubChans))
		}
		count := 0
		for k := range tcs[0].SubChans {
			sc := channelz.GetSubChannel(k)
			if sc == nil {
				return false, fmt.Errorf("got <nil> subchannel")
			}
			count += len(sc.Sockets)
		}
		if count != num {
			return false, fmt.Errorf("there should be %d sockets not %d", num, count)
		}

		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	r.UpdateState(resolver.State{Addresses: svrAddrs[:len(svrAddrs)-1]})

	if err := verifyResultWithDelay(func() (bool, error) {
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tcs))
		}
		if len(tcs[0].SubChans) != num-1 {
			return false, fmt.Errorf("there should be %d subchannel not %d", num-1, len(tcs[0].SubChans))
		}
		count := 0
		for k := range tcs[0].SubChans {
			sc := channelz.GetSubChannel(k)
			if sc == nil {
				return false, fmt.Errorf("got <nil> subchannel")
			}
			count += len(sc.Sockets)
		}
		if count != num-1 {
			return false, fmt.Errorf("there should be %d sockets not %d", num-1, count)
		}

		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func (s) TestCZServerSocketRegistrationAndDeletion(t *testing.T) {
	testcases := []struct {
		total  int
		start  int64
		max    int64
		length int64
		end    bool
	}{
		{total: int(channelz.EntryPerPage), start: 0, max: 0, length: channelz.EntryPerPage, end: true},
		{total: int(channelz.EntryPerPage) - 1, start: 0, max: 0, length: channelz.EntryPerPage - 1, end: true},
		{total: int(channelz.EntryPerPage) + 1, start: 0, max: 0, length: channelz.EntryPerPage, end: false},
		{total: int(channelz.EntryPerPage), start: 1, max: 0, length: channelz.EntryPerPage - 1, end: true},
		{total: int(channelz.EntryPerPage) + 1, start: channelz.EntryPerPage + 1, max: 0, length: 0, end: true},
		{total: int(channelz.EntryPerPage), start: 0, max: 1, length: 1, end: false},
		{total: int(channelz.EntryPerPage), start: 0, max: channelz.EntryPerPage - 1, length: channelz.EntryPerPage - 1, end: false},
	}

	for _, c := range testcases {
		czCleanup := channelz.NewChannelzStorage()
		defer czCleanupWrapper(czCleanup, t)
		e := tcpClearRREnv
		te := newTest(t, e)
		te.startServer(&testServer{security: e.security})
		var ccs []*grpc.ClientConn
		for i := 0; i < c.total; i++ {
			cc := te.clientConn()
			te.cc = nil
			ccs = append(ccs, cc)
		}

		var svrID int64
		if err := verifyResultWithDelay(func() (bool, error) {
			ss, _ := channelz.GetServers(0, 0)
			if len(ss) != 1 {
				return false, fmt.Errorf("there should only be one server, not %d", len(ss))
			}
			if len(ss[0].ListenSockets) != 1 {
				return false, fmt.Errorf("there should only be one server listen socket, not %d", len(ss[0].ListenSockets))
			}

			startID := c.start
			if startID != 0 {
				ns, _ := channelz.GetServerSockets(ss[0].ID, 0, int64(c.total))
				if int64(len(ns)) < c.start {
					return false, fmt.Errorf("there should more than %d sockets, not %d", len(ns), c.start)
				}
				startID = ns[c.start-1].ID + 1
			}

			ns, end := channelz.GetServerSockets(ss[0].ID, startID, c.max)
			if int64(len(ns)) != c.length || end != c.end {
				return false, fmt.Errorf("GetServerSockets(%d) = %+v (len of which: %d), end: %+v, want len(GetServerSockets(%d)) = %d, end: %+v", c.start, ns, len(ns), end, c.start, c.length, c.end)
			}

			svrID = ss[0].ID
			return true, nil
		}); err != nil {
			t.Fatal(err)
		}

		for _, cc := range ccs {
			cc.Close()
		}

		if err := verifyResultWithDelay(func() (bool, error) {
			ns, _ := channelz.GetServerSockets(svrID, c.start, c.max)
			if len(ns) != 0 {
				return false, fmt.Errorf("there should be %d normal sockets not %d", 0, len(ns))
			}
			return true, nil
		}); err != nil {
			t.Fatal(err)
		}
		te.tearDown()
	}
}

func (s) TestCZServerListenSocketDeletion(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	s := grpc.NewServer()
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	go s.Serve(lis)
	if err := verifyResultWithDelay(func() (bool, error) {
		ss, _ := channelz.GetServers(0, 0)
		if len(ss) != 1 {
			return false, fmt.Errorf("there should only be one server, not %d", len(ss))
		}
		if len(ss[0].ListenSockets) != 1 {
			return false, fmt.Errorf("there should only be one server listen socket, not %d", len(ss[0].ListenSockets))
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	lis.Close()
	if err := verifyResultWithDelay(func() (bool, error) {
		ss, _ := channelz.GetServers(0, 0)
		if len(ss) != 1 {
			return false, fmt.Errorf("there should be 1 server, not %d", len(ss))
		}
		if len(ss[0].ListenSockets) != 0 {
			return false, fmt.Errorf("there should only be %d server listen socket, not %d", 0, len(ss[0].ListenSockets))
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
	s.Stop()
}

type dummyChannel struct{}

func (d *dummyChannel) ChannelzMetric() *channelz.ChannelInternalMetric {
	return &channelz.ChannelInternalMetric{}
}

type dummySocket struct{}

func (d *dummySocket) ChannelzMetric() *channelz.SocketInternalMetric {
	return &channelz.SocketInternalMetric{}
}

func (s) TestCZRecusivelyDeletionOfEntry(t *testing.T) {
	//           +--+TopChan+---+
	//           |              |
	//           v              v
	//    +-+SubChan1+--+   SubChan2
	//    |             |
	//    v             v
	// Socket1       Socket2
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	topChanID := channelz.RegisterChannel(&dummyChannel{}, 0, "")
	subChanID1 := channelz.RegisterSubChannel(&dummyChannel{}, topChanID, "")
	subChanID2 := channelz.RegisterSubChannel(&dummyChannel{}, topChanID, "")
	sktID1 := channelz.RegisterNormalSocket(&dummySocket{}, subChanID1, "")
	sktID2 := channelz.RegisterNormalSocket(&dummySocket{}, subChanID1, "")

	tcs, _ := channelz.GetTopChannels(0, 0)
	if tcs == nil || len(tcs) != 1 {
		t.Fatalf("There should be one TopChannel entry")
	}
	if len(tcs[0].SubChans) != 2 {
		t.Fatalf("There should be two SubChannel entries")
	}
	sc := channelz.GetSubChannel(subChanID1)
	if sc == nil || len(sc.Sockets) != 2 {
		t.Fatalf("There should be two Socket entries")
	}

	channelz.RemoveEntry(topChanID)
	tcs, _ = channelz.GetTopChannels(0, 0)
	if tcs == nil || len(tcs) != 1 {
		t.Fatalf("There should be one TopChannel entry")
	}

	channelz.RemoveEntry(subChanID1)
	channelz.RemoveEntry(subChanID2)
	tcs, _ = channelz.GetTopChannels(0, 0)
	if tcs == nil || len(tcs) != 1 {
		t.Fatalf("There should be one TopChannel entry")
	}
	if len(tcs[0].SubChans) != 1 {
		t.Fatalf("There should be one SubChannel entry")
	}

	channelz.RemoveEntry(sktID1)
	channelz.RemoveEntry(sktID2)
	tcs, _ = channelz.GetTopChannels(0, 0)
	if tcs != nil {
		t.Fatalf("There should be no TopChannel entry")
	}
}

func (s) TestCZChannelMetrics(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	num := 3 // number of backends
	te := newTest(t, e)
	te.maxClientSendMsgSize = newInt(8)
	var svrAddrs []resolver.Address
	te.startServers(&testServer{security: e.security}, num)
	r, cleanup := manual.GenerateAndRegisterManualResolver()
	defer cleanup()
	for _, a := range te.srvAddrs {
		svrAddrs = append(svrAddrs, resolver.Address{Addr: a})
	}
	r.InitialState(resolver.State{Addresses: svrAddrs})
	te.resolverScheme = r.Scheme()
	cc := te.clientConn()
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(cc)
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}); err != nil {
		t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want _, <nil>", err)
	}

	const smallSize = 1
	const largeSize = 8

	largePayload, err := newPayload(testpb.PayloadType_COMPRESSABLE, largeSize)
	if err != nil {
		t.Fatal(err)
	}
	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE,
		ResponseSize: int32(smallSize),
		Payload:      largePayload,
	}

	if _, err := tc.UnaryCall(context.Background(), req); err == nil || status.Code(err) != codes.ResourceExhausted {
		t.Fatalf("TestService/UnaryCall(_, _) = _, %v, want _, error code: %s", err, codes.ResourceExhausted)
	}

	stream, err := tc.FullDuplexCall(context.Background())
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}
	defer stream.CloseSend()
	// Here, we just wait for all sockets to be up. In the future, if we implement
	// IDLE, we may need to make several rpc calls to create the sockets.
	if err := verifyResultWithDelay(func() (bool, error) {
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tcs))
		}
		if len(tcs[0].SubChans) != num {
			return false, fmt.Errorf("there should be %d subchannel not %d", num, len(tcs[0].SubChans))
		}
		var cst, csu, cf int64
		for k := range tcs[0].SubChans {
			sc := channelz.GetSubChannel(k)
			if sc == nil {
				return false, fmt.Errorf("got <nil> subchannel")
			}
			cst += sc.ChannelData.CallsStarted
			csu += sc.ChannelData.CallsSucceeded
			cf += sc.ChannelData.CallsFailed
		}
		if cst != 3 {
			return false, fmt.Errorf("there should be 3 CallsStarted not %d", cst)
		}
		if csu != 1 {
			return false, fmt.Errorf("there should be 1 CallsSucceeded not %d", csu)
		}
		if cf != 1 {
			return false, fmt.Errorf("there should be 1 CallsFailed not %d", cf)
		}
		if tcs[0].ChannelData.CallsStarted != 3 {
			return false, fmt.Errorf("there should be 3 CallsStarted not %d", tcs[0].ChannelData.CallsStarted)
		}
		if tcs[0].ChannelData.CallsSucceeded != 1 {
			return false, fmt.Errorf("there should be 1 CallsSucceeded not %d", tcs[0].ChannelData.CallsSucceeded)
		}
		if tcs[0].ChannelData.CallsFailed != 1 {
			return false, fmt.Errorf("there should be 1 CallsFailed not %d", tcs[0].ChannelData.CallsFailed)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func (s) TestCZServerMetrics(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	te := newTest(t, e)
	te.maxServerReceiveMsgSize = newInt(8)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}); err != nil {
		t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want _, <nil>", err)
	}

	const smallSize = 1
	const largeSize = 8

	largePayload, err := newPayload(testpb.PayloadType_COMPRESSABLE, largeSize)
	if err != nil {
		t.Fatal(err)
	}
	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE,
		ResponseSize: int32(smallSize),
		Payload:      largePayload,
	}
	if _, err := tc.UnaryCall(context.Background(), req); err == nil || status.Code(err) != codes.ResourceExhausted {
		t.Fatalf("TestService/UnaryCall(_, _) = _, %v, want _, error code: %s", err, codes.ResourceExhausted)
	}

	stream, err := tc.FullDuplexCall(context.Background())
	if err != nil {
		t.Fatalf("%v.FullDuplexCall(_) = _, %v, want <nil>", tc, err)
	}
	defer stream.CloseSend()

	if err := verifyResultWithDelay(func() (bool, error) {
		ss, _ := channelz.GetServers(0, 0)
		if len(ss) != 1 {
			return false, fmt.Errorf("there should only be one server, not %d", len(ss))
		}
		if ss[0].ServerData.CallsStarted != 3 {
			return false, fmt.Errorf("there should be 3 CallsStarted not %d", ss[0].ServerData.CallsStarted)
		}
		if ss[0].ServerData.CallsSucceeded != 1 {
			return false, fmt.Errorf("there should be 1 CallsSucceeded not %d", ss[0].ServerData.CallsSucceeded)
		}
		if ss[0].ServerData.CallsFailed != 1 {
			return false, fmt.Errorf("there should be 1 CallsFailed not %d", ss[0].ServerData.CallsFailed)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

type testServiceClientWrapper struct {
	testpb.TestServiceClient
	mu             sync.RWMutex
	streamsCreated int
}

func (t *testServiceClientWrapper) getCurrentStreamID() uint32 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return uint32(2*t.streamsCreated - 1)
}

func (t *testServiceClientWrapper) EmptyCall(ctx context.Context, in *testpb.Empty, opts ...grpc.CallOption) (*testpb.Empty, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.streamsCreated++
	return t.TestServiceClient.EmptyCall(ctx, in, opts...)
}

func (t *testServiceClientWrapper) UnaryCall(ctx context.Context, in *testpb.SimpleRequest, opts ...grpc.CallOption) (*testpb.SimpleResponse, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.streamsCreated++
	return t.TestServiceClient.UnaryCall(ctx, in, opts...)
}

func (t *testServiceClientWrapper) StreamingOutputCall(ctx context.Context, in *testpb.StreamingOutputCallRequest, opts ...grpc.CallOption) (testpb.TestService_StreamingOutputCallClient, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.streamsCreated++
	return t.TestServiceClient.StreamingOutputCall(ctx, in, opts...)
}

func (t *testServiceClientWrapper) StreamingInputCall(ctx context.Context, opts ...grpc.CallOption) (testpb.TestService_StreamingInputCallClient, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.streamsCreated++
	return t.TestServiceClient.StreamingInputCall(ctx, opts...)
}

func (t *testServiceClientWrapper) FullDuplexCall(ctx context.Context, opts ...grpc.CallOption) (testpb.TestService_FullDuplexCallClient, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.streamsCreated++
	return t.TestServiceClient.FullDuplexCall(ctx, opts...)
}

func (t *testServiceClientWrapper) HalfDuplexCall(ctx context.Context, opts ...grpc.CallOption) (testpb.TestService_HalfDuplexCallClient, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.streamsCreated++
	return t.TestServiceClient.HalfDuplexCall(ctx, opts...)
}

func doSuccessfulUnaryCall(tc testpb.TestServiceClient, t *testing.T) {
	if _, err := tc.EmptyCall(context.Background(), &testpb.Empty{}); err != nil {
		t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want _, <nil>", err)
	}
}

func doStreamingInputCallWithLargePayload(tc testpb.TestServiceClient, t *testing.T) {
	s, err := tc.StreamingInputCall(context.Background())
	if err != nil {
		t.Fatalf("TestService/StreamingInputCall(_) = _, %v, want <nil>", err)
	}
	payload, err := newPayload(testpb.PayloadType_COMPRESSABLE, 10000)
	if err != nil {
		t.Fatal(err)
	}
	s.Send(&testpb.StreamingInputCallRequest{Payload: payload})
}

func doServerSideFailedUnaryCall(tc testpb.TestServiceClient, t *testing.T) {
	const smallSize = 1
	const largeSize = 2000

	largePayload, err := newPayload(testpb.PayloadType_COMPRESSABLE, largeSize)
	if err != nil {
		t.Fatal(err)
	}
	req := &testpb.SimpleRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE,
		ResponseSize: int32(smallSize),
		Payload:      largePayload,
	}
	if _, err := tc.UnaryCall(context.Background(), req); err == nil || status.Code(err) != codes.ResourceExhausted {
		t.Fatalf("TestService/UnaryCall(_, _) = _, %v, want _, error code: %s", err, codes.ResourceExhausted)
	}
}

func doClientSideInitiatedFailedStream(tc testpb.TestServiceClient, t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := tc.FullDuplexCall(ctx)
	if err != nil {
		t.Fatalf("TestService/FullDuplexCall(_) = _, %v, want <nil>", err)
	}

	const smallSize = 1
	smallPayload, err := newPayload(testpb.PayloadType_COMPRESSABLE, smallSize)
	if err != nil {
		t.Fatal(err)
	}

	sreq := &testpb.StreamingOutputCallRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE,
		ResponseParameters: []*testpb.ResponseParameters{
			{Size: smallSize},
		},
		Payload: smallPayload,
	}

	if err := stream.Send(sreq); err != nil {
		t.Fatalf("%v.Send(%v) = %v, want <nil>", stream, sreq, err)
	}
	if _, err := stream.Recv(); err != nil {
		t.Fatalf("%v.Recv() = %v, want <nil>", stream, err)
	}
	// By canceling the call, the client will send rst_stream to end the call, and
	// the stream will failed as a result.
	cancel()
}

// This func is to be used to test client side counting of failed streams.
func doServerSideInitiatedFailedStreamWithRSTStream(tc testpb.TestServiceClient, t *testing.T, l *listenerWrapper) {
	stream, err := tc.FullDuplexCall(context.Background())
	if err != nil {
		t.Fatalf("TestService/FullDuplexCall(_) = _, %v, want <nil>", err)
	}

	const smallSize = 1
	smallPayload, err := newPayload(testpb.PayloadType_COMPRESSABLE, smallSize)
	if err != nil {
		t.Fatal(err)
	}

	sreq := &testpb.StreamingOutputCallRequest{
		ResponseType: testpb.PayloadType_COMPRESSABLE,
		ResponseParameters: []*testpb.ResponseParameters{
			{Size: smallSize},
		},
		Payload: smallPayload,
	}

	if err := stream.Send(sreq); err != nil {
		t.Fatalf("%v.Send(%v) = %v, want <nil>", stream, sreq, err)
	}
	if _, err := stream.Recv(); err != nil {
		t.Fatalf("%v.Recv() = %v, want <nil>", stream, err)
	}

	rcw := l.getLastConn()

	if rcw != nil {
		rcw.writeRSTStream(tc.(*testServiceClientWrapper).getCurrentStreamID(), http2.ErrCodeCancel)
	}
	if _, err := stream.Recv(); err == nil {
		t.Fatalf("%v.Recv() = %v, want <non-nil>", stream, err)
	}
}

// this func is to be used to test client side counting of failed streams.
func doServerSideInitiatedFailedStreamWithGoAway(tc testpb.TestServiceClient, t *testing.T, l *listenerWrapper) {
	// This call is just to keep the transport from shutting down (socket will be deleted
	// in this case, and we will not be able to get metrics).
	s, err := tc.FullDuplexCall(context.Background())
	if err != nil {
		t.Fatalf("TestService/FullDuplexCall(_) = _, %v, want <nil>", err)
	}
	if err := s.Send(&testpb.StreamingOutputCallRequest{ResponseParameters: []*testpb.ResponseParameters{
		{
			Size: 1,
		},
	}}); err != nil {
		t.Fatalf("s.Send() failed with error: %v", err)
	}
	if _, err := s.Recv(); err != nil {
		t.Fatalf("s.Recv() failed with error: %v", err)
	}

	s, err = tc.FullDuplexCall(context.Background())
	if err != nil {
		t.Fatalf("TestService/FullDuplexCall(_) = _, %v, want <nil>", err)
	}
	if err := s.Send(&testpb.StreamingOutputCallRequest{ResponseParameters: []*testpb.ResponseParameters{
		{
			Size: 1,
		},
	}}); err != nil {
		t.Fatalf("s.Send() failed with error: %v", err)
	}
	if _, err := s.Recv(); err != nil {
		t.Fatalf("s.Recv() failed with error: %v", err)
	}

	rcw := l.getLastConn()
	if rcw != nil {
		rcw.writeGoAway(tc.(*testServiceClientWrapper).getCurrentStreamID()-2, http2.ErrCodeCancel, []byte{})
	}
	if _, err := s.Recv(); err == nil {
		t.Fatalf("%v.Recv() = %v, want <non-nil>", s, err)
	}
}

func doIdleCallToInvokeKeepAlive(tc testpb.TestServiceClient, t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	_, err := tc.FullDuplexCall(ctx)
	if err != nil {
		t.Fatalf("TestService/FullDuplexCall(_) = _, %v, want <nil>", err)
	}
	// Allow for at least 2 keepalives (1s per ping interval)
	time.Sleep(4 * time.Second)
	cancel()
}

func (s) TestCZClientSocketMetricsStreamsAndMessagesCount(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	te := newTest(t, e)
	te.maxServerReceiveMsgSize = newInt(20)
	te.maxClientReceiveMsgSize = newInt(20)
	rcw := te.startServerWithConnControl(&testServer{security: e.security})
	defer te.tearDown()
	cc := te.clientConn()
	tc := &testServiceClientWrapper{TestServiceClient: testpb.NewTestServiceClient(cc)}

	doSuccessfulUnaryCall(tc, t)
	var scID, skID int64
	if err := verifyResultWithDelay(func() (bool, error) {
		tchan, _ := channelz.GetTopChannels(0, 0)
		if len(tchan) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tchan))
		}
		if len(tchan[0].SubChans) != 1 {
			return false, fmt.Errorf("there should only be one subchannel under top channel %d, not %d", tchan[0].ID, len(tchan[0].SubChans))
		}

		for scID = range tchan[0].SubChans {
			break
		}
		sc := channelz.GetSubChannel(scID)
		if sc == nil {
			return false, fmt.Errorf("there should only be one socket under subchannel %d, not 0", scID)
		}
		if len(sc.Sockets) != 1 {
			return false, fmt.Errorf("there should only be one socket under subchannel %d, not %d", sc.ID, len(sc.Sockets))
		}
		for skID = range sc.Sockets {
			break
		}
		skt := channelz.GetSocket(skID)
		sktData := skt.SocketData
		if sktData.StreamsStarted != 1 || sktData.StreamsSucceeded != 1 || sktData.MessagesSent != 1 || sktData.MessagesReceived != 1 {
			return false, fmt.Errorf("channelz.GetSocket(%d), want (StreamsStarted, StreamsSucceeded, MessagesSent, MessagesReceived) = (1, 1, 1, 1), got (%d, %d, %d, %d)", skt.ID, sktData.StreamsStarted, sktData.StreamsSucceeded, sktData.MessagesSent, sktData.MessagesReceived)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	doServerSideFailedUnaryCall(tc, t)
	if err := verifyResultWithDelay(func() (bool, error) {
		skt := channelz.GetSocket(skID)
		sktData := skt.SocketData
		if sktData.StreamsStarted != 2 || sktData.StreamsSucceeded != 2 || sktData.MessagesSent != 2 || sktData.MessagesReceived != 1 {
			return false, fmt.Errorf("channelz.GetSocket(%d), want (StreamsStarted, StreamsSucceeded, MessagesSent, MessagesReceived) = (2, 2, 2, 1), got (%d, %d, %d, %d)", skt.ID, sktData.StreamsStarted, sktData.StreamsSucceeded, sktData.MessagesSent, sktData.MessagesReceived)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	doClientSideInitiatedFailedStream(tc, t)
	if err := verifyResultWithDelay(func() (bool, error) {
		skt := channelz.GetSocket(skID)
		sktData := skt.SocketData
		if sktData.StreamsStarted != 3 || sktData.StreamsSucceeded != 2 || sktData.StreamsFailed != 1 || sktData.MessagesSent != 3 || sktData.MessagesReceived != 2 {
			return false, fmt.Errorf("channelz.GetSocket(%d), want (StreamsStarted, StreamsSucceeded, StreamsFailed, MessagesSent, MessagesReceived) = (3, 2, 1, 3, 2), got (%d, %d, %d, %d, %d)", skt.ID, sktData.StreamsStarted, sktData.StreamsSucceeded, sktData.StreamsFailed, sktData.MessagesSent, sktData.MessagesReceived)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	doServerSideInitiatedFailedStreamWithRSTStream(tc, t, rcw)
	if err := verifyResultWithDelay(func() (bool, error) {
		skt := channelz.GetSocket(skID)
		sktData := skt.SocketData
		if sktData.StreamsStarted != 4 || sktData.StreamsSucceeded != 2 || sktData.StreamsFailed != 2 || sktData.MessagesSent != 4 || sktData.MessagesReceived != 3 {
			return false, fmt.Errorf("channelz.GetSocket(%d), want (StreamsStarted, StreamsSucceeded, StreamsFailed, MessagesSent, MessagesReceived) = (4, 2, 2, 4, 3), got (%d, %d, %d, %d, %d)", skt.ID, sktData.StreamsStarted, sktData.StreamsSucceeded, sktData.StreamsFailed, sktData.MessagesSent, sktData.MessagesReceived)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	doServerSideInitiatedFailedStreamWithGoAway(tc, t, rcw)
	if err := verifyResultWithDelay(func() (bool, error) {
		skt := channelz.GetSocket(skID)
		sktData := skt.SocketData
		if sktData.StreamsStarted != 6 || sktData.StreamsSucceeded != 2 || sktData.StreamsFailed != 3 || sktData.MessagesSent != 6 || sktData.MessagesReceived != 5 {
			return false, fmt.Errorf("channelz.GetSocket(%d), want (StreamsStarted, StreamsSucceeded, StreamsFailed, MessagesSent, MessagesReceived) = (6, 2, 3, 6, 5), got (%d, %d, %d, %d, %d)", skt.ID, sktData.StreamsStarted, sktData.StreamsSucceeded, sktData.StreamsFailed, sktData.MessagesSent, sktData.MessagesReceived)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

// This test is to complete TestCZClientSocketMetricsStreamsAndMessagesCount and
// TestCZServerSocketMetricsStreamsAndMessagesCount by adding the test case of
// server sending RST_STREAM to client due to client side flow control violation.
// It is separated from other cases due to setup incompatibly, i.e. max receive
// size violation will mask flow control violation.
func (s) TestCZClientAndServerSocketMetricsStreamsCountFlowControlRSTStream(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	te := newTest(t, e)
	te.serverInitialWindowSize = 65536
	// Avoid overflowing connection level flow control window, which will lead to
	// transport being closed.
	te.serverInitialConnWindowSize = 65536 * 2
	ts := &funcServer{fullDuplexCall: func(stream testpb.TestService_FullDuplexCallServer) error {
		stream.Send(&testpb.StreamingOutputCallResponse{})
		<-stream.Context().Done()
		return status.Errorf(codes.DeadlineExceeded, "deadline exceeded or cancelled")
	}}
	te.startServer(ts)
	defer te.tearDown()
	cc, dw := te.clientConnWithConnControl()
	tc := &testServiceClientWrapper{TestServiceClient: testpb.NewTestServiceClient(cc)}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	stream, err := tc.FullDuplexCall(ctx)
	if err != nil {
		t.Fatalf("TestService/FullDuplexCall(_) = _, %v, want <nil>", err)
	}
	if _, err := stream.Recv(); err != nil {
		t.Fatalf("stream.Recv() = %v, want nil", err)
	}
	go func() {
		payload := make([]byte, 16384)
		for i := 0; i < 6; i++ {
			dw.getRawConnWrapper().writeRawFrame(http2.FrameData, 0, tc.getCurrentStreamID(), payload)
		}
	}()
	if _, err := stream.Recv(); status.Code(err) != codes.ResourceExhausted {
		t.Fatalf("stream.Recv() = %v, want error code: %v", err, codes.ResourceExhausted)
	}
	cancel()

	if err := verifyResultWithDelay(func() (bool, error) {
		tchan, _ := channelz.GetTopChannels(0, 0)
		if len(tchan) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tchan))
		}
		if len(tchan[0].SubChans) != 1 {
			return false, fmt.Errorf("there should only be one subchannel under top channel %d, not %d", tchan[0].ID, len(tchan[0].SubChans))
		}
		var id int64
		for id = range tchan[0].SubChans {
			break
		}
		sc := channelz.GetSubChannel(id)
		if sc == nil {
			return false, fmt.Errorf("there should only be one socket under subchannel %d, not 0", id)
		}
		if len(sc.Sockets) != 1 {
			return false, fmt.Errorf("there should only be one socket under subchannel %d, not %d", sc.ID, len(sc.Sockets))
		}
		for id = range sc.Sockets {
			break
		}
		skt := channelz.GetSocket(id)
		sktData := skt.SocketData
		if sktData.StreamsStarted != 1 || sktData.StreamsSucceeded != 0 || sktData.StreamsFailed != 1 {
			return false, fmt.Errorf("channelz.GetSocket(%d), want (StreamsStarted, StreamsSucceeded, StreamsFailed) = (1, 0, 1), got (%d, %d, %d)", skt.ID, sktData.StreamsStarted, sktData.StreamsSucceeded, sktData.StreamsFailed)
		}
		ss, _ := channelz.GetServers(0, 0)
		if len(ss) != 1 {
			return false, fmt.Errorf("there should only be one server, not %d", len(ss))
		}

		ns, _ := channelz.GetServerSockets(ss[0].ID, 0, 0)
		if len(ns) != 1 {
			return false, fmt.Errorf("there should be one server normal socket, not %d", len(ns))
		}
		sktData = ns[0].SocketData
		if sktData.StreamsStarted != 1 || sktData.StreamsSucceeded != 0 || sktData.StreamsFailed != 1 {
			return false, fmt.Errorf("server socket metric with ID %d, want (StreamsStarted, StreamsSucceeded, StreamsFailed) = (1, 0, 1), got (%d, %d, %d)", ns[0].ID, sktData.StreamsStarted, sktData.StreamsSucceeded, sktData.StreamsFailed)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func (s) TestCZClientAndServerSocketMetricsFlowControl(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	te := newTest(t, e)
	// disable BDP
	te.serverInitialWindowSize = 65536
	te.serverInitialConnWindowSize = 65536
	te.clientInitialWindowSize = 65536
	te.clientInitialConnWindowSize = 65536
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)

	for i := 0; i < 10; i++ {
		doSuccessfulUnaryCall(tc, t)
	}

	var cliSktID, svrSktID int64
	if err := verifyResultWithDelay(func() (bool, error) {
		tchan, _ := channelz.GetTopChannels(0, 0)
		if len(tchan) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tchan))
		}
		if len(tchan[0].SubChans) != 1 {
			return false, fmt.Errorf("there should only be one subchannel under top channel %d, not %d", tchan[0].ID, len(tchan[0].SubChans))
		}
		var id int64
		for id = range tchan[0].SubChans {
			break
		}
		sc := channelz.GetSubChannel(id)
		if sc == nil {
			return false, fmt.Errorf("there should only be one socket under subchannel %d, not 0", id)
		}
		if len(sc.Sockets) != 1 {
			return false, fmt.Errorf("there should only be one socket under subchannel %d, not %d", sc.ID, len(sc.Sockets))
		}
		for id = range sc.Sockets {
			break
		}
		skt := channelz.GetSocket(id)
		sktData := skt.SocketData
		// 65536 - 5 (Length-Prefixed-Message size) * 10 = 65486
		if sktData.LocalFlowControlWindow != 65486 || sktData.RemoteFlowControlWindow != 65486 {
			return false, fmt.Errorf("client: (LocalFlowControlWindow, RemoteFlowControlWindow) size should be (65536, 65486), not (%d, %d)", sktData.LocalFlowControlWindow, sktData.RemoteFlowControlWindow)
		}
		ss, _ := channelz.GetServers(0, 0)
		if len(ss) != 1 {
			return false, fmt.Errorf("there should only be one server, not %d", len(ss))
		}
		ns, _ := channelz.GetServerSockets(ss[0].ID, 0, 0)
		sktData = ns[0].SocketData
		if sktData.LocalFlowControlWindow != 65486 || sktData.RemoteFlowControlWindow != 65486 {
			return false, fmt.Errorf("server: (LocalFlowControlWindow, RemoteFlowControlWindow) size should be (65536, 65486), not (%d, %d)", sktData.LocalFlowControlWindow, sktData.RemoteFlowControlWindow)
		}
		cliSktID, svrSktID = id, ss[0].ID
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	doStreamingInputCallWithLargePayload(tc, t)

	if err := verifyResultWithDelay(func() (bool, error) {
		skt := channelz.GetSocket(cliSktID)
		sktData := skt.SocketData
		// Local: 65536 - 5 (Length-Prefixed-Message size) * 10 = 65486
		// Remote: 65536 - 5 (Length-Prefixed-Message size) * 10 - 10011 = 55475
		if sktData.LocalFlowControlWindow != 65486 || sktData.RemoteFlowControlWindow != 55475 {
			return false, fmt.Errorf("client: (LocalFlowControlWindow, RemoteFlowControlWindow) size should be (65486, 55475), not (%d, %d)", sktData.LocalFlowControlWindow, sktData.RemoteFlowControlWindow)
		}
		ss, _ := channelz.GetServers(0, 0)
		if len(ss) != 1 {
			return false, fmt.Errorf("there should only be one server, not %d", len(ss))
		}
		ns, _ := channelz.GetServerSockets(svrSktID, 0, 0)
		sktData = ns[0].SocketData
		if sktData.LocalFlowControlWindow != 55475 || sktData.RemoteFlowControlWindow != 65486 {
			return false, fmt.Errorf("server: (LocalFlowControlWindow, RemoteFlowControlWindow) size should be (55475, 65486), not (%d, %d)", sktData.LocalFlowControlWindow, sktData.RemoteFlowControlWindow)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	// triggers transport flow control window update on server side, since unacked
	// bytes should be larger than limit now. i.e. 50 + 20022 > 65536/4.
	doStreamingInputCallWithLargePayload(tc, t)
	if err := verifyResultWithDelay(func() (bool, error) {
		skt := channelz.GetSocket(cliSktID)
		sktData := skt.SocketData
		// Local: 65536 - 5 (Length-Prefixed-Message size) * 10 = 65486
		// Remote: 65536
		if sktData.LocalFlowControlWindow != 65486 || sktData.RemoteFlowControlWindow != 65536 {
			return false, fmt.Errorf("client: (LocalFlowControlWindow, RemoteFlowControlWindow) size should be (65486, 65536), not (%d, %d)", sktData.LocalFlowControlWindow, sktData.RemoteFlowControlWindow)
		}
		ss, _ := channelz.GetServers(0, 0)
		if len(ss) != 1 {
			return false, fmt.Errorf("there should only be one server, not %d", len(ss))
		}
		ns, _ := channelz.GetServerSockets(svrSktID, 0, 0)
		sktData = ns[0].SocketData
		if sktData.LocalFlowControlWindow != 65536 || sktData.RemoteFlowControlWindow != 65486 {
			return false, fmt.Errorf("server: (LocalFlowControlWindow, RemoteFlowControlWindow) size should be (65536, 65486), not (%d, %d)", sktData.LocalFlowControlWindow, sktData.RemoteFlowControlWindow)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func (s) TestCZClientSocketMetricsKeepAlive(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	defer func(t time.Duration) { internal.KeepaliveMinPingTime = t }(internal.KeepaliveMinPingTime)
	internal.KeepaliveMinPingTime = time.Second
	e := tcpClearRREnv
	te := newTest(t, e)
	te.customDialOptions = append(te.customDialOptions, grpc.WithKeepaliveParams(
		keepalive.ClientParameters{
			Time:                time.Second,
			Timeout:             500 * time.Millisecond,
			PermitWithoutStream: true,
		}))
	te.customServerOptions = append(te.customServerOptions, grpc.KeepaliveEnforcementPolicy(
		keepalive.EnforcementPolicy{
			MinTime:             500 * time.Millisecond,
			PermitWithoutStream: true,
		}))
	te.startServer(&testServer{security: e.security})
	te.clientConn() // Dial the server
	defer te.tearDown()
	if err := verifyResultWithDelay(func() (bool, error) {
		tchan, _ := channelz.GetTopChannels(0, 0)
		if len(tchan) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tchan))
		}
		if len(tchan[0].SubChans) != 1 {
			return false, fmt.Errorf("there should only be one subchannel under top channel %d, not %d", tchan[0].ID, len(tchan[0].SubChans))
		}
		var id int64
		for id = range tchan[0].SubChans {
			break
		}
		sc := channelz.GetSubChannel(id)
		if sc == nil {
			return false, fmt.Errorf("there should only be one socket under subchannel %d, not 0", id)
		}
		if len(sc.Sockets) != 1 {
			return false, fmt.Errorf("there should only be one socket under subchannel %d, not %d", sc.ID, len(sc.Sockets))
		}
		for id = range sc.Sockets {
			break
		}
		skt := channelz.GetSocket(id)
		if skt.SocketData.KeepAlivesSent != 2 {
			return false, fmt.Errorf("there should be 2 KeepAlives sent, not %d", skt.SocketData.KeepAlivesSent)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func (s) TestCZServerSocketMetricsStreamsAndMessagesCount(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	te := newTest(t, e)
	te.maxServerReceiveMsgSize = newInt(20)
	te.maxClientReceiveMsgSize = newInt(20)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	cc, _ := te.clientConnWithConnControl()
	tc := &testServiceClientWrapper{TestServiceClient: testpb.NewTestServiceClient(cc)}

	var svrID int64
	if err := verifyResultWithDelay(func() (bool, error) {
		ss, _ := channelz.GetServers(0, 0)
		if len(ss) != 1 {
			return false, fmt.Errorf("there should only be one server, not %d", len(ss))
		}
		svrID = ss[0].ID
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	doSuccessfulUnaryCall(tc, t)
	if err := verifyResultWithDelay(func() (bool, error) {
		ns, _ := channelz.GetServerSockets(svrID, 0, 0)
		sktData := ns[0].SocketData
		if sktData.StreamsStarted != 1 || sktData.StreamsSucceeded != 1 || sktData.StreamsFailed != 0 || sktData.MessagesSent != 1 || sktData.MessagesReceived != 1 {
			return false, fmt.Errorf("server socket metric with ID %d, want (StreamsStarted, StreamsSucceeded, MessagesSent, MessagesReceived) = (1, 1, 1, 1), got (%d, %d, %d, %d, %d)", ns[0].ID, sktData.StreamsStarted, sktData.StreamsSucceeded, sktData.StreamsFailed, sktData.MessagesSent, sktData.MessagesReceived)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	doServerSideFailedUnaryCall(tc, t)
	if err := verifyResultWithDelay(func() (bool, error) {
		ns, _ := channelz.GetServerSockets(svrID, 0, 0)
		sktData := ns[0].SocketData
		if sktData.StreamsStarted != 2 || sktData.StreamsSucceeded != 2 || sktData.StreamsFailed != 0 || sktData.MessagesSent != 1 || sktData.MessagesReceived != 1 {
			return false, fmt.Errorf("server socket metric with ID %d, want (StreamsStarted, StreamsSucceeded, StreamsFailed, MessagesSent, MessagesReceived) = (2, 2, 0, 1, 1), got (%d, %d, %d, %d, %d)", ns[0].ID, sktData.StreamsStarted, sktData.StreamsSucceeded, sktData.StreamsFailed, sktData.MessagesSent, sktData.MessagesReceived)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	doClientSideInitiatedFailedStream(tc, t)
	if err := verifyResultWithDelay(func() (bool, error) {
		ns, _ := channelz.GetServerSockets(svrID, 0, 0)
		sktData := ns[0].SocketData
		if sktData.StreamsStarted != 3 || sktData.StreamsSucceeded != 2 || sktData.StreamsFailed != 1 || sktData.MessagesSent != 2 || sktData.MessagesReceived != 2 {
			return false, fmt.Errorf("server socket metric with ID %d, want (StreamsStarted, StreamsSucceeded, StreamsFailed, MessagesSent, MessagesReceived) = (3, 2, 1, 2, 2), got (%d, %d, %d, %d, %d)", ns[0].ID, sktData.StreamsStarted, sktData.StreamsSucceeded, sktData.StreamsFailed, sktData.MessagesSent, sktData.MessagesReceived)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func (s) TestCZServerSocketMetricsKeepAlive(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	te := newTest(t, e)
	// We setup the server keepalive parameters to send one keepalive every
	// second, and verify that the actual number of keepalives is very close to
	// the number of seconds elapsed in the test.  We had a bug wherein the
	// server was sending one keepalive every [Time+Timeout] instead of every
	// [Time] period, and since Timeout is configured to a low value here, we
	// should be able to verify that the fix works with the above mentioned
	// logic.
	kpOption := grpc.KeepaliveParams(keepalive.ServerParameters{
		Time:    time.Second,
		Timeout: 100 * time.Millisecond,
	})
	te.customServerOptions = append(te.customServerOptions, kpOption)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	cc := te.clientConn()
	tc := testpb.NewTestServiceClient(cc)
	start := time.Now()
	doIdleCallToInvokeKeepAlive(tc, t)

	if err := verifyResultWithDelay(func() (bool, error) {
		ss, _ := channelz.GetServers(0, 0)
		if len(ss) != 1 {
			return false, fmt.Errorf("there should be one server, not %d", len(ss))
		}
		ns, _ := channelz.GetServerSockets(ss[0].ID, 0, 0)
		if len(ns) != 1 {
			return false, fmt.Errorf("there should be one server normal socket, not %d", len(ns))
		}
		wantKeepalivesCount := int64(time.Since(start).Seconds()) - 1
		if gotKeepalivesCount := ns[0].SocketData.KeepAlivesSent; gotKeepalivesCount != wantKeepalivesCount {
			return false, fmt.Errorf("got keepalivesCount: %v, want keepalivesCount: %v", gotKeepalivesCount, wantKeepalivesCount)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

var cipherSuites = []string{
	"TLS_RSA_WITH_RC4_128_SHA",
	"TLS_RSA_WITH_3DES_EDE_CBC_SHA",
	"TLS_RSA_WITH_AES_128_CBC_SHA",
	"TLS_RSA_WITH_AES_256_CBC_SHA",
	"TLS_RSA_WITH_AES_128_GCM_SHA256",
	"TLS_RSA_WITH_AES_256_GCM_SHA384",
	"TLS_ECDHE_ECDSA_WITH_RC4_128_SHA",
	"TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA",
	"TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA",
	"TLS_ECDHE_RSA_WITH_RC4_128_SHA",
	"TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA",
	"TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA",
	"TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA",
	"TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
	"TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
	"TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
	"TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
	"TLS_FALLBACK_SCSV",
	"TLS_RSA_WITH_AES_128_CBC_SHA256",
	"TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256",
	"TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256",
	"TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305",
	"TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305",
	"TLS_AES_128_GCM_SHA256",
	"TLS_AES_256_GCM_SHA384",
	"TLS_CHACHA20_POLY1305_SHA256",
}

func (s) TestCZSocketGetSecurityValueTLS(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpTLSRREnv
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	defer te.tearDown()
	te.clientConn()
	if err := verifyResultWithDelay(func() (bool, error) {
		tchan, _ := channelz.GetTopChannels(0, 0)
		if len(tchan) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tchan))
		}
		if len(tchan[0].SubChans) != 1 {
			return false, fmt.Errorf("there should only be one subchannel under top channel %d, not %d", tchan[0].ID, len(tchan[0].SubChans))
		}
		var id int64
		for id = range tchan[0].SubChans {
			break
		}
		sc := channelz.GetSubChannel(id)
		if sc == nil {
			return false, fmt.Errorf("there should only be one socket under subchannel %d, not 0", id)
		}
		if len(sc.Sockets) != 1 {
			return false, fmt.Errorf("there should only be one socket under subchannel %d, not %d", sc.ID, len(sc.Sockets))
		}
		for id = range sc.Sockets {
			break
		}
		skt := channelz.GetSocket(id)
		cert, _ := tls.LoadX509KeyPair(testdata.Path("server1.pem"), testdata.Path("server1.key"))
		securityVal, ok := skt.SocketData.Security.(*credentials.TLSChannelzSecurityValue)
		if !ok {
			return false, fmt.Errorf("the SocketData.Security is of type: %T, want: *credentials.TLSChannelzSecurityValue", skt.SocketData.Security)
		}
		if !reflect.DeepEqual(securityVal.RemoteCertificate, cert.Certificate[0]) {
			return false, fmt.Errorf("SocketData.Security.RemoteCertificate got: %v, want: %v", securityVal.RemoteCertificate, cert.Certificate[0])
		}
		for _, v := range cipherSuites {
			if v == securityVal.StandardName {
				return true, nil
			}
		}
		return false, fmt.Errorf("SocketData.Security.StandardName got: %v, want it to be one of %v", securityVal.StandardName, cipherSuites)
	}); err != nil {
		t.Fatal(err)
	}
}

func (s) TestCZChannelTraceCreationDeletion(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	// avoid calling API to set balancer type, which will void service config's change of balancer.
	e.balancer = ""
	te := newTest(t, e)
	r, cleanup := manual.GenerateAndRegisterManualResolver()
	defer cleanup()
	resolvedAddrs := []resolver.Address{{Addr: "127.0.0.1:0", Type: resolver.GRPCLB, ServerName: "grpclb.server"}}
	r.InitialState(resolver.State{Addresses: resolvedAddrs})
	te.resolverScheme = r.Scheme()
	te.clientConn()
	defer te.tearDown()
	var nestedConn int64
	if err := verifyResultWithDelay(func() (bool, error) {
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tcs))
		}
		if len(tcs[0].NestedChans) != 1 {
			return false, fmt.Errorf("there should be one nested channel from grpclb, not %d", len(tcs[0].NestedChans))
		}
		for k := range tcs[0].NestedChans {
			nestedConn = k
		}
		for _, e := range tcs[0].Trace.Events {
			if e.RefID == nestedConn && e.RefType != channelz.RefChannel {
				return false, fmt.Errorf("nested channel trace event shoud have RefChannel as RefType")
			}
		}
		ncm := channelz.GetChannel(nestedConn)
		if ncm.Trace == nil {
			return false, fmt.Errorf("trace for nested channel should not be empty")
		}
		if len(ncm.Trace.Events) == 0 {
			return false, fmt.Errorf("there should be at least one trace event for nested channel not 0")
		}
		if ncm.Trace.Events[0].Desc != "Channel Created" {
			return false, fmt.Errorf("the first trace event should be \"Channel Created\", not %q", ncm.Trace.Events[0].Desc)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "127.0.0.1:0"}}, ServiceConfig: parseCfg(r, `{"loadBalancingPolicy": "round_robin"}`)})

	// wait for the shutdown of grpclb balancer
	if err := verifyResultWithDelay(func() (bool, error) {
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tcs))
		}
		if len(tcs[0].NestedChans) != 0 {
			return false, fmt.Errorf("there should be 0 nested channel from grpclb, not %d", len(tcs[0].NestedChans))
		}
		ncm := channelz.GetChannel(nestedConn)
		if ncm == nil {
			return false, fmt.Errorf("nested channel should still exist due to parent's trace reference")
		}
		if ncm.Trace == nil {
			return false, fmt.Errorf("trace for nested channel should not be empty")
		}
		if len(ncm.Trace.Events) == 0 {
			return false, fmt.Errorf("there should be at least one trace event for nested channel not 0")
		}
		if ncm.Trace.Events[len(ncm.Trace.Events)-1].Desc != "Channel Deleted" {
			return false, fmt.Errorf("the first trace event should be \"Channel Deleted\", not %q", ncm.Trace.Events[0].Desc)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func (s) TestCZSubChannelTraceCreationDeletion(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	r, cleanup := manual.GenerateAndRegisterManualResolver()
	defer cleanup()
	r.InitialState(resolver.State{Addresses: []resolver.Address{{Addr: te.srvAddr}}})
	te.resolverScheme = r.Scheme()
	te.clientConn()
	defer te.tearDown()
	var subConn int64
	// Here, we just wait for all sockets to be up. In the future, if we implement
	// IDLE, we may need to make several rpc calls to create the sockets.
	if err := verifyResultWithDelay(func() (bool, error) {
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tcs))
		}
		if len(tcs[0].SubChans) != 1 {
			return false, fmt.Errorf("there should be 1 subchannel not %d", len(tcs[0].SubChans))
		}
		for k := range tcs[0].SubChans {
			subConn = k
		}
		for _, e := range tcs[0].Trace.Events {
			if e.RefID == subConn && e.RefType != channelz.RefSubChannel {
				return false, fmt.Errorf("subchannel trace event shoud have RefType to be RefSubChannel")
			}
		}
		scm := channelz.GetSubChannel(subConn)
		if scm == nil {
			return false, fmt.Errorf("subChannel does not exist")
		}
		if scm.Trace == nil {
			return false, fmt.Errorf("trace for subChannel should not be empty")
		}
		if len(scm.Trace.Events) == 0 {
			return false, fmt.Errorf("there should be at least one trace event for subChannel not 0")
		}
		if scm.Trace.Events[0].Desc != "Subchannel Created" {
			return false, fmt.Errorf("the first trace event should be \"Subchannel Created\", not %q", scm.Trace.Events[0].Desc)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	// Wait for ready
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	for src := te.cc.GetState(); src != connectivity.Ready; src = te.cc.GetState() {
		if !te.cc.WaitForStateChange(ctx, src) {
			t.Fatalf("timed out waiting for state change.  got %v; want %v", src, connectivity.Ready)
		}
	}
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "fake address"}}})
	// Wait for not-ready.
	for src := te.cc.GetState(); src == connectivity.Ready; src = te.cc.GetState() {
		if !te.cc.WaitForStateChange(ctx, src) {
			t.Fatalf("timed out waiting for state change.  got %v; want !%v", src, connectivity.Ready)
		}
	}

	if err := verifyResultWithDelay(func() (bool, error) {
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tcs))
		}
		if len(tcs[0].SubChans) != 1 {
			return false, fmt.Errorf("there should be 1 subchannel not %d", len(tcs[0].SubChans))
		}
		scm := channelz.GetSubChannel(subConn)
		if scm == nil {
			return false, fmt.Errorf("subChannel should still exist due to parent's trace reference")
		}
		if scm.Trace == nil {
			return false, fmt.Errorf("trace for SubChannel should not be empty")
		}
		if len(scm.Trace.Events) == 0 {
			return false, fmt.Errorf("there should be at least one trace event for subChannel not 0")
		}
		if got, want := scm.Trace.Events[len(scm.Trace.Events)-1].Desc, "Subchannel Deleted"; got != want {
			return false, fmt.Errorf("the last trace event should be %q, not %q", want, got)
		}

		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func (s) TestCZChannelAddressResolutionChange(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	e.balancer = ""
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	r, cleanup := manual.GenerateAndRegisterManualResolver()
	defer cleanup()
	addrs := []resolver.Address{{Addr: te.srvAddr}}
	r.InitialState(resolver.State{Addresses: addrs})
	te.resolverScheme = r.Scheme()
	te.clientConn()
	defer te.tearDown()
	var cid int64
	// Here, we just wait for all sockets to be up. In the future, if we implement
	// IDLE, we may need to make several rpc calls to create the sockets.
	if err := verifyResultWithDelay(func() (bool, error) {
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tcs))
		}
		cid = tcs[0].ID
		for i := len(tcs[0].Trace.Events) - 1; i >= 0; i-- {
			if strings.Contains(tcs[0].Trace.Events[i].Desc, "resolver returned new addresses") {
				break
			}
			if i == 0 {
				return false, fmt.Errorf("events do not contain expected address resolution from empty address state.  Got: %+v", tcs[0].Trace.Events)
			}
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
	r.UpdateState(resolver.State{Addresses: addrs, ServiceConfig: parseCfg(r, `{"loadBalancingPolicy": "round_robin"}`)})

	if err := verifyResultWithDelay(func() (bool, error) {
		cm := channelz.GetChannel(cid)
		for i := len(cm.Trace.Events) - 1; i >= 0; i-- {
			if cm.Trace.Events[i].Desc == fmt.Sprintf("Channel switches to new LB policy %q", roundrobin.Name) {
				break
			}
			if i == 0 {
				return false, fmt.Errorf("events do not contain expected address resolution change of LB policy")
			}
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	newSC := parseCfg(r, `{
    "methodConfig": [
        {
            "name": [
                {
                    "service": "grpc.testing.TestService",
                    "method": "EmptyCall"
                }
            ],
            "waitForReady": false,
            "timeout": ".001s"
        }
    ]
}`)
	r.UpdateState(resolver.State{Addresses: addrs, ServiceConfig: newSC})

	if err := verifyResultWithDelay(func() (bool, error) {
		cm := channelz.GetChannel(cid)

		var es []string
		for i := len(cm.Trace.Events) - 1; i >= 0; i-- {
			if strings.Contains(cm.Trace.Events[i].Desc, "service config updated") {
				break
			}
			es = append(es, cm.Trace.Events[i].Desc)
			if i == 0 {
				return false, fmt.Errorf("events do not contain expected address resolution of new service config\n Events:\n%v", strings.Join(es, "\n"))
			}
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	r.UpdateState(resolver.State{Addresses: []resolver.Address{}, ServiceConfig: newSC})

	if err := verifyResultWithDelay(func() (bool, error) {
		cm := channelz.GetChannel(cid)
		for i := len(cm.Trace.Events) - 1; i >= 0; i-- {
			if strings.Contains(cm.Trace.Events[i].Desc, "resolver returned an empty address list") {
				break
			}
			if i == 0 {
				return false, fmt.Errorf("events do not contain expected address resolution of empty address")
			}
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func (s) TestCZSubChannelPickedNewAddress(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	e.balancer = ""
	te := newTest(t, e)
	te.startServers(&testServer{security: e.security}, 3)
	r, cleanup := manual.GenerateAndRegisterManualResolver()
	defer cleanup()
	var svrAddrs []resolver.Address
	for _, a := range te.srvAddrs {
		svrAddrs = append(svrAddrs, resolver.Address{Addr: a})
	}
	r.InitialState(resolver.State{Addresses: svrAddrs})
	te.resolverScheme = r.Scheme()
	cc := te.clientConn()
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(cc)
	// make sure the connection is up
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, err := tc.EmptyCall(ctx, &testpb.Empty{}); err != nil {
		t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want _, <nil>", err)
	}
	te.srvs[0].Stop()
	te.srvs[1].Stop()
	// Here, we just wait for all sockets to be up. In the future, if we implement
	// IDLE, we may need to make several rpc calls to create the sockets.
	if err := verifyResultWithDelay(func() (bool, error) {
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tcs))
		}
		if len(tcs[0].SubChans) != 1 {
			return false, fmt.Errorf("there should be 1 subchannel not %d", len(tcs[0].SubChans))
		}
		var subConn int64
		for k := range tcs[0].SubChans {
			subConn = k
		}
		scm := channelz.GetSubChannel(subConn)
		if scm.Trace == nil {
			return false, fmt.Errorf("trace for SubChannel should not be empty")
		}
		if len(scm.Trace.Events) == 0 {
			return false, fmt.Errorf("there should be at least one trace event for subChannel not 0")
		}
		for i := len(scm.Trace.Events) - 1; i >= 0; i-- {
			if scm.Trace.Events[i].Desc == fmt.Sprintf("Subchannel picks a new address %q to connect", te.srvAddrs[2]) {
				break
			}
			if i == 0 {
				return false, fmt.Errorf("events do not contain expected address resolution of subchannel picked new address")
			}
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func (s) TestCZSubChannelConnectivityState(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	r, cleanup := manual.GenerateAndRegisterManualResolver()
	defer cleanup()
	r.InitialState(resolver.State{Addresses: []resolver.Address{{Addr: te.srvAddr}}})
	te.resolverScheme = r.Scheme()
	cc := te.clientConn()
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(cc)
	// make sure the connection is up
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, err := tc.EmptyCall(ctx, &testpb.Empty{}); err != nil {
		t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want _, <nil>", err)
	}
	var subConn int64
	te.srv.Stop()

	if err := verifyResultWithDelay(func() (bool, error) {
		// we need to obtain the SubChannel id before it gets deleted from Channel's children list (due
		// to effect of r.UpdateState(resolver.State{Addresses:[]resolver.Address{}}))
		if subConn == 0 {
			tcs, _ := channelz.GetTopChannels(0, 0)
			if len(tcs) != 1 {
				return false, fmt.Errorf("there should only be one top channel, not %d", len(tcs))
			}
			if len(tcs[0].SubChans) != 1 {
				return false, fmt.Errorf("there should be 1 subchannel not %d", len(tcs[0].SubChans))
			}
			for k := range tcs[0].SubChans {
				// get the SubChannel id for further trace inquiry.
				subConn = k
			}
		}
		scm := channelz.GetSubChannel(subConn)
		if scm == nil {
			return false, fmt.Errorf("subChannel should still exist due to parent's trace reference")
		}
		if scm.Trace == nil {
			return false, fmt.Errorf("trace for SubChannel should not be empty")
		}
		if len(scm.Trace.Events) == 0 {
			return false, fmt.Errorf("there should be at least one trace event for subChannel not 0")
		}
		var ready, connecting, transient, shutdown int
		for _, e := range scm.Trace.Events {
			if e.Desc == fmt.Sprintf("Subchannel Connectivity change to %v", connectivity.TransientFailure) {
				transient++
			}
		}
		// Make sure the SubChannel has already seen transient failure before shutting it down through
		// r.UpdateState(resolver.State{Addresses:[]resolver.Address{}}).
		if transient == 0 {
			return false, fmt.Errorf("transient failure has not happened on SubChannel yet")
		}
		transient = 0
		r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "fake address"}}})
		for _, e := range scm.Trace.Events {
			if e.Desc == fmt.Sprintf("Subchannel Connectivity change to %v", connectivity.Ready) {
				ready++
			}
			if e.Desc == fmt.Sprintf("Subchannel Connectivity change to %v", connectivity.Connecting) {
				connecting++
			}
			if e.Desc == fmt.Sprintf("Subchannel Connectivity change to %v", connectivity.TransientFailure) {
				transient++
			}
			if e.Desc == fmt.Sprintf("Subchannel Connectivity change to %v", connectivity.Shutdown) {
				shutdown++
			}
		}
		// example:
		// Subchannel Created
		// Subchannel's connectivity state changed to CONNECTING
		// Subchannel picked a new address: "localhost:36011"
		// Subchannel's connectivity state changed to READY
		// Subchannel's connectivity state changed to TRANSIENT_FAILURE
		// Subchannel's connectivity state changed to CONNECTING
		// Subchannel picked a new address: "localhost:36011"
		// Subchannel's connectivity state changed to SHUTDOWN
		// Subchannel Deleted
		if ready != 1 || connecting < 1 || transient < 1 || shutdown != 1 {
			return false, fmt.Errorf("got: ready = %d, connecting = %d, transient = %d, shutdown = %d, want: 1, >=1, >=1, 1", ready, connecting, transient, shutdown)
		}

		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func (s) TestCZChannelConnectivityState(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	r, cleanup := manual.GenerateAndRegisterManualResolver()
	defer cleanup()
	r.InitialState(resolver.State{Addresses: []resolver.Address{{Addr: te.srvAddr}}})
	te.resolverScheme = r.Scheme()
	cc := te.clientConn()
	defer te.tearDown()
	tc := testpb.NewTestServiceClient(cc)
	// make sure the connection is up
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, err := tc.EmptyCall(ctx, &testpb.Empty{}); err != nil {
		t.Fatalf("TestService/EmptyCall(_, _) = _, %v, want _, <nil>", err)
	}
	te.srv.Stop()
	if err := verifyResultWithDelay(func() (bool, error) {
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tcs))
		}

		var ready, connecting, transient int
		for _, e := range tcs[0].Trace.Events {
			if e.Desc == fmt.Sprintf("Channel Connectivity change to %v", connectivity.Ready) {
				ready++
			}
			if e.Desc == fmt.Sprintf("Channel Connectivity change to %v", connectivity.Connecting) {
				connecting++
			}
			if e.Desc == fmt.Sprintf("Channel Connectivity change to %v", connectivity.TransientFailure) {
				transient++
			}
		}

		// example:
		// Channel Created
		// Adressses resolved (from empty address state): "localhost:40467"
		// SubChannel (id: 4[]) Created
		// Channel's connectivity state changed to CONNECTING
		// Channel's connectivity state changed to READY
		// Channel's connectivity state changed to TRANSIENT_FAILURE
		// Channel's connectivity state changed to CONNECTING
		// Channel's connectivity state changed to TRANSIENT_FAILURE
		if ready != 1 || connecting < 1 || transient < 1 {
			return false, fmt.Errorf("got: ready = %d, connecting = %d, transient = %d, want: 1, >=1, >=1", ready, connecting, transient)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func (s) TestCZTraceOverwriteChannelDeletion(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	// avoid newTest using WithBalancer, which would override service config's change of balancer below.
	e.balancer = ""
	te := newTest(t, e)
	channelz.SetMaxTraceEntry(1)
	defer channelz.ResetMaxTraceEntryToDefault()
	r, cleanup := manual.GenerateAndRegisterManualResolver()
	defer cleanup()
	resolvedAddrs := []resolver.Address{{Addr: "127.0.0.1:0", Type: resolver.GRPCLB, ServerName: "grpclb.server"}}
	r.InitialState(resolver.State{Addresses: resolvedAddrs})
	te.resolverScheme = r.Scheme()
	te.clientConn()
	defer te.tearDown()
	var nestedConn int64
	if err := verifyResultWithDelay(func() (bool, error) {
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tcs))
		}
		if len(tcs[0].NestedChans) != 1 {
			return false, fmt.Errorf("there should be one nested channel from grpclb, not %d", len(tcs[0].NestedChans))
		}
		for k := range tcs[0].NestedChans {
			nestedConn = k
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "127.0.0.1:0"}}, ServiceConfig: parseCfg(r, `{"loadBalancingPolicy": "round_robin"}`)})

	// wait for the shutdown of grpclb balancer
	if err := verifyResultWithDelay(func() (bool, error) {
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tcs))
		}
		if len(tcs[0].NestedChans) != 0 {
			return false, fmt.Errorf("there should be 0 nested channel from grpclb, not %d", len(tcs[0].NestedChans))
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	// verify that the nested channel no longer exist due to trace referencing it got overwritten.
	if err := verifyResultWithDelay(func() (bool, error) {
		cm := channelz.GetChannel(nestedConn)
		if cm != nil {
			return false, fmt.Errorf("nested channel should have been deleted since its parent's trace should not contain any reference to it anymore")
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func (s) TestCZTraceOverwriteSubChannelDeletion(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	te := newTest(t, e)
	channelz.SetMaxTraceEntry(1)
	defer channelz.ResetMaxTraceEntryToDefault()
	te.startServer(&testServer{security: e.security})
	r, cleanup := manual.GenerateAndRegisterManualResolver()
	defer cleanup()
	r.InitialState(resolver.State{Addresses: []resolver.Address{{Addr: te.srvAddr}}})
	te.resolverScheme = r.Scheme()
	te.clientConn()
	defer te.tearDown()
	var subConn int64
	// Here, we just wait for all sockets to be up. In the future, if we implement
	// IDLE, we may need to make several rpc calls to create the sockets.
	if err := verifyResultWithDelay(func() (bool, error) {
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tcs))
		}
		if len(tcs[0].SubChans) != 1 {
			return false, fmt.Errorf("there should be 1 subchannel not %d", len(tcs[0].SubChans))
		}
		for k := range tcs[0].SubChans {
			subConn = k
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	// Wait for ready
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	for src := te.cc.GetState(); src != connectivity.Ready; src = te.cc.GetState() {
		if !te.cc.WaitForStateChange(ctx, src) {
			t.Fatalf("timed out waiting for state change.  got %v; want %v", src, connectivity.Ready)
		}
	}
	r.UpdateState(resolver.State{Addresses: []resolver.Address{{Addr: "fake address"}}})
	// Wait for not-ready.
	for src := te.cc.GetState(); src == connectivity.Ready; src = te.cc.GetState() {
		if !te.cc.WaitForStateChange(ctx, src) {
			t.Fatalf("timed out waiting for state change.  got %v; want !%v", src, connectivity.Ready)
		}
	}

	// verify that the subchannel no longer exist due to trace referencing it got overwritten.
	if err := verifyResultWithDelay(func() (bool, error) {
		cm := channelz.GetChannel(subConn)
		if cm != nil {
			return false, fmt.Errorf("subchannel should have been deleted since its parent's trace should not contain any reference to it anymore")
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}

func (s) TestCZTraceTopChannelDeletionTraceClear(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer czCleanupWrapper(czCleanup, t)
	e := tcpClearRREnv
	te := newTest(t, e)
	te.startServer(&testServer{security: e.security})
	r, cleanup := manual.GenerateAndRegisterManualResolver()
	defer cleanup()
	r.InitialState(resolver.State{Addresses: []resolver.Address{{Addr: te.srvAddr}}})
	te.resolverScheme = r.Scheme()
	te.clientConn()
	var subConn int64
	// Here, we just wait for all sockets to be up. In the future, if we implement
	// IDLE, we may need to make several rpc calls to create the sockets.
	if err := verifyResultWithDelay(func() (bool, error) {
		tcs, _ := channelz.GetTopChannels(0, 0)
		if len(tcs) != 1 {
			return false, fmt.Errorf("there should only be one top channel, not %d", len(tcs))
		}
		if len(tcs[0].SubChans) != 1 {
			return false, fmt.Errorf("there should be 1 subchannel not %d", len(tcs[0].SubChans))
		}
		for k := range tcs[0].SubChans {
			subConn = k
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
	te.tearDown()
	// verify that the subchannel no longer exist due to parent channel got deleted and its trace cleared.
	if err := verifyResultWithDelay(func() (bool, error) {
		cm := channelz.GetChannel(subConn)
		if cm != nil {
			return false, fmt.Errorf("subchannel should have been deleted since its parent's trace should not contain any reference to it anymore")
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
}
